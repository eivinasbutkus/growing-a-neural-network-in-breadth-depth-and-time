"""Re-evaluate saved final pruning masks on the full test set, averaged over noise seeds.

Background: in pruning._is_threshold_hit the accuracy recorded for each search step is the
1024-image fast-check value (even when the full-set check overrode a fast-check failure), and
fast-check passes are never confirmed on the full set. The "after" accuracies shown in figures
therefore come from 1024 images, while "before" comes from the full test set.

This script does NOT rerun the search. For every model it:
  1. loads the checkpoint and computes the full-set baseline accuracy over N noise seeds,
  2. applies the saved final mask (last search step with threshold_hit == False),
  3. computes the full-set accuracy of the pruned model over the same N noise seeds,
and writes pruning_reeval_thresh{thresh}.csv next to the existing pruning files (never overwrites
the original files). A combined summary is written to results/pruning_reeval_summary.csv.

Usage:
    python scripts/reeval_pruning_masks.py --configs config_breadth_depth_time_v7.yaml config_datasets_v4_train.yaml
    python scripts/reeval_pruning_masks.py --configs config_datasets_v4_train.yaml --limit 1   # quick test
    # SLURM array: one shard per task, then aggregate once all tasks finish
    python scripts/reeval_pruning_masks.py --shard $SLURM_ARRAY_TASK_ID --n_shards 20
    python scripts/reeval_pruning_masks.py --aggregate_only
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import nn_bowl
from nn_bowl.evaluation.pruning import _zero_channel_rank


def full_accuracy(model, test_loader, seed):
    torch.manual_seed(seed)
    return nn_bowl.utils.get_test_accuracy(model, test_loader)  # full test set


def load_model(cfg, device):
    model = nn_bowl.Model(cfg).to(device)
    try:
        path = nn_bowl.utils.get_checkpoint_path(cfg, cfg.train.n_epochs - 1)
        model = nn_bowl.utils.load_checkpoint(model, path, compile_clean=True)
    except FileNotFoundError:
        path = nn_bowl.utils.get_last_checkpoint_path(cfg)
        model = nn_bowl.utils.load_checkpoint(model, path, compile_clean=True)
    model.eval()
    return model


def reeval_one(cfg, combo, n_seeds, device, overwrite=False):
    thresh = cfg.analysis.pruning_accuracy_threshold
    results_dir = nn_bowl.utils.get_results_dir(cfg)
    results_path = results_dir / f"pruning_results_thresh{thresh}.csv"
    masks_path = results_dir / f"pruning_masks_thresh{thresh}.npy"
    out_path = results_dir / f"pruning_reeval_thresh{thresh}.csv"

    if not results_path.exists() or not masks_path.exists():
        return None
    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path).iloc[0].to_dict()

    old_df = pd.read_csv(results_path)
    passed = old_df[~old_df["threshold_hit"]]
    mask_index = int(passed.iloc[-1]["mask_index"])
    mask = np.load(masks_path)[mask_index]  # [n_layers, n_channels] in channel-rank order
    old_baseline = float(old_df[old_df["phase"] == "baseline"]["accuracy"].values[0])
    old_final = float(passed.iloc[-1]["accuracy"])

    model = load_model(cfg, device)
    test_loader = nn_bowl.utils.get_dataloaders(cfg)["test"]

    # channel ranking computed on the unpruned model, exactly as in the original search
    cost_manager = nn_bowl.CostManager(cfg, model).to(device)
    _, _, layer_channel_indices = cost_manager.costs["space"].get_space_costs()

    seeds = list(range(n_seeds))
    base_accs = [full_accuracy(model, test_loader, s) for s in seeds]

    with torch.no_grad():
        for layer_index, rank in zip(*np.where(~mask)):
            _zero_channel_rank(model, [int(layer_index)], layer_channel_indices, int(rank))
    final_accs = [full_accuracy(model, test_loader, s) for s in seeds]

    chance = 1.0 / model.n_classes
    base_mean = float(np.mean(base_accs))
    final_mean = float(np.mean(final_accs))
    thr_new = chance + thresh * (base_mean - chance)
    thr_old = chance + thresh * (old_baseline - chance)

    row = {
        **{f"combo.{k}": v for k, v in combo.items()},
        "experiment": cfg.experiment.name,
        "n_seeds": n_seeds,
        "mask_index": mask_index,
        "layers_used": int((mask.sum(axis=1) > 0).sum()),
        "channels_total": int(mask.sum()),
        "old_baseline_acc": old_baseline,
        "old_final_acc_1024": old_final,
        "baseline_acc_mean": base_mean,
        "baseline_acc_std": float(np.std(base_accs)),
        "final_acc_mean": final_mean,
        "final_acc_std": float(np.std(final_accs)),
        "threshold_acc": thr_new,
        "threshold_acc_old_baseline": thr_old,
        "passes": final_mean >= thr_new,
        "passes_old_baseline": final_mean >= thr_old,
        "frac_above_chance_kept": (final_mean - chance) / (base_mean - chance),
    }
    pd.DataFrame([row]).to_csv(out_path, index=False)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["config_breadth_depth_time_v7.yaml", "config_datasets_v4_train.yaml"])
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="max models per config (for testing)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--filter", nargs="*", default=[], help="only combos matching key=value, e.g. costs.time.lam=0.1 instance=0")
    parser.add_argument("--shard", type=int, default=0, help="this shard's index (e.g. $SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--n_shards", type=int, default=1, help="total shards; models are split round-robin")
    parser.add_argument("--aggregate_only", action="store_true", help="only collect existing per-model reeval CSVs into the summary")
    args = parser.parse_args()

    device = nn_bowl.utils.get_device()
    rows = []
    for config_name in args.configs:
        cfg_base = nn_bowl.utils.load_config(config_name)
        combos = nn_bowl.utils.get_combinations(cfg_base)
        n_done = 0
        for i, combo in enumerate(combos):
            if args.limit is not None and n_done >= args.limit:
                break
            if any(str(combo.get(k)) != v for k, v in (f.split("=", 1) for f in args.filter)):
                continue
            cfg = nn_bowl.utils.get_cfg(cfg_base, combo)
            if args.aggregate_only:
                path = nn_bowl.utils.get_results_dir(cfg) / f"pruning_reeval_thresh{cfg.analysis.pruning_accuracy_threshold}.csv"
                if path.exists():
                    rows.append(pd.read_csv(path).iloc[0].to_dict())
                continue
            if i % args.n_shards != args.shard:
                continue
            t0 = time.time()
            try:
                row = reeval_one(cfg, combo, args.n_seeds, device, overwrite=args.overwrite)
            except Exception as e:
                print(f"[{config_name} {i+1}/{len(combos)}] ERROR {combo}: {e}", flush=True)
                continue
            if row is None:
                continue
            n_done += 1
            rows.append(row)
            print(f"[{config_name} {i+1}/{len(combos)}] {time.time()-t0:5.1f}s "
                  f"base {row['baseline_acc_mean']:.4f} final {row['final_acc_mean']:.4f} "
                  f"(old 1024: {row['old_final_acc_1024']:.4f}) thr {row['threshold_acc']:.4f} "
                  f"pass={row['passes']}", flush=True)

    if rows and args.n_shards == 1:
        df = pd.DataFrame(rows)
        out = Path(nn_bowl.paths.RESULTS_DIR) / "pruning_reeval_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nWrote {len(df)} rows to {out}")
        for exp, g in df.groupby("experiment"):
            fails = g[~g["passes"]]
            print(f"{exp}: {len(g)} models, {len(fails)} below threshold; "
                  f"worst kept fraction {g['frac_above_chance_kept'].min():.3f}; "
                  f"median shortfall of failures {(fails['threshold_acc'] - fails['final_acc_mean']).median() if len(fails) else 0:.4f}")


if __name__ == "__main__":
    main()
