"""Table 7 (App O): resources used by CIFAR-10 breadth-depth models (lam_time = 0) at three pruning thresholds.

Same definitions as notebooks/final_notebooks/01_breadth_depth.ipynb (cell "add_layers_channels_used"):
  layers used   = number of layers with at least one surviving channel
  channels used = surviving channels / layers used
For each model instance, values are averaged over the six values of the other spatial pressure;
entries are mean (std, ddof=1) across instances.

Usage:
    python scripts/make_table7.py            # v2 pruning files (current)
    python scripts/make_table7.py --v1       # original files, to check against the submitted Table 7
"""
import argparse

import numpy as np
import pandas as pd

import nn_bowl

THRESHOLDS = [0.99, 0.98, 0.95]


def pruning_paths(cfg, v1):
    if not v1:
        return nn_bowl.utils.get_pruning_paths(cfg)
    d, t = nn_bowl.utils.get_results_dir(cfg), cfg.analysis.pruning_accuracy_threshold
    return d / f"pruning_results_thresh{t}.csv", d / f"pruning_masks_thresh{t}.npy"


def load_resources(cfg_base, v1):
    rows, missing = [], []
    for thr in THRESHOLDS:
        cfg_base.analysis.pruning_accuracy_threshold = thr
        for combo in nn_bowl.utils.get_combinations(cfg_base):
            if combo["costs.time.lam"] != 0.0:
                continue
            cfg = nn_bowl.utils.get_cfg(cfg_base, combo)
            results_path, masks_path = pruning_paths(cfg, v1)
            if not (results_path.exists() and masks_path.exists()):
                missing.append((thr, combo))
                continue
            df = pd.read_csv(results_path)
            mask = np.load(masks_path)[int(df[~df["threshold_hit"]].iloc[-1]["mask_index"])]
            layers_used = int((mask.sum(axis=1) > 0).sum())
            rows.append({
                "threshold": thr,
                "instance": combo["instance"],
                "lam_breadth": combo["costs.space.lam_breadth"],
                "lam_depth": combo["costs.space.lam_depth"],
                "channels_used": mask.sum() / layers_used,
                "layers_used": layers_used,
            })
    return pd.DataFrame(rows), missing


def summarize(df, lam_col, value_col):
    per_instance = df.groupby(["threshold", lam_col, "instance"])[value_col].mean()
    stats = per_instance.groupby(["threshold", lam_col]).agg(["mean", "std"])
    table = {}
    for (thr, lam), r in stats.iterrows():
        table.setdefault(lam, {})[thr] = f"{r['mean']:.2f} ({r['std']:.2f})"
    return pd.DataFrame(table).T[THRESHOLDS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", action="store_true", help="use the original (pre-fix) pruning files")
    args = parser.parse_args()

    cfg_base = nn_bowl.utils.load_config("config_breadth_depth.yaml")
    df, missing = load_resources(cfg_base, args.v1)
    if missing:
        print(f"WARNING: {len(missing)} models missing pruning files, e.g. {missing[:3]}")

    pd.set_option("display.width", 200)
    print(f"Table 7 ({'v1 original' if args.v1 else 'v2 corrected'} pruning files)\n")
    print("Channels used")
    print(summarize(df, "lam_breadth", "channels_used").to_string())
    print("\nLayers used")
    print(summarize(df, "lam_depth", "layers_used").to_string())


if __name__ == "__main__":
    main()
