
import pickle

import numpy as np
import tqdm
import pandas as pd

import nn_bowl

__all__ = [
    "get_combo_key",
    "get_combo_from_row",
    "compute_statistics",
    "load_all_probs",
    "js_divergence_fast",
    "get_model_df",
    "get_image_df",
    "get_layer_channel_norms",
    "get_pruning_results_and_masks",
    "get_time_df",
    "get_occlusion_df",
    "load_heatmaps",
]


def get_combo_key(combo):
    """Stable hashable key from a combo dict, for use in returned lookup dicts."""
    return tuple(sorted(combo.items()))


def get_combo_from_row(cfg_base, row):
    """Reconstruct a combo dict from a model_df row."""
    combo = {var.key: row[var.key] for var in cfg_base.experiment.variables}
    combo["instance"] = int(row["instance"])
    return combo


def compute_statistics(results_data, labels, dim_red_models={}):

    statistics = {}
    # accuracy
    probs = results_data["probs"]
    statistics["accuracy"] = (labels == np.argmax(probs, axis=1)).mean()

    # dimensionality reduction
    for dr_model_name, dr_model in dim_red_models.items():
        print(probs.reshape(1, -1).shape)
        dr_embeddings = dr_model.transform(probs.reshape(1, -1))
        statistics[f"{dr_model_name}_embeddings"] = dr_embeddings

    return statistics


def load_all_probs(cfg_base, model_df):
    print("loading all probs based on existing rows in model df..")
    all_probs = []

    for _, row in tqdm.tqdm(model_df.iterrows(), total=model_df.shape[0]):
        combo = get_combo_from_row(cfg_base, row)
        cfg = nn_bowl.utils.get_cfg(cfg_base, combo)
        results_dir = nn_bowl.utils.get_results_dir(cfg)
        try:
            results_data = np.load(results_dir / "results.npz")
        except FileNotFoundError:
            print(f"Results not found for {combo}")
            continue

        all_probs.append(results_data["probs"])

    return np.stack(all_probs, axis=0)


def js_divergence_fast(p, q):
    """
    Fast vectorized JS divergence computation.
    p, q: (n_examples, n_classes)
    Returns: scalar (average JS divergence across all examples)
    """
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=-1)
    kl_qm = np.sum(q * np.log(q / m), axis=-1)
    js_per_example = 0.5 * (kl_pm + kl_qm)
    return np.sqrt(np.mean(js_per_example))


def get_model_df(cfg_base):
    variables = list(cfg_base.experiment.variables)
    var_keys = [var.key for var in variables]

    data = {
        "model_name": [],
        **{key: [] for key in var_keys},
        "instance": [],
        "cost_depth": [],
        "cost_breadth": [],
        "cost_time": [],
    }

    labels_cache = {}

    def get_labels(cfg):
        dataset_key = cfg.dataset.name if hasattr(cfg.dataset, "name") else str(cfg.dataset)
        if dataset_key not in labels_cache:
            lbls = []
            test_loader = nn_bowl.utils.get_dataloaders(cfg)["test"]
            for batch in test_loader:
                lbls.append(batch["label"].numpy())
            labels_cache[dataset_key] = np.concatenate(lbls, axis=0)
            print(f"Labels shape for {dataset_key}:", labels_cache[dataset_key].shape)
        return labels_cache[dataset_key]

    for combo in tqdm.tqdm(nn_bowl.utils.get_combinations(cfg_base)):
        cfg = nn_bowl.utils.get_cfg(cfg_base, combo)
        results_dir = nn_bowl.utils.get_results_dir(cfg)
        labels = get_labels(cfg)

        try:
            results_data = np.load(results_dir / "results.npz")
            statistics = compute_statistics(results_data, labels, dim_red_models={})
            for key in statistics:
                if key not in data:
                    data[key] = []
                data[key].append(statistics[key])

            for key in var_keys:
                data[key].append(combo[key])
            data["instance"].append(combo["instance"])
            data["cost_depth"].append(results_data["depth_cost"].item())
            data["cost_breadth"].append(results_data["breadth_cost"].item())
            data["cost_time"].append(results_data["time_cost"].mean())
            name = "_".join(f"{k.split('.')[-1]}={v}" for k, v in combo.items() if k != "instance")
            data["model_name"].append(f"{name}_i{combo['instance']}")
        except Exception as e:
            print(f"Error computing statistics for combination: {combo}")
            print(e)
            raise e
            # continue

    return pd.DataFrame(data)


def get_image_df(cfg_base, model_df):
    labels = []
    test_loader = nn_bowl.utils.get_dataloaders(cfg_base)["test"]
    for batch in test_loader:
        labels.append(batch["label"].numpy())
    labels = np.concatenate(labels, axis=0)

    dfs = []
    for _, row in tqdm.tqdm(model_df.iterrows(), total=model_df.shape[0]):
        combo = get_combo_from_row(cfg_base, row)
        cfg = nn_bowl.utils.get_cfg(cfg_base, combo)
        results_dir = nn_bowl.utils.get_results_dir(cfg)
        try:
            results_data = np.load(results_dir / "results.npz")
        except FileNotFoundError:
            print(f"Results not found for {combo}")
            continue

        probs = results_data["probs"]  # (n_examples, n_classes)
        preds = np.argmax(probs, axis=1)
        correct = (preds == labels).astype(int)
        confidence = probs[np.arange(len(labels)), labels]
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

        df = pd.DataFrame({
            "model_name": row["model_name"],
            **{k: v for k, v in combo.items()},
            "dataset_idx": np.arange(len(labels)),
            "label": labels,
            "pred": preds,
            "correct": correct,
            "confidence": confidence,
            "entropy": entropy,
        })

        for c in range(probs.shape[1]):
            df[f"prob_{c}"] = probs[:, c]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def get_layer_channel_norms(cfg_base):
    """Returns dict keyed by _combo_key(combo) → layer_channel_norms array."""
    all_layer_channel_norms = {}
    for combo in tqdm.tqdm(nn_bowl.utils.get_combinations(cfg_base)):
        cfg = nn_bowl.utils.get_cfg(cfg_base, combo)
        results_dir = nn_bowl.utils.get_results_dir(cfg)
        try:
            results_data = np.load(results_dir / "results.npz")
            all_layer_channel_norms[get_combo_key(combo)] = results_data["layer_channel_norms"]
        except Exception as e:
            print(f"Error loading results for combination: {combo}")
            print(e)
            continue

    return all_layer_channel_norms


def get_pruning_results_and_masks(cfg_base, model_df):
    """Return dict keyed by _combo_key(combo) → {n_active_layers, n_active_channels}."""

    masks = {}
    dfs = {}
    missing = []
    for _, row in model_df.iterrows():
        combo = get_combo_from_row(cfg_base, row)
        cfg = nn_bowl.utils.get_cfg(cfg_base, combo)
        pruning_results_path, pruning_masks_path = nn_bowl.utils.get_pruning_paths(cfg)
        if not pruning_masks_path.exists() or not pruning_results_path.exists():
            missing.append(combo)
            continue
        
        pruning_results_df = pd.read_csv(pruning_results_path)
        dfs[get_combo_key(combo)] = pruning_results_df

        # the last mask that did not hit the threshold (smallest mask for layer channel norms that preserved accuracy above threshold)
        mask_index = pruning_results_df[~pruning_results_df["threshold_hit"]].iloc[-1]["mask_index"]
        masks[get_combo_key(combo)] = np.load(pruning_masks_path)[mask_index] # loading that mask

    if missing:
        print(f"WARNING: pruning files ({nn_bowl.utils.PRUNING_VERSION}) missing for {len(missing)} models, e.g. {missing[0]}")

    return dfs, masks


def get_occlusion_df(cfg_base, model_df):
    dfs = []
    for _, row in tqdm.tqdm(model_df.iterrows(), total=model_df.shape[0]):
        combo = get_combo_from_row(cfg_base, row)
        cfg = nn_bowl.utils.get_cfg(cfg_base, combo)
        results_dir = nn_bowl.utils.get_results_dir(cfg)

        occlusion_files = sorted(results_dir.glob("results_occlusion_*.npz"))
        if not occlusion_files:
            print(f"No occlusion results found for {combo}")
            continue

        for results_path in occlusion_files:
            try:
                results_data = np.load(results_path)
            except Exception as e:
                print(f"Error loading {results_path}: {e}")
                continue

            probs = results_data["probs"]
            labels = results_data["labels"]
            occlusion_proportion = results_data["occlusion_proportion"].item()
            time_selection = results_data["time_selection"]  # (n_examples, n_timesteps)
            preds = np.argmax(probs, axis=1)
            correct = (preds == labels).astype(int)
            confidence = probs[np.arange(len(labels)), labels]
            entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
            n_timesteps = time_selection.shape[1]
            time_used = time_selection @ np.arange(1, n_timesteps + 1)

            df = pd.DataFrame({
                "model_name": row["model_name"],
                **{k: v for k, v in combo.items()},
                "occlusion_proportion": occlusion_proportion,
                "dataset_idx": np.arange(len(labels)),
                "label": labels,
                "pred": preds,
                "correct": correct,
                "confidence": confidence,
                "entropy": entropy,
                "time_used": time_used,
            })
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_heatmaps(cfg_base, model_df):
    """Load attribution heatmaps for all models in model_df.

    Returns a list of heatmap dicts (one per row) and adds a 'heatmaps_idx' column
    to model_df in-place, so model_df.loc[i, 'heatmaps_idx'] indexes into the returned list.

    Each heatmap dict has the structure loaded from attribution_results.pkl:
        {img_index: {'grad_cam': {layer: heatmap}, 'ablation_cam': {layer: heatmap}, 'image_perturbation': heatmap}}
    """
    all_heatmaps = []
    heatmap_indices = []
    for _, row in tqdm.tqdm(model_df.iterrows(), total=model_df.shape[0]):
        combo = get_combo_from_row(cfg_base, row)
        cfg = nn_bowl.utils.get_cfg(cfg_base, combo)
        results_dir = nn_bowl.utils.get_results_dir(cfg)
        attribution_path = results_dir / "attribution_results.pkl"
        if not attribution_path.exists():
            raise FileNotFoundError(f"Attribution results not found for {combo}: {attribution_path}")
        with open(attribution_path, "rb") as f:
            all_heatmaps.append(pickle.load(f))
        heatmap_indices.append(len(all_heatmaps) - 1)
    model_df["heatmaps_idx"] = heatmap_indices

    def _normalize_heatmap(heatmap):
        hm = heatmap - heatmap.min()
        hm = hm / (hm.sum() + 1e-8)
        return hm

    # normalize every heatmap to sum to 1
    for heatmaps in all_heatmaps:
        for img_idx, methods in heatmaps.items():
            for method_name, method_heatmaps in methods.items():
                if method_name in ["grad_cam", "ablation_cam"]:
                    for layer, heatmap in method_heatmaps.items():
                        method_heatmaps[layer] = _normalize_heatmap(heatmap)
                elif method_name == "image_perturbation":
                    methods[method_name] = _normalize_heatmap(method_heatmaps)

    return all_heatmaps


def get_time_df(cfg_base, model_df):
    dfs = []
    for _, row in tqdm.tqdm(model_df.iterrows(), total=model_df.shape[0]):
        combo = get_combo_from_row(cfg_base, row)
        cfg = nn_bowl.utils.get_cfg(cfg_base, combo)
        results_dir = nn_bowl.utils.get_results_dir(cfg)
        try:
            results_data = np.load(results_dir / "results.npz", allow_pickle=True)
        except FileNotFoundError:
            print(f"Results not found for {combo}")
            continue

        time_selection = np.array(results_data["time_selection"])
        if time_selection.shape == ():
            continue

        n_examples, n_timesteps = time_selection.shape
        dataset_idx, t = np.meshgrid(np.arange(n_examples), np.arange(n_timesteps), indexing="ij")

        df = pd.DataFrame({
            "model_name": row["model_name"],
            **{k: v for k, v in combo.items()},
            "dataset_idx": dataset_idx.ravel(),
            "t": t.ravel(),
            "t_weight": time_selection.ravel(),
        })
        dfs.append(df)

    time_df = pd.concat(dfs, ignore_index=True)
    lam_time_key = next(var.key for var in cfg_base.experiment.variables if "time.lam" in var.key)
    time_df["str_lam_time"] = time_df[lam_time_key].apply(lambda x: f"lam_time={x}")

    return time_df


