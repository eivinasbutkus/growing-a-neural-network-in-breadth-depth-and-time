
import traceback
import torch

from nn_bowl.paths import RESULTS_DIR
from .config_utils import checkpoint_elements_to_str
import nn_bowl
import tqdm
from itertools import product


def get_results_dir(cfg):
    results_dir = RESULTS_DIR / f"experiment_{cfg.experiment.name}" / f"model{checkpoint_elements_to_str(cfg)}" / f"instance{cfg.train.instance}"
    return results_dir

def maybe_compile(cfg, model, **compile_kwargs):
    """torch.compile the model if enabled, running on CUDA, and the GPU supports Triton (capability >= 7.0).

    Otherwise the model runs eagerly: torch.compile is not reliable on Apple GPUs (MPS) or CPU, and older
    GPUs (e.g. GTX 1080 Ti) can't run Triton kernels.
    """
    if not cfg.train.get("compile_model", True):
        return model
    if not torch.cuda.is_available():
        print("torch.compile is only used on CUDA GPUs; running uncompiled")
        return model
    if torch.cuda.get_device_capability() < (7, 0):
        print(f"{torch.cuda.get_device_name()} does not support torch.compile (Triton); running uncompiled")
        return model
    print("Compiling model with torch.compile...")
    return torch.compile(model, **compile_kwargs)


# v2: every binary-search step is checked on the full test set (v1 accepted steps on a 1024-image
# subset and recorded subset accuracies). New filenames so v1 files are never reused or overwritten.
PRUNING_VERSION = "v2"

def get_pruning_paths(cfg):
    """(results_csv_path, masks_npy_path) for the current pruning version and threshold."""
    results_dir = get_results_dir(cfg)
    thresh = cfg.analysis.pruning_accuracy_threshold
    suffix = f"thresh{thresh}_{PRUNING_VERSION}"
    return results_dir / f"pruning_results_{suffix}.csv", results_dir / f"pruning_masks_{suffix}.npy"

def get_combinations(cfg_base):
    """Return list of combo dicts, each with variable keys + 'instance'."""
    instances = list(range(cfg_base.experiment.n_instances))
    variables = list(cfg_base.experiment.variables)
    keys = ["instance"] + [var.key for var in variables]
    product_list = [instances] + [list(var.vals) for var in variables]
    return [dict(zip(keys, combo)) for combo in product(*product_list)]



def ensure_evaluation_files_exist(cfg_base, instances=None, skip_pruning=False):
    print("ensuring that all evaluation files exist...")
    if skip_pruning:
        print("Skipping pruning as requested.")

    if instances is None:
        instances = list(range(cfg_base.experiment.n_instances))

    combinations = get_combinations(cfg_base)

    for combo in tqdm.tqdm(combinations):
        cfg = nn_bowl.utils.get_cfg(cfg_base, combo)

        results_dir = nn_bowl.utils.get_results_dir(cfg)
        results_dir.mkdir(parents=True, exist_ok=True)
        try:
            nn_bowl.evaluation.evaluate(cfg, skip_pruning=skip_pruning)
        except Exception as e:
            print(f"Error evaluating combination: {combo}")
            print(e)
            traceback.print_exc()


def get_test_accuracy(model, test_loader, limit_n_imgs=None):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["img"].to(device)
            labels = batch["label"].to(device)
            if limit_n_imgs is not None:
                remaining = limit_n_imgs - total
                imgs = imgs[:remaining]
                labels = labels[:remaining]
            out = model(imgs, noise_anneal=1.0)
            probs = out["probs"]
            predicted = torch.argmax(probs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            if limit_n_imgs is not None and total >= limit_n_imgs:
                break
    accuracy = correct / total
    return accuracy