import torch
import nn_bowl
import pandas as pd
import numpy as np


def _get_conv_state(model):
    return {name: m.weight.data.clone() for name, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)}


def _restore_conv_state(model, state):
    for name, module in model.named_modules():
        if name in state and isinstance(module, torch.nn.Conv2d):
            module.weight.data.copy_(state[name])


def _zero_layer(model, layer_index):
    suffixes = [f"bottom_up_{layer_index}", f"lateral_{layer_index}", f"top_down_{layer_index}"]
    for name, module in model.named_modules():
        if any(name.endswith(s) for s in suffixes) and isinstance(module, torch.nn.Conv2d):
            module.weight.data.fill_(0)


def _zero_channel_rank(model, target_layers, layer_channel_indices, channel_rank):
    for layer_index in target_layers:
        channel_idx = layer_channel_indices[layer_index, channel_rank].item()
        suffixes = [f"bottom_up_{layer_index}", f"lateral_{layer_index}", f"top_down_{layer_index}"]
        for name, module in model.named_modules():
            if any(name.endswith(s) for s in suffixes) and isinstance(module, torch.nn.Conv2d):
                module.weight.data[channel_idx] = 0

def _is_threshold_hit(cfg, model, test_loader, threshold_acc, previous_acc, threshold_delta_acc, double_checking=False):
    limit_n_imgs = 1024 if not double_checking else cfg.analysis.n_pruning_imgs
    acc = nn_bowl.utils.get_test_accuracy(model, test_loader, limit_n_imgs=limit_n_imgs)
    delta_acc = previous_acc - acc

    if not double_checking:
        # fast check: only test global threshold, skip delta (too noisy at 1024 imgs)
        threshold_hit = acc < threshold_acc
        if not threshold_hit:
            return False, acc

        # hit: confirm with full eval (return the full-set accuracy that made the decision)
        threshold_hit, acc_confirm = _is_threshold_hit(cfg, model, test_loader, threshold_acc, previous_acc, threshold_delta_acc, double_checking=True)
        return threshold_hit, acc_confirm
    else:
        # full eval: apply both checks (accuracy and delta accuracy check that drop is big enough to prune the channel)
        print("Double-checking decision to keep with full evaluation...")
        threshold_hit = (acc < threshold_acc) or (delta_acc > threshold_delta_acc)
        return threshold_hit, acc



def _full_test_accuracy(cfg, model, test_loader):
    """Accuracy on the full test set with a fixed noise seed.

    The same seed is used for the baseline and every candidate mask, so all candidates are compared
    under the same noise draws (common random numbers) and results are reproducible.
    """
    torch.manual_seed(cfg.analysis.get("pruning_eval_seed", 0))
    return nn_bowl.utils.get_test_accuracy(model, test_loader)


def get_pruning_accuracies(cfg, box=True, epoch=None):
    device = nn_bowl.utils.get_device()

    base_model = nn_bowl.Model(cfg)
    base_model.to(device)

    test_loader = nn_bowl.utils.get_dataloaders(cfg)["test"]


    epoch = epoch if epoch is not None else cfg.train.n_epochs - 1
    try:
        checkpoint_path = nn_bowl.utils.get_checkpoint_path(cfg, cfg.train.n_epochs - 1)
        base_model = nn_bowl.utils.load_checkpoint(base_model, checkpoint_path, compile_clean=True)
    except FileNotFoundError:
        print(f"Checkpoint for epoch {cfg.train.n_epochs - 1} not found. Attempting to load the last available checkpoint.")
        checkpoint_path = nn_bowl.utils.get_last_checkpoint_path(cfg)
        base_model = nn_bowl.utils.load_checkpoint(base_model, checkpoint_path, compile_clean=True)

    model = nn_bowl.utils.maybe_compile(cfg, base_model)
    model.eval()


    n_layers = cfg.model.n_layers
    n_channels = cfg.model.n_channels

    cost_manager = nn_bowl.CostManager(cfg, model)
    cost_manager.to(device)
    _, layer_channel_norms, layer_channel_indices = cost_manager.costs["space"].get_space_costs()  # [n_layers, n_channels], descending

    # using relative threshold (e.g. preserve 90% of above chance performance)
    baseline_acc = nn_bowl.utils.get_test_accuracy(model, test_loader)
    previous_acc = baseline_acc
    threshold = cfg.analysis.pruning_accuracy_threshold
    chance_acc = 1.0 / model.n_classes
    threshold_acc = chance_acc + threshold * (baseline_acc - chance_acc)
    threshold_delta_acc = cfg.analysis.pruning_accuracy_delta * (baseline_acc - threshold_acc)
    print("Baseline accuracy before pruning:", baseline_acc)
    print(f"Pruning accuracy threshold (relative {threshold}):", threshold_acc)
    print(f"Pruning accuracy delta threshold (relative {cfg.analysis.pruning_accuracy_delta} of above-chance budget):", threshold_delta_acc)

    # full model accuracy before pruning
    mask = np.ones((n_layers, n_channels), dtype=bool)
    masks = [mask.copy()]
    results_list = [{"phase": "baseline", "mask_index" : 0, "accuracy": baseline_acc, "threshold_hit": False}]

    # Phase 1: zero layers one by one (last = least important first)
    n_layers_preserved = 0
    for layer_index in range(n_layers - 1, -1, -1):
        print(f"Layer pruning: zeroing layer {layer_index}")
        state_before = _get_conv_state(model)

        _zero_layer(model, layer_index)
        mask[layer_index, :] = False

        threshold_hit, acc = _is_threshold_hit(cfg, model, test_loader, threshold_acc, previous_acc, threshold_delta_acc)
        masks.append(mask.copy())
        results_list.append({"phase": "layer", "mask_index": len(masks)-1, "accuracy": acc, "threshold_hit": threshold_hit})

        if threshold_hit:
            _restore_conv_state(model, state_before)
            n_layers_preserved = layer_index + 1
            mask[layer_index, :] = True
            break
        else:
            previous_acc = acc

    # Phase 2: zero channel ranks one by one (last col = least important first)
    if box:
        for channel_rank in range(n_channels - 1, -1, -1):
            print(f"BOX Channel pruning: zeroing rank {channel_rank} (layers preserved: {n_layers_preserved})")

            target_layers = list(range(n_layers_preserved)) # all layers up to n_layers_preserved
            _zero_channel_rank(model, target_layers, layer_channel_indices, channel_rank)
            mask[:, channel_rank] = False

            threshold_hit, acc = _is_threshold_hit(cfg, model, test_loader, threshold_acc, previous_acc, threshold_delta_acc)

            # for each layer, how many channels are preserved [layer1_n_channels_preserved, layer2_n_channels_preserved, ...]
            masks.append(mask.copy())
            results_list.append({"phase": "channel", "mask_index": len(masks)-1, "accuracy": acc, "threshold_hit": threshold_hit})

            if threshold_hit:
                break
            else:
                previous_acc = acc

    # free-form pruning implementation: zero each rank across all preserved layers at once,
    # then individually if threshold hit
    else:
        active_layers = set(range(n_layers_preserved))

        for channel_rank in range(n_channels - 1, -1, -1):
            if not active_layers:
                break

            target_layers = sorted(active_layers)
            max_norm = max(layer_channel_norms[li, channel_rank].item() for li in target_layers)
            if max_norm < 1e-6:
                _zero_channel_rank(model, target_layers, layer_channel_indices, channel_rank)
                for li in target_layers:
                    mask[li, channel_rank] = False

                print(f"Skipping channel rank {channel_rank} with near-zero max norm {max_norm:.2e} across active layers {target_layers}")
                continue

            print(f"FREEFORM Channel pruning: zeroing rank {channel_rank} across layers {target_layers}")

            # try zeroing this rank across all active layers at once
            state_before = _get_conv_state(model)
            _zero_channel_rank(model, target_layers, layer_channel_indices, channel_rank)
            for li in target_layers:
                mask[li, channel_rank] = False

            threshold_hit, acc = _is_threshold_hit(cfg, model, test_loader, threshold_acc, previous_acc, threshold_delta_acc)

            masks.append(mask.copy())
            results_list.append({"phase": "channel", "mask_index": len(masks)-1, "accuracy": acc, "threshold_hit": threshold_hit})

            if threshold_hit:
                # failed jointly — restore and test each layer individually
                _restore_conv_state(model, state_before)
                for li in target_layers:
                    mask[li, channel_rank] = True

                for li in reversed(target_layers): # from top to bottom
                    print(f"Testing layer {li} for survival at channel rank {channel_rank}")
                    # save state before zeroing this single layer-channel
                    state_before_single = _get_conv_state(model)
                    _zero_channel_rank(model, [li], layer_channel_indices, channel_rank)
                    mask[li, channel_rank] = False

                    threshold_hit, acc_single = _is_threshold_hit(cfg, model, test_loader, threshold_acc, previous_acc, threshold_delta_acc)

                    masks.append(mask.copy())
                    results_list.append({"phase": "channel", "mask_index": len(masks)-1, "accuracy": acc_single, "threshold_hit": threshold_hit})

                    if threshold_hit:
                        # this layer needs the rank — restore and deactivate layer
                        _restore_conv_state(model, state_before_single)
                        mask[li, channel_rank] = True
                        active_layers.discard(li)
                    else:
                        previous_acc = acc_single
            else:
                previous_acc = acc


    df = pd.DataFrame(results_list)
    df["pruning_type"] = "box" if box else "freeform"

    return df, np.stack(masks)


def get_pruning_accuracies_iterative(cfg):
    device = nn_bowl.utils.get_device()

    base_model = nn_bowl.Model(cfg)
    base_model.to(device)

    test_loader = nn_bowl.utils.get_dataloaders(cfg)["test"]

    epoch = cfg.train.n_epochs - 1

    try:
        checkpoint_path = nn_bowl.utils.get_checkpoint_path(cfg, cfg.train.n_epochs - 1)
        base_model = nn_bowl.utils.load_checkpoint(base_model, checkpoint_path, compile_clean=True)
    except FileNotFoundError:
        print(f"Checkpoint for epoch {cfg.train.n_epochs - 1} not found. Attempting to load the last available checkpoint.")
        checkpoint_path = nn_bowl.utils.get_last_checkpoint_path(cfg)
        base_model = nn_bowl.utils.load_checkpoint(base_model, checkpoint_path, compile_clean=True)

    model = nn_bowl.utils.maybe_compile(cfg, base_model)
    model.eval()

    n_layers = cfg.model.n_layers
    n_channels = cfg.model.n_channels

    cost_manager = nn_bowl.CostManager(cfg, model)
    cost_manager.to(device)
    _, layer_channel_norms, layer_channel_indices = cost_manager.costs["space"].get_space_costs()  # [n_layers, n_channels], descending

    # using relative threshold (e.g. preserve 90% of above chance performance)
    baseline_acc = _full_test_accuracy(cfg, model, test_loader)
    threshold = cfg.analysis.pruning_accuracy_threshold
    chance_acc = 1.0 / model.n_classes
    threshold_acc = chance_acc + threshold * (baseline_acc - chance_acc)
    print("Baseline accuracy before pruning:", baseline_acc)
    print(f"Pruning accuracy threshold (relative {threshold}):", threshold_acc)

    # full model accuracy before pruning
    mask = np.ones((n_layers, n_channels), dtype=bool)
    masks = [mask]

    low = layer_channel_norms[layer_channel_norms > 0].min().item() * 0.1
    high = layer_channel_norms.max().item() * 1.1
    results_list = [{"phase": "baseline", "mask_index" : 0, "accuracy": baseline_acc, "threshold_hit": False, "threshold_norm": high}]
    state_baseline = _get_conv_state(model)

    # iterative binary search for the threshold in log space that preserves the target accuracy
    for iteration in range(cfg.analysis.pruning_binary_search_iterations):
        print(f"Binary search iteration {iteration+1}/{cfg.analysis.pruning_binary_search_iterations}")
        mid_norm = np.exp((np.log(low) + np.log(high)) / 2)
        print(f"Testing threshold norm {mid_norm:.4e} (low={low:.4e}, high={high:.4e})")

        # zero channels below this norm threshold
        mask = np.ones((n_layers, n_channels), dtype=bool)
        for layer_index in range(n_layers):
            for channel_index in range(n_channels):
                if layer_channel_norms[layer_index, channel_index].item() < mid_norm:
                    _zero_channel_rank(model, [layer_index], layer_channel_indices, channel_index)
                    mask[layer_index, channel_index] = False

        # every candidate is checked on the full test set (v1 accepted candidates on a 1024-image subset)
        acc = _full_test_accuracy(cfg, model, test_loader)
        threshold_hit = acc < threshold_acc

        masks.append(mask)
        results_list.append({"phase": "iterative", "mask_index": len(masks)-1, "accuracy": acc, "threshold_hit": threshold_hit, "threshold_norm": mid_norm})

        if threshold_hit:
            high = mid_norm # accuracy dropped too much - decrease threshold to preserve more channels
        else:
            low = mid_norm # accuracy preserved - increase threshold to prune more

        _restore_conv_state(model, state_baseline)


    df = pd.DataFrame(results_list)
    df["pruning_type"] = "iterative_binary_search"

    return df, np.stack(masks)

