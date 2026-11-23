
import torch
import tqdm
import numpy as np
import nn_bowl
from nn_bowl.evaluation import pruning


def evaluate(cfg, skip_pruning=False):
    # evaluate the model
    results_dir = nn_bowl.utils.get_results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.npz"
    if not results_path.exists():
        device = nn_bowl.utils.get_device()

        # init model
        base_model = nn_bowl.Model(cfg)
        base_model.to(device)

        # Get test loader
        test_loader = nn_bowl.utils.get_dataloaders(cfg)["test"]

        # init cost manager
        cost_manager = nn_bowl.CostManager(cfg, base_model)
        cost_manager.to(device)

        checkpoint_path = nn_bowl.utils.get_checkpoint_path(cfg, cfg.train.n_epochs - 1) # last epoch
        base_model = nn_bowl.utils.load_checkpoint(base_model, checkpoint_path, compile_clean=True)
        
        model = nn_bowl.utils.maybe_compile(cfg, base_model)

        # Set gumbel_temperature to end value for inference (since it's annealed during training)
        if hasattr(model.time_selection, 'gumbel_temperature'):
            model.time_selection.set_gumbel_temperature(cfg.costs.time.gumbel_temperature.end)

        model.eval()

        all_probs = []
        all_labels = []
        all_time_selections = []
        all_time_costs = []


        # going to be saved in the npz file
        data = {}
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader, desc="Evaluating"):
                imgs = batch["img"].to(device)
                labels = batch["label"].to(device)
                
                # Forward pass - get outputs including time_selection
                out = model(imgs, noise_anneal=1.0)  # Use appropriate noise_anneal value
                # print(out)

                all_probs.append(out["probs"].cpu())
                all_labels.append(labels.cpu())
                all_time_selections.append(out["time_selection"].cpu())

                # time_cost = cost_manager.costs["time"](out, labels)
                time_cost = cost_manager.costs["time"].get_time_cost(out, labels)
                all_time_costs.append(time_cost.cpu())

            # space costs and layer channel norms
            space_costs, layer_channel_norms, _ = cost_manager.costs["space"].get_space_costs()

            # concat batches
            data["probs"] = torch.cat(all_probs, dim=0).numpy()
            data["labels"] = torch.cat(all_labels, dim=0).numpy()
            data["time_selection"] = torch.cat(all_time_selections, dim=0).numpy()
            data["time_cost"] = torch.cat(all_time_costs, dim=0).numpy()
            data["depth_cost"] = space_costs["depth"].detach().cpu().numpy()
            data["breadth_cost"] = space_costs["breadth"].detach().cpu().numpy()
            data["layer_channel_norms"] = layer_channel_norms.detach().cpu().numpy()

            print(f"Saving results to {results_path}...")
            np.savez(results_path, **data)

    if skip_pruning:
        # print("Skipping pruning evaluation as requested.")
        return

    # Save pruning results to a separate CSV and pruning masks to npy file
    pruning_results_path, pruning_masks_path = nn_bowl.utils.get_pruning_paths(cfg)
    if not pruning_results_path.exists() or not pruning_masks_path.exists():
        print("Evaluating pruning...")
        # pruning_df, masks = pruning.get_pruning_accuracies(cfg, box=False)
        pruning_df, masks = pruning.get_pruning_accuracies_iterative(cfg)
        pruning_df.to_csv(pruning_results_path, index=False)
        np.save(pruning_masks_path, masks)

    # attribution