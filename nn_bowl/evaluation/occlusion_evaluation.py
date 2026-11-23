
import torch
import tqdm
import numpy as np
import nn_bowl


def evaluate_occlusion(cfg):

    # evaluate the model
    results_dir = nn_bowl.utils.get_results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = nn_bowl.utils.get_device()

    # init model
    base_model = nn_bowl.Model(cfg)
    base_model.to(device)

    # init cost manager
    cost_manager = nn_bowl.CostManager(cfg, base_model)
    cost_manager.to(device)

    checkpoint_path = nn_bowl.utils.get_checkpoint_path(cfg, cfg.train.n_epochs - 1) # last epoch
    base_model = nn_bowl.utils.load_checkpoint(base_model, checkpoint_path, compile_clean=True)
    
    model = nn_bowl.utils.maybe_compile(cfg, base_model)

    model.eval()

    for occlusion_proportion in cfg.analysis.occlusion_proportions:
        results_path = results_dir / f"results_occlusion_{occlusion_proportion}.npz"
        if not results_path.exists():
            data = {} # going to be saved in the npz file

            print(f"Evaluating occlusion proportion {occlusion_proportion}...")
            # Get test loader
            cfg_proportion = cfg.copy()
            cfg_proportion.dataset.occlusion_proportion = occlusion_proportion
            test_loader = nn_bowl.utils.get_dataloaders(cfg_proportion)["test"]

            all_probs = []
            all_labels = []
            all_time_selections = []

            with torch.no_grad():
                for batch in tqdm.tqdm(test_loader, desc="Evaluating"):
                    imgs = batch["img"].to(device)
                    labels = batch["label"].to(device)
                    
                    # Forward pass - get outputs including time_selection
                    out = model(imgs, noise_anneal=1.0)  # Use appropriate noise_anneal value

                    all_probs.append(out["probs"].cpu())
                    all_labels.append(labels.cpu())
                    all_time_selections.append(out["time_selection"].cpu())

                # concat batches
                data["probs"] = torch.cat(all_probs, dim=0).numpy()
                data["labels"] = torch.cat(all_labels, dim=0).numpy()
                data["occlusion_proportion"] = occlusion_proportion
                data["time_selection"] = torch.cat(all_time_selections, dim=0).numpy()

            print(f"Saving results to {results_path}...")
            np.savez(results_path, **data)
