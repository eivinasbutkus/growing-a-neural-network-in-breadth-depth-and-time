
import torch
import pickle
import hydra

import nn_bowl

@hydra.main(version_base=None, config_name='config', config_path=nn_bowl.paths.CONFIG_PATH)
def evaluate_attribution(cfg):
    results_dir = nn_bowl.utils.get_results_dir(cfg)
    results_dir.mkdir(parents=True, exist_ok=True)

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
    
    # if cfg.train.compile_model:
    #     model = torch.compile(base_model)
    # else:
    #     model = base_model
    model = base_model # no compilation for attribution

    model.eval()

    attribution_results_path = results_dir / "attribution_results.pkl"
    if not attribution_results_path.exists():
        print("Evaluating attribution...")
        attribution_heatmaps = nn_bowl.utils.get_attribution_heatmaps_raw_for_test_images(cfg, model, test_loader, device)
        with open(attribution_results_path, "wb") as f:
            pickle.dump(attribution_heatmaps, f)


if __name__ == '__main__':
    evaluate_attribution()