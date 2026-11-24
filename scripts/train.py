
import torch
import hydra
import tqdm
import nn_bowl

from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler # mixed precision training


@hydra.main(version_base=None, config_name='config', config_path=nn_bowl.paths.CONFIG_PATH)
def train(cfg):
    nn_bowl.utils.pretty_print_cfg(cfg)
    device = nn_bowl.utils.get_device()
    print("Using device: ", device)

    # before compiling, used when saving the checkpoint
    base_model = nn_bowl.Model(cfg)
    base_model.to(device)

    
    # mixed precision
    scaler = GradScaler()
    if cfg.train.mixed_precision and 'cuda' not in str(device):
        print("Warning: mixed precision training is enabled but CUDA is not available. Disabling mixed precision.")
        cfg.train.mixed_precision = False

    dataloaders = nn_bowl.utils.get_dataloaders(cfg)
    optimizer = torch.optim.AdamW(base_model.parameters(), **cfg.train.adamw)
    lr_scheduler = nn_bowl.utils.get_lr_scheduler(cfg, optimizer)

    # Loads checkpoint if available
    start_epoch, checkpoint_dir = nn_bowl.utils.load_checkpoint_full(cfg, base_model, optimizer, lr_scheduler, scaler)

    # Override weight_decay from config (allows changing it mid-run without retraining)
    for g in optimizer.param_groups:
        g['weight_decay'] = cfg.train.adamw.weight_decay

    # optionally compile the model for faster training (CUDA only)
    model = nn_bowl.utils.maybe_compile(cfg, base_model, mode='reduce-overhead')  # or 'default', 'max-autotune'

    writer = SummaryWriter(log_dir=checkpoint_dir / "tensorboard")

    for epoch in range(start_epoch, cfg.train.n_epochs):
        print("lr:", optimizer.param_groups[0]["lr"])

        for split in ["train", "valid"]:
            dataloader = dataloaders[split]

            # initialize cost manager for each epoch and split (restarting all metrics for the costs)
            cost_manager = nn_bowl.CostManager(cfg, model).to(device)

            train = split == "train"
            noise_anneal = torch.tensor(nn_bowl.utils.linear_anneal(epoch, *dict(cfg.train.noise_anneal).values(), train=train), device=device)
            print("noise_anneal:", noise_anneal)

            # setting gumbel temperature for time weights
            # gumbel_anneal = nn_bowl.utils.linear_anneal(epoch, *dict(cfg.costs.time.anneal).values(), train=train)
            # start, end = cfg.costs.time.gumbel_temperature.start, cfg.costs.time.gumbel_temperature.end
            # temp = gumbel_anneal * end + (1 - gumbel_anneal) * start
            # print("gumbel_anneal:", gumbel_anneal, "gumbel_temperature:", temp)
            # model.time_selection.set_gumbel_temperature(temp)

            print("noise_anneal:", nn_bowl.utils.linear_anneal(epoch, *dict(cfg.train.noise_anneal).values(), train=train))
            model.train() if train else model.eval()

            progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch} ({split})", unit="batch")

            for batch in progress_bar:
                x = batch["img"].to(device)
                y = batch["label"].to(device)

                with torch.no_grad() if not train else torch.enable_grad():
                    with autocast('cuda', enabled=cfg.train.mixed_precision):
                        out = model(x, noise_anneal=noise_anneal)
                        loss = cost_manager(out, y, epoch)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                progress_bar.set_postfix(cost_manager.get_postfix())
            cost_manager.write_to_tensorboard(writer, split, epoch)

        print(f"Epoch {epoch}, {split} loss: {loss.detach().item()}\n")
            
        lr_scheduler.step()

        # saving checkpoint every epoch_interval epochs and the last epoch
        if epoch % cfg.train.checkpoint_epoch_interval == 0 or epoch == cfg.train.n_epochs - 1:
            nn_bowl.utils.save_checkpoint(cfg, base_model, optimizer, lr_scheduler, epoch, scaler)

if __name__ == '__main__':
    train()