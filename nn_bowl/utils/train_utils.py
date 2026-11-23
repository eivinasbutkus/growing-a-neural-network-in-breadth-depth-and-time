import torch
import yaml
from omegaconf import DictConfig, OmegaConf
import datetime
from pathlib import Path
from nn_bowl.paths import CHECKPOINTS_DIR
from .config_utils import checkpoint_elements_to_str

def get_checkpoint_dir(cfg):
    checkpoint_dir = CHECKPOINTS_DIR / f"experiment_{cfg.experiment.name}" / f"{cfg.dataset.name}{checkpoint_elements_to_str(cfg)}" / f"instance{cfg.train.instance}" # can also hardcode 'model' value
    return checkpoint_dir


def get_checkpoint_path(cfg, epoch):
    checkpoint_path = get_checkpoint_dir(cfg)  /  f"checkpoint_{epoch:03d}.pt" 
    return checkpoint_path

def get_last_checkpoint_path(cfg):
    checkpoint_dir = get_checkpoint_dir(cfg)
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Sort or use max based on the epoch number suffix
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
    return latest_checkpoint

def save_checkpoint(cfg, model, optimizer, scheduler, epoch, scaler):
    checkpoint_path = get_checkpoint_path(cfg, epoch)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    print(checkpoint_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        # 'scaler_state_dict': scaler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else {},
    }, checkpoint_path)

    # save config (using yaml)
    config_path = checkpoint_path.parent / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, default_flow_style=False, indent=2)


def load_checkpoint(model, checkpoint_path, compile_clean=False):
    model_state_dict = torch.load(checkpoint_path, weights_only=False)["model_state_dict"]
    
    # optionally clean the state dict from torch.compile prefixes
    #(TODO remove later (after retraining models) because we're no longer saving compiled versions of the model)
    if compile_clean: 
        model_state_dict = {key.replace("_orig_mod.", ""): value for key, value in model_state_dict.items()}

    model.load_state_dict(model_state_dict)

    return model


def load_checkpoint_from_dir(checkpoint_dir, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler:
            # scaler.load_state_dict(checkpoint['scaler_state_dict'])
            scaler_state = checkpoint.get('scaler_state_dict')
            if scaler_state:
                scaler.load_state_dict(scaler_state)

        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch} (File: {latest_checkpoint.name})")
        
        # Also load the config if it exists
        config_path = checkpoint_dir / "config.yaml"
        cfg = None
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            
        return start_epoch, cfg
    else:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

# loads the latest checkpoint from the checkpoints directory
def load_checkpoint_full(cfg, model, optimizer, scheduler, scaler):
    checkpoint_dir = get_checkpoint_dir(cfg)
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stem.split('_')[-1])
        checkpoint = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        scaler_state = checkpoint.get('scaler_state_dict')
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        else:
            print("No scaler state found in checkpoint, skipping scaler loading.")

        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found, starting with epoch 0")
    
    return start_epoch, checkpoint_dir


def get_device():
    # cuda -> mps -> cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device



def get_lr_scheduler(cfg, optimizer):
    warmup_epochs = cfg.train.lr_scheduler.warmup_epochs
    hold_epochs = cfg.train.lr_scheduler.get('hold_epochs', 0)

    schedulers = []
    milestones = []

    # Phase 1: linear warmup (or no-op if warmup_epochs=0)
    if warmup_epochs > 0:
        schedulers.append(torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        ))
    else:
        schedulers.append(torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0
        ))
    milestones.append(warmup_epochs)

    # Phase 2: hold at peak LR (bridges warmup to when costs kick in)
    if hold_epochs > 0:
        schedulers.append(torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=hold_epochs
        ))
        milestones.append(warmup_epochs + hold_epochs)

    # Phase 3: main decay (T_max derived from n_epochs so it always fills remaining epochs)
    if cfg.train.lr_scheduler.name == "cosine":
        cosine_args = dict(cfg.train.lr_scheduler.args)
        cosine_args['T_max'] = cfg.train.n_epochs - warmup_epochs - hold_epochs
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **cosine_args))
    elif cfg.train.lr_scheduler.name == "step":
        schedulers.append(torch.optim.lr_scheduler.StepLR(optimizer, **cfg.train.lr_scheduler.args))
    else:
        raise ValueError(f"Unsupported lr_scheduler: {cfg.train.lr_scheduler.name}")

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=milestones
    )


def linear_anneal(epoch, enabled, warmup_epochs, anneal_epochs, train=True):
    if train and enabled:
        if epoch < warmup_epochs:  # warmup period
            return 0.0
        elif epoch > warmup_epochs + anneal_epochs:  # post annealing period
            return 1.0
        else:
            return (epoch - warmup_epochs) / anneal_epochs

    else:  # validation or test phase
        return 1.0
    
