import torch
from datasets import load_dataset # huggingface datasets
from datasets import Image
import torchvision.transforms as T

from nn_bowl.paths import DATASETS_DIR

NORMALIZATION_PARAMS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "mnist": ((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "tiny-imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}

AUGMENTATIONS = {
    "cifar10": [T.RandomHorizontalFlip(p=0.5)],
    "cifar100": [T.RandomHorizontalFlip(p=0.5)],
    "mnist": [T.RandomRotation(10)],  # Rotate by up to 10 degrees
    "imagenet": [T.RandomHorizontalFlip(p=0.5)],
    "tiny-imagenet": [T.RandomHorizontalFlip(p=0.5)]
}

HUGGINGFACE_NAMES = {
    "cifar10": "cifar10",
    "cifar100": "cifar100",
    "mnist": "mnist",
    "imagenet": "imagenet-1k",
    "tiny-imagenet": "slegroux/tiny-imagenet-200-clean"
}

def _get_occlusion_transform(cfg):
    if cfg.dataset.occlusion_proportion < 0.0 or cfg.dataset.occlusion_proportion > 1.0:
        raise ValueError("occlusion_proportion must be between 0.0 and 1.0")
    
    if cfg.dataset.occlusion_proportion == 0.0:
        return T.Lambda(lambda x: x)  # identity transform (no occlusion)
    
    proportion = cfg.dataset.occlusion_proportion

    def _random_gray_square(x):
        # x is a tensor with shape (C, H, W)
        _, h, w = x.shape
        side = int(round((proportion * h * w) ** 0.5))
        side = min(side, h, w)
        top = 0 if h == side else int(torch.randint(0, h - side + 1, (1,)).item())
        left = 0 if w == side else int(torch.randint(0, w - side + 1, (1,)).item())

        x = x.clone()
        x[:, top:top + side, left:left + side] = 0.0
        return x

    return T.Lambda(_random_gray_square)

def _get_transform(cfg, augment=False):
    if cfg.dataset.name not in NORMALIZATION_PARAMS.keys():
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    augmentation_transforms = AUGMENTATIONS.get(cfg.dataset.name, []) if augment else []
    # occlusion_transform = _get_occlusion_transform(cfg) if occlude else T.Lambda(lambda x: x)
    occlusion_transform = _get_occlusion_transform(cfg)
    
    transforms = [
        T.Resize(cfg.model.spatial_size),  # Resize to model's expected input size
    ]
    
    if cfg.dataset.name == "mnist":
        transforms.append(T.Grayscale(num_output_channels=3))
        
    transforms.extend([
        *augmentation_transforms,
        T.ToTensor(),
        T.Normalize(*NORMALIZATION_PARAMS[cfg.dataset.name]),  # Dataset-specific stats
        occlusion_transform,
    ])

    return T.Compose(transforms)

def _make_formatter(transform):
    return lambda examples: {
        'img': [transform(image) for image in examples['img']],
        'label': examples['label']
    }

def get_dataloaders(cfg):
    cache_dir = DATASETS_DIR if cfg.dataset.path is None else cfg.dataset.path
    hf_dataset_name = HUGGINGFACE_NAMES.get(cfg.dataset.name)
    dataset = load_dataset(hf_dataset_name, cache_dir=cache_dir, download_mode="reuse_dataset_if_exists")

    # normalize label column name to 'label' (cifar-100 has coarse_label and fine_label...)
    train_columns = dataset["train"].column_names if "train" in dataset else next(iter(dataset.values())).column_names
    if "fine_label" in train_columns:
        dataset = dataset.rename_column("fine_label", "label")

    # normalize image column name to 'img' (standard for cifar-10, not for everything else...)
    if "image" in train_columns:
        dataset = dataset.rename_column("image", "img")

    dataset = dataset.cast_column("img", Image(decode=True)) # suggested by chatGPT to handle errno24 - too many open files

    if cfg.dataset.name == "tiny-imagenet": # tiny-imagenet has a separate validation set, but we want to combine it with training for our train/val split
        from datasets import concatenate_datasets
        dataset["train"] = concatenate_datasets([dataset["train"], dataset["validation"]])
        del dataset["validation"]

    dataloader_params = {
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.train.num_workers,
        'pin_memory': True,
    }

    dataloaders = {}
    for split in dataset:
        # print(f"Processing {split} split of {cfg.dataset.name} dataset...")
        
        if split == "train":

            # Split training data into train/validation
            train_val_dataset = dataset[split]
            
            # Create indices for splitting
            total_size = len(train_val_dataset)
            val_size = int(cfg.train.val_split * total_size)  # e.g., 0.1 for 10%
            train_size = total_size - val_size
            
            # Use fixed seed for reproducible splits
            generator = torch.Generator().manual_seed(cfg.train.val_seed)
            train_indices, val_indices = torch.utils.data.random_split(
                range(total_size), [train_size, val_size], generator=generator
            )

            # Create train dataloader
            train_subset = train_val_dataset.select(train_indices.indices)
            dataloaders["train"] = torch.utils.data.DataLoader(
                train_subset.with_transform(_make_formatter(_get_transform(cfg, augment=True))),  # Augment only training data
                **dataloader_params,
                shuffle=True,
            )
            
            # Create validation dataloader
            val_subset = train_val_dataset.select(val_indices.indices)
            dataloaders["valid"] = torch.utils.data.DataLoader(
                val_subset.with_transform(_make_formatter(_get_transform(cfg, augment=False))),  # No augmentation for val
                **dataloader_params,
                shuffle=False,
            )
            
        elif split == "test":
            # Keep test set as-is
            dataloaders[split] = torch.utils.data.DataLoader(
                dataset[split].with_transform(_make_formatter(_get_transform(cfg, augment=False))),
                **dataloader_params,
                shuffle=False,  # Don't shuffle test set
            )
        
    return dataloaders


def get_cifar_labels():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def denormalize(img_tensor, dataset_name):
    mean, std = NORMALIZATION_PARAMS[dataset_name]
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(img_tensor * std + mean, 0, 1)