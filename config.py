"""
Training Configuration & Utilities

Shared configuration classes and factory functions for model and data loading.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import models, transforms, datasets


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    # Model & Data
    model_name: str = "resnet50"
    dataset: str = "cifar10"
    num_classes: int = 10

    # Training hyperparameters
    num_epochs: int = 10
    batch_size: int = 128        # per-rank batch size
    learning_rate: float = 0.01  # NOTE: 0.1 is too high for CIFAR + DDP(4 GPUs)
    momentum: float = 0.9
    weight_decay: float = 5e-4
    # Universal default: 1.0 is a no-op when gradient norms are already below
    # 1.0 (typical for shallow models after warmup), and prevents explosion for
    # deep models (ResNeXt101, ConvNeXt). Set to None to disable explicitly.
    grad_clip: Optional[float] = 1.0
    # Warmup epochs: LR ramps linearly from lr×1e-3 to lr over this many epochs,
    # then cosine-decays. 5 epochs prevents the abrupt 1000× LR jump that causes
    # NaN in deep models (ResNeXt101) even when grad_clip is active.
    warmup_epochs: int = 5

    # Distributed training
    backend: str = "mpi"         # "nccl" or "mpi"
    comm_algorithm: Optional[str] = None  # None, "default", "ring", "recursive_doubling"

    # CIFAR stem adaptation (replace 7x7 conv + maxpool with 3x3 conv)
    cifar_stem: bool = True
    # Load ImageNet pretrained weights. Required for large models (ResNeXt101,
    # ConvNeXt) on CIFAR-10 — training 88M+ params from scratch on 50K samples
    # produces forward-pass float32 overflow that optimizer tuning cannot prevent.
    pretrained: bool = False

    # Paths & misc
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42

    def validate(self):
        assert self.backend in ("nccl", "mpi"), f"Unknown backend: {self.backend}"
        assert self.num_epochs > 0
        assert self.batch_size > 0
        assert self.learning_rate > 0


def create_model(
    model_name: str, num_classes: int,
    cifar_stem: bool = True, pretrained: bool = False,
) -> nn.Module:
    builders = {
        "resnet18":         lambda: models.resnet18(num_classes=num_classes),
        "resnet50":         lambda: models.resnet50(num_classes=num_classes),
        "resnet101":        lambda: models.resnet101(num_classes=num_classes),
        "wide_resnet50_2":  lambda: models.wide_resnet50_2(num_classes=num_classes),
        "resnext101_32x8d": lambda: models.resnext101_32x8d(num_classes=num_classes),
        "convnext_tiny":    lambda: models.convnext_tiny(num_classes=num_classes),
        "convnext_small":   lambda: models.convnext_small(num_classes=num_classes),
    }
    pretrained_weights = {
        "resnet18":         models.ResNet18_Weights.DEFAULT,
        "resnet50":         models.ResNet50_Weights.DEFAULT,
        "resnet101":        models.ResNet101_Weights.DEFAULT,
        "wide_resnet50_2":  models.Wide_ResNet50_2_Weights.DEFAULT,
        "resnext101_32x8d": models.ResNeXt101_32X8D_Weights.DEFAULT,
        "convnext_tiny":    models.ConvNeXt_Tiny_Weights.DEFAULT,
        "convnext_small":   models.ConvNeXt_Small_Weights.DEFAULT,
    }

    if model_name not in builders:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(builders.keys())}"
        )

    if pretrained:
        # Load full ImageNet weights (1000-class head), then replace the head.
        # All layers except the classifier are preserved.
        w = pretrained_weights[model_name]
        pretrained_builders = {
            "resnet18":         lambda: models.resnet18(weights=w),
            "resnet50":         lambda: models.resnet50(weights=w),
            "resnet101":        lambda: models.resnet101(weights=w),
            "wide_resnet50_2":  lambda: models.wide_resnet50_2(weights=w),
            "resnext101_32x8d": lambda: models.resnext101_32x8d(weights=w),
            "convnext_tiny":    lambda: models.convnext_tiny(weights=w),
            "convnext_small":   lambda: models.convnext_small(weights=w),
        }
        model = pretrained_builders[model_name]()
        if hasattr(model, 'fc'):                          # ResNet / ResNeXt
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:                                             # ConvNeXt
            model.classifier[-1] = nn.Linear(
                model.classifier[-1].in_features, num_classes
            )
    else:
        model = builders[model_name]()

    # Adapt ResNet/ResNeXt stem for 32×32 CIFAR images (same for both paths)
    if cifar_stem and model_name.startswith(("resnet", "resnext", "wide_resnet")):
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    return model


def get_data_loaders(
    config: TrainingConfig,
    rank: int,
    world_size: int,
) -> tuple:
    """Load dataset and create distributed dataloaders."""
    if config.dataset == "cifar10":
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        ds_cls = datasets.CIFAR10
    elif config.dataset == "cifar100":
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )
        ds_cls = datasets.CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ds_cls(
        root=config.data_dir, train=True, download=False,
        transform=train_transform,
    )
    val_dataset = ds_cls(
        root=config.data_dir, train=False, download=False,
        transform=val_transform,
    )

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, seed=config.seed,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank,
        shuffle=False, seed=config.seed,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        sampler=val_sampler, num_workers=4, pin_memory=True,
    )

    return train_loader, val_loader, train_sampler