"""
Training Configuration & Utilities

Shared configuration classes and factory functions for model and data loading.

Data layout expected by default:
    ./data/
      cifar/       # torchvision CIFAR10/CIFAR100 root
      imagenet/    # ImageFolder-style dataset
        train/
        val/
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, models, transforms


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Model & data
    model_name: str = "resnet50"
    dataset: str = "cifar10"
    num_classes: int = 10
    image_size: int = 32

    # Training hyperparameters
    num_epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 5e-4
    grad_clip: Optional[float] = None

    # Scheduler
    scheduler: str = "constant"
    warmup_epochs: int = 0

    # DataLoader behavior
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False

    # Distributed training
    backend: str = "mpi"
    comm_algorithm: Optional[str] = None

    # ZFP compression
    zfp_rate: float = 16.0

    # Model initialization
    cifar_stem: bool = True
    pretrained: bool = False
    init_cifar_stem_from_pretrained_center: bool = False

    # Paths & misc
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    seed: int = 42

    def validate(self):
        valid_datasets = ("cifar10", "cifar100", "imagenet", "imagenet_like")
        valid_schedulers = ("constant", "cosine")

        assert self.dataset in valid_datasets, f"Unknown dataset: {self.dataset}"
        assert self.backend in ("nccl", "mpi"), f"Unknown backend: {self.backend}"
        assert self.scheduler in valid_schedulers, (
            f"Unknown scheduler: {self.scheduler}"
        )
        assert self.num_epochs > 0
        assert self.batch_size > 0
        assert self.learning_rate > 0
        assert self.num_workers >= 0

        if self.dataset.startswith("cifar"):
            assert self.image_size == 32, "CIFAR experiments should use image_size=32"
        else:
            assert self.image_size > 0


def apply_cifar_stem(
    model: nn.Module,
    pretrained: bool = False,
    init_from_pretrained_center: bool = False,
) -> nn.Module:
    old_conv1 = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=old_conv1.in_channels,
        out_channels=old_conv1.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    if pretrained and init_from_pretrained_center:
        if old_conv1.weight.shape[-2:] != (7, 7):
            raise ValueError(
                "Cannot center-crop pretrained stem: expected old conv1 to be 7x7."
            )

        with torch.no_grad():
            model.conv1.weight.copy_(old_conv1.weight[:, :, 2:5, 2:5])

    model.maxpool = nn.Identity()
    return model


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
    else:
        raise ValueError(
            f"Do not know how to replace classifier for {type(model).__name__}"
        )
    return model


def create_model(
    model_name: str,
    num_classes: int,
    cifar_stem: bool = True,
    pretrained: bool = False,
    init_cifar_stem_from_pretrained_center: bool = False,
) -> nn.Module:
    builders = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "wide_resnet50_2": models.wide_resnet50_2,
        "resnext101_32x8d": models.resnext101_32x8d,
        "convnext_tiny": models.convnext_tiny,
        "convnext_small": models.convnext_small,
    }

    pretrained_weights = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet101": models.ResNet101_Weights.DEFAULT,
        "wide_resnet50_2": models.Wide_ResNet50_2_Weights.DEFAULT,
        "resnext101_32x8d": models.ResNeXt101_32X8D_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "convnext_small": models.ConvNeXt_Small_Weights.DEFAULT,
    }

    if model_name not in builders:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(builders.keys())}"
        )

    if pretrained:
        model = builders[model_name](weights=pretrained_weights[model_name])
        model = _replace_classifier(model, num_classes)
    else:
        model = builders[model_name](weights=None, num_classes=num_classes)

    if cifar_stem and model_name.startswith(("resnet", "resnext", "wide_resnet")):
        model = apply_cifar_stem(
            model,
            pretrained=pretrained,
            init_from_pretrained_center=init_cifar_stem_from_pretrained_center,
        )

    return model


def _cifar_root(config: TrainingConfig) -> str:
    return f"{config.data_dir}/cifar"


def _imagenet_root(config: TrainingConfig) -> str:
    return f"{config.data_dir}/imagenet"


def get_data_loaders(
    config: TrainingConfig,
    rank: int,
    world_size: int,
) -> tuple:
    if config.dataset == "cifar10":
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
        ds_cls = datasets.CIFAR10

        train_dataset = ds_cls(
            root=_cifar_root(config),
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
        )

        val_dataset = ds_cls(
            root=_cifar_root(config),
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
        )

    elif config.dataset == "cifar100":
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761],
        )
        ds_cls = datasets.CIFAR100

        train_dataset = ds_cls(
            root=_cifar_root(config),
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
        )

        val_dataset = ds_cls(
            root=_cifar_root(config),
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
        )

    elif config.dataset in ("imagenet", "imagenet_like"):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        train_dataset = datasets.ImageFolder(
            root=f"{_imagenet_root(config)}/train",
            transform=transforms.Compose([
                transforms.RandomResizedCrop(config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
        )

        val_dataset = datasets.ImageFolder(
            root=f"{_imagenet_root(config)}/val",
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(config.image_size),
                transforms.ToTensor(),
                normalize,
            ]),
        )

    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, train_sampler