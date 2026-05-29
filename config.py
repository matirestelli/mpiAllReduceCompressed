

config.py


New project
ddp-allreduce-eval-framework
config.py



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
    """Complete training configuration.

    Important batch-size convention:
        batch_size is always the local/per-rank batch size.
        global_batch_size = batch_size * world_size.

    For strong scaling, manually set:
        batch_size = desired_global_batch_size // world_size

    For weak scaling, keep:
        batch_size = desired_local_batch_size
    """

    # Model & data
    model_name: str = "resnet50"
    dataset: str = "cifar10"  # "cifar10", "cifar100", or "imagenet"
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
    # Use "constant" for paper-style LR=0.001 reproduction.
    # Use "cosine" for improved CIFAR/ImageNet-style experiments.
    scheduler: str = "constant"  # "constant" or "cosine"
    warmup_epochs: int = 0

    # DataLoader behavior
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False

    # Distributed training
    backend: str = "mpi"  # "nccl" or "mpi"
    comm_algorithm: Optional[str] = None

    # ZFP compression
    zfp_rate: float = 16.0

    # Model initialization
    # CIFAR from scratch:
    #     pretrained=False, cifar_stem=True
    # ImageNet/ImageNet-like from scratch:
    #     pretrained=False, cifar_stem=False
    # ImageNet/ImageNet-like fine-tuning:
    #     pretrained=True, cifar_stem=False
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