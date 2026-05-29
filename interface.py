"""
Training Configuration & Interface

User-facing interface to configure distributed training parameters.

Usage:
    mpirun -np 8 --ppn 8 --depth=8 --cpu-bind depth \
        -env MPICH_GPU_SUPPORT_ENABLED=1 \
        python interface.py

You can configure experiments in two ways:
    1. Edit the TrainingConfig defaults below for a single/manual run.
    2. Set environment variables from a job script to run sweeps without
       editing this file between queued jobs.

Environment variables only override values when they are set.
"""

import os

from config import TrainingConfig
from ddp_training import train


def _read_optional_env(name):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def _normalize_optional_value(value):
    if value.lower() in {"none", "null"}:
        return None
    return value


def _read_int_env(name):
    value = _read_optional_env(name)
    if value is None:
        return None
    value = _normalize_optional_value(value)
    if value is None:
        return None
    return int(value)


def _read_float_env(name):
    value = _read_optional_env(name)
    if value is None:
        return None
    value = _normalize_optional_value(value)
    if value is None:
        return None
    return float(value)


def _read_bool_env(name):
    value = _read_optional_env(name)
    if value is None:
        return None

    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"Invalid boolean value for {name}: {value}")


def _set_if_present(config, attr, value):
    if value is not None:
        setattr(config, attr, value)


def apply_job_overrides(config):
    """Allow PBS/train job scripts to override experiment settings."""
    _set_if_present(config, "model_name", _read_optional_env("MODEL_NAME"))
    _set_if_present(config, "dataset", _read_optional_env("DATASET"))
    _set_if_present(config, "num_classes", _read_int_env("NUM_CLASSES"))
    _set_if_present(config, "image_size", _read_int_env("IMAGE_SIZE"))

    _set_if_present(config, "num_epochs", _read_int_env("NUM_EPOCHS"))
    _set_if_present(config, "batch_size", _read_int_env("BATCH_SIZE"))
    _set_if_present(config, "learning_rate", _read_float_env("LEARNING_RATE"))
    _set_if_present(config, "momentum", _read_float_env("MOMENTUM"))
    _set_if_present(config, "weight_decay", _read_float_env("WEIGHT_DECAY"))
    _set_if_present(config, "grad_clip", _read_float_env("GRAD_CLIP"))

    _set_if_present(config, "scheduler", _read_optional_env("SCHEDULER"))
    _set_if_present(config, "warmup_epochs", _read_int_env("WARMUP_EPOCHS"))

    _set_if_present(config, "num_workers", _read_int_env("NUM_WORKERS"))
    _set_if_present(config, "pin_memory", _read_bool_env("PIN_MEMORY"))
    _set_if_present(config, "drop_last", _read_bool_env("DROP_LAST"))

    _set_if_present(config, "backend", _read_optional_env("BACKEND"))

    comm_algorithm = _read_optional_env("COMM_ALGORITHM")
    if comm_algorithm is not None:
        config.comm_algorithm = _normalize_optional_value(comm_algorithm)

    _set_if_present(config, "zfp_rate", _read_float_env("ZFP_RATE"))

    _set_if_present(config, "pretrained", _read_bool_env("PRETRAINED"))
    _set_if_present(config, "cifar_stem", _read_bool_env("CIFAR_STEM"))
    _set_if_present(
        config,
        "init_cifar_stem_from_pretrained_center",
        _read_bool_env("INIT_CIFAR_STEM_FROM_PRETRAINED_CENTER"),
    )

    _set_if_present(config, "data_dir", _read_optional_env("DATA_DIR"))
    _set_if_present(config, "checkpoint_dir", _read_optional_env("CHECKPOINT_DIR"))
    _set_if_present(config, "seed", _read_int_env("SEED"))


if __name__ == "__main__":

    config = TrainingConfig(
        # Model
        # model_name="resnet18",
        # model_name="resnet50",
        # model_name="resnet101",
        # model_name="wide_resnet50_2",
        model_name="resnext101_32x8d",
        # model_name="convnext_tiny",
        # model_name="convnext_small",

        # Dataset
        dataset="cifar10",
        # dataset="imagenet",
        num_classes=10,
        image_size=32,

        # Training
        num_epochs=20,
        batch_size=16,          # local/per-rank batch size
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=5e-4,
        grad_clip=None,

        # Scheduler
        scheduler="constant",   # paper-style LR=0.001 reproduction
        # scheduler="cosine",
        warmup_epochs=0,

        # DataLoader
        num_workers=4,
        pin_memory=True,
        drop_last=False,

        # Distributed / communication
        backend="mpi",
        # comm_algorithm=None,
        comm_algorithm="default",
        # comm_algorithm="ring",
        # comm_algorithm="recursive_doubling",
        # comm_algorithm="ring_zfp_naive",
        # comm_algorithm="recursive_doubling_zfp_naive",
        # comm_algorithm="ring_zfp_online_coll",
        # comm_algorithm="recursive_doubling_zfp_online_coll",
        # comm_algorithm="ring_async",
        zfp_rate=16.0,

        # Model initialization
        pretrained=False,
        cifar_stem=True,
        init_cifar_stem_from_pretrained_center=False,

        # Paths
        data_dir="./data",
        checkpoint_dir="./checkpoints",
        seed=42,
    )

    apply_job_overrides(config)

    print("\n" + "=" * 80)
    print("DISTRIBUTED TRAINING LAUNCHER")
    print("=" * 80)
    print(f"  Model:      {config.model_name}"
          f"{' (CIFAR stem)' if config.cifar_stem else ''}"
          f"{' (pretrained)' if config.pretrained else ' (from scratch)'}")
    print(f"  Dataset:    {config.dataset} ({config.num_classes} classes)")
    print(f"  Image size: {config.image_size}")
    print(f"  Backend:    {config.backend}")
    print(f"  Algorithm:  {config.comm_algorithm or 'built-in (no hook)'}")
    if config.comm_algorithm and "zfp" in config.comm_algorithm:
        print(f"  ZFP rate:   {config.zfp_rate:g}")
    print(f"  Epochs:     {config.num_epochs}")
    print(f"  Batch/rank: {config.batch_size}")
    print(f"  LR:         {config.learning_rate}")
    print(f"  Scheduler:  {config.scheduler}")
    print(f"  Warmup:     {config.warmup_epochs}")
    print(f"  Grad clip:  {config.grad_clip}")
    print(f"  Drop last:  {config.drop_last}")
    print(f"  Data dir:   {config.data_dir}")
    print("=" * 80 + "\n")

    train(config)