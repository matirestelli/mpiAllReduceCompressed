"""
Training Configuration & Interface

User-facing interface to configure distributed training parameters.

Usage:
    mpirun -np 8 --ppn 8 --depth=8 --cpu-bind depth \
        -env MPICH_GPU_SUPPORT_ENABLED=1 \
        python interface.py

Optional job-script overrides:
    BACKEND=mpi|nccl
    COMM_ALGORITHM=default|ring|recursive_doubling|ring_zfp_naive|...

The values selected below remain the defaults. Environment variables only
override them for the current launched training run.
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


def _read_float_env(name):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


def apply_job_overrides(config):
    """Allow PBS/train job scripts to override backend, hook, and ZFP rate."""
    backend = _read_optional_env("BACKEND")
    comm_algorithm = _read_optional_env("COMM_ALGORITHM")
    zfp_rate = _read_float_env("ZFP_RATE")

    if backend is not None:
        config.backend = backend

    if comm_algorithm is not None:
        config.comm_algorithm = _normalize_optional_value(comm_algorithm)

    if zfp_rate is not None:
        config.zfp_rate = zfp_rate


if __name__ == "__main__":

    config = TrainingConfig(
        # -- Model (uncomment one, comment the rest) -------------------------
        # model_name="resnet18",           # ~11M params - fastest, lightweight baseline
        # model_name="resnet50",           # ~25M params - standard CIFAR-10 benchmark
            # if model resnet50 also uncomment this next 2 instructions
        # grad_clip=None,
        # warmup_epochs=1,
        # model_name="resnet101",          # ~44M params - deeper ResNet
        model_name="wide_resnet50_2",
        # model_name="resnext101_32x8d",   # ~88M params - grouped convs (32 groups x 8 width)
        # model_name="convnext_tiny",      # ~28M params - modern architecture, no stem fix needed
        # model_name="convnext_small",     # ~50M params - larger ConvNeXt variant
        dataset="cifar10",
        num_classes=10,
        num_epochs=20,           # big to see if it trains all
        # -- Batch size ------------------------------------------------------
        # Memory: fixed ~1 GB (params+grads+optimizer) + ~50-80 MB/sample activations.
        # resnext101 on CIFAR-10 (32x32): ~8-12 GB at batch_size=128 on A100 -> comfortable.
        # Go to 64 if OOM. Go to 256 (and scale lr proportionally) if you want larger effective batch.
        # Quality: effective_batch = batch_size x world_size. Keep <= 2048 for CIFAR-10
        # without LR adjustment. At 4 GPUs x 128 = 512 effective - conservative and safe.
        # the total batch size is 128 -> so based on the number of gpus here set: 128/Number of GPUs
        # batch_size=32, # for 4 gpus
        # batch_size=16, # for 8 gpus
        batch_size=8, # for 16 gpus
        # no wait in this way goes slower with 8 gpus.
        #   4 GPUs  -> effective batch 512
        #   8 GPUs  -> effective batch 1024
        #   16 GPUs -> effective batch 2048
        # batch_size=128,
        learning_rate=0.001,
        momentum=0.9,
        weight_decay=5e-4,
        warmup_epochs=1,
        grad_clip=None,
        backend="mpi", # mpi or nccl; can be overridden by BACKEND in trainJobs.sh
        # -- Communication algorithm (uncomment one, comment the rest) -------
        # comm_algorithm=None,
        comm_algorithm="default",                   # built-in wrapped with NVTX timing only
        # comm_algorithm="ring",                    # custom ring: reduce-scatter + allgather via P2P
        # comm_algorithm="recursive_doubling",      # MPICH recursive doubling, handles non-power-of-2
        # comm_algorithm="ring_zfp_naive",          # ring + ZFP compression (naive paper baseline)
        # comm_algorithm="recursive_doubling_zfp_naive",  # recursive doubling + ZFP compression
        # comm_algorithm="ring_zfp_online_coll",    # Algorithm 1: ring + ZFP collective-level online compression
        # comm_algorithm="recursive_doubling_zfp_online_coll",
        # comm_algorithm="ring_async"
        # grad_clip defaults to 1.0 in TrainingConfig (model-agnostic universal
        # default). Override here only to disable (None) or tune the threshold.
        # pretrained=True required for ResNeXt101/ConvNeXt on CIFAR-10 - see
        # TRAINING_STABILITY.md for why training from scratch is not viable.
        zfp_rate=16.0,
        pretrained=True,
        cifar_stem=False,
        data_dir="./data",
        checkpoint_dir="./checkpoints",
        seed=42,
        )

    # Keep this interface as the visible default, but allow the train job to
    # run sweeps without editing this file between queued jobs.
    apply_job_overrides(config)

    # ================================================================
    # BENCHMARKING EXAMPLES
    # ================================================================

    # To benchmark ring allreduce:
    # config.comm_algorithm = "ring"

    # To benchmark recursive doubling:
    # config.comm_algorithm = "recursive_doubling"

    # To use NCCL backend instead:
    # config.backend = "nccl"
    # config.comm_algorithm = "ring"

    # ================================================================

    print("\n" + "=" * 80)
    print("DISTRIBUTED TRAINING LAUNCHER")
    print("=" * 80)
    print(f"  Model:      {config.model_name}"
          f"{' (CIFAR stem)' if config.cifar_stem else ''}"
          f"{' (pretrained)' if config.pretrained else ' (from scratch)'}")
    print(f"  Dataset:    {config.dataset} ({config.num_classes} classes)")
    print(f"  Backend:    {config.backend}")
    print(f"  Algorithm:  {config.comm_algorithm or 'built-in (no hook)'}")
    if config.comm_algorithm and "zfp" in config.comm_algorithm:
        print(f"  ZFP rate:   {config.zfp_rate:g}")
    print(f"  Epochs:     {config.num_epochs}")
    print(f"  Batch/rank: {config.batch_size}")
    print(f"  LR:         {config.learning_rate}")
    print("=" * 80 + "\n")

    train(config)