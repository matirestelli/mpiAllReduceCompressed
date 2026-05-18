"""
Training Configuration & Interface

User-facing interface to configure distributed training parameters.

Usage:
    mpirun -np 4 --ppn 4 --depth=8 --cpu-bind depth \
        -env MPICH_GPU_SUPPORT_ENABLED=1 \
        python interface.py
"""

from config import TrainingConfig
from ddp_training import train


if __name__ == "__main__":

    config = TrainingConfig(
        # ── Model (uncomment one, comment the rest) ──────────────────────────
        # model_name="resnet18",           # ~11M params — fastest, lightweight baseline
        model_name="resnet50",           # ~25M params — standard CIFAR-10 benchmark
        # model_name="resnet101",          # ~44M params — deeper ResNet
        # model_name="resnext101_32x8d",     # ~88M params — grouped convs (32 groups × 8 width)
        # model_name="convnext_tiny",      # ~28M params — modern architecture, no stem fix needed
        # model_name="convnext_small",     # ~50M params — larger ConvNeXt variant
        dataset="cifar10",
        num_classes=10,
        num_epochs=10,           # big to see if it trains all
        # ── Batch size ────────────────────────────────────────────────────────
        # Memory: fixed ~1 GB (params+grads+optimizer) + ~50-80 MB/sample activations.
        # resnext101 on CIFAR-10 (32×32): ~8-12 GB at batch_size=128 on A100 → comfortable.
        # Go to 64 if OOM. Go to 256 (and scale lr proportionally) if you want larger effective batch.
        # Quality: effective_batch = batch_size × world_size. Keep ≤ 2048 for CIFAR-10
        # without LR adjustment. At 4 GPUs × 128 = 512 effective — conservative and safe.
        batch_size=128,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        backend="mpi", # mpi or nccl
        # ── Communication algorithm (uncomment one, comment the rest) ────────
        # comm_algorithm="recursive_doubling_zfp",                        # DDP built-in allreduce (no hook registered)
        # comm_algorithm="default",                   # built-in wrapped with NVTX timing only
        # comm_algorithm="ring",                      # custom ring: reduce-scatter + allgather via P2P
        # comm_algorithm="recursive_doubling",        # MPICH recursive doubling, handles non-power-of-2
        # comm_algorithm="ring_zfp",                  # ring + ZFP compression (naive paper baseline)
        # comm_algorithm="ring_zfp_naive",            # alias for ring_zfp
        # comm_algorithm="recursive_doubling_zfp",     # recursive doubling + ZFP compression
        # comm_algorithm="recursive_doubling_zfp_naive",  # alias for recursive_doubling_zfp
        comm_algorithm="ring_zfp_online_coll",     # Algorithm 1: ring + ZFP collective-level online compression
        # grad_clip defaults to 1.0 in TrainingConfig (model-agnostic universal
        # default). Override here only to disable (None) or tune the threshold.
        # pretrained=True required for ResNeXt101/ConvNeXt on CIFAR-10 — see
        # TRAINING_STABILITY.md for why training from scratch is not viable.
        # pretrained=True,
        cifar_stem=True,
        data_dir="./data",
        checkpoint_dir="./checkpoints",
        seed=42,
    )

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
    print(f"  Epochs:     {config.num_epochs}")
    print(f"  Batch/rank: {config.batch_size}")
    print(f"  LR:         {config.learning_rate}")
    print("=" * 80 + "\n")

    train(config)