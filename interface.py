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
        model_name="resnet50",
        dataset="cifar10",
        num_classes=10,
        num_epochs=15,
        batch_size=128,
        learning_rate=0.01,     # safe default for CIFAR + 4-GPU DDP
        momentum=0.9,
        weight_decay=5e-4,
        backend="mpi",
        comm_algorithm=None,    # None = DDP built-in (fastest)
                                # "default" = our baseline hook
                                # "ring" = ring allreduce
                                # "recursive_doubling" = recursive doubling
        cifar_stem=True,        # adapt ResNet stem for 32x32 images
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
          f"{' (CIFAR stem)' if config.cifar_stem else ''}")
    print(f"  Dataset:    {config.dataset} ({config.num_classes} classes)")
    print(f"  Backend:    {config.backend}")
    print(f"  Algorithm:  {config.comm_algorithm or 'built-in (no hook)'}")
    print(f"  Epochs:     {config.num_epochs}")
    print(f"  Batch/rank: {config.batch_size}")
    print(f"  LR:         {config.learning_rate}")
    print("=" * 80 + "\n")

    train(config)