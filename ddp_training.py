"""
Distributed Data Parallel Training

Core training loop using PyTorch DDP with configurable communication strategies.
Supports both NCCL and MPI backends with custom AllReduce hooks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys
import time

from config import TrainingConfig, create_model, get_data_loaders
from communication_strategy import get_comm_hook


class Tee:
    """Duplicate output to console and log file simultaneously."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


def train_epoch(
    model: DDP,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    rank: int,
    device: torch.device,
) -> tuple:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if rank == 0 and (batch_idx + 1) % 50 == 0:
            accuracy = 100.0 * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(
    model: DDP,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Validate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(val_loader), 100.0 * correct / total


def verify_gradient_sync(model: DDP, rank: int, world_size: int):
    """
    Check that gradients are identical across all ranks after backward().
    Call this AFTER loss.backward() but BEFORE optimizer.step() to verify
    that the allreduce (whether built-in or custom hook) is working.
    """
    first_grad = list(model.parameters())[0].grad
    if first_grad is None:
        print(f"[Rank {rank}] WARNING: no gradient on first param")
        return

    grad_sample = first_grad.detach().cpu().flatten()[:5].tolist()
    dist.barrier()
    for r in range(world_size):
        if rank == r:
            print(f"[Rank {rank}] grad[:5] = "
                  f"{[f'{v:.10f}' for v in grad_sample]}", flush=True)
        dist.barrier()


def train(config: TrainingConfig) -> None:
    """
    Main training function.

    Initializes distributed training, creates model and dataloaders,
    optionally registers a communication hook, and runs the training loop.
    """
    config.validate()

    # ── Initialize process group ─────────────────────────────────────────
    if config.backend == "nccl":
        # NCCL uses env:// init — set env vars from MPI rank info
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        mpi_world_size = MPI.COMM_WORLD.Get_size()
        os.environ['RANK'] = str(mpi_rank)
        os.environ['WORLD_SIZE'] = str(mpi_world_size)
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '29500')

    dist.init_process_group(backend=config.backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ── Setup logging to file (rank 0 only) ──────────────────────────────
    if rank == 0:
        algo_str = config.comm_algorithm or "builtin"
        log_filename = (
            f"{config.model_name}_{config.num_epochs}_"
            f"{config.backend}_{algo_str}.log"
        )
        log_file = open(log_filename, 'w')
        sys.stdout = Tee(sys.__stdout__, log_file)

    # Seed for reproducible model init (DDP broadcasts rank 0's params anyway)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    if rank == 0:
        print("\n" + "=" * 80)
        print(f"Distributed Training: {config.model_name.upper()} on "
              f"{config.dataset.upper()}")
        print(f"Backend: {config.backend.upper()}, "
              f"Algorithm: {config.comm_algorithm or 'built-in (no hook)'}")
        print("=" * 80)
        print(f"Ranks: {world_size}, Epochs: {config.num_epochs}, "
              f"Batch/rank: {config.batch_size}, "
              f"Effective batch: {config.batch_size * world_size}")
        print(f"LR: {config.learning_rate}, Device: {device}\n")

    # Model 
    model = create_model(config.model_name, config.num_classes).to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=True)

    if rank == 0:
        print(f"[Setup] Model on {device}, wrapped in DDP")

    # ── Communication hook ───────────────────────────────────────────────
    comm_hook = get_comm_hook(config.backend, config.comm_algorithm)
    if comm_hook is not None:
        model.register_comm_hook(state=None, hook=comm_hook)
        if rank == 0:
            print(f"[Setup] Hook registered: {config.backend}/"
                  f"{config.comm_algorithm}")
    else:
        if rank == 0:
            print(f"[Setup] No hook — using DDP built-in "
                  f"{config.backend.upper()} allreduce")

    # ── Optimizer + scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, train_sampler = get_data_loaders(
        config, rank, world_size
    )
    if rank == 0:
        print(f"[Data] Train batches: {len(train_loader)}, "
              f"Val batches: {len(val_loader)}\n")

    # ── Training loop ────────────────────────────────────────────────────
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n[Epoch {epoch}] Training...")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, rank, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        if rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, "
                  f"Acc: {train_acc:.2f}%")
            print(f"[Epoch {epoch}] Val   Loss: {val_loss:.4f}, "
                  f"Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                path = os.path.join(
                    config.checkpoint_dir, f"{config.model_name}_best.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, path)
                print(f"  -> Best model saved ({val_acc:.2f}%)")

    elapsed = time.time() - start_time

    # ── Final summary ────────────────────────────────────────────────────
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"Done in {elapsed:.1f}s | Best val acc: {best_val_acc:.2f}%")
        print("=" * 80)

    # ── Gradient sync verification (on first param, proves hook works) ───
    # Do one dummy forward-backward to get fresh gradients for checking
    if rank == 0:
        print("\n[Verify] Checking gradient synchronization across ranks...")
    model.train()
    dummy_input = torch.randn(2, 3, 32, 32, device=device)
    dummy_target = torch.zeros(2, dtype=torch.long, device=device)
    optimizer.zero_grad()
    loss = criterion(model(dummy_input), dummy_target)
    loss.backward()
    verify_gradient_sync(model, rank, world_size)

    dist.barrier()
    dist.destroy_process_group()