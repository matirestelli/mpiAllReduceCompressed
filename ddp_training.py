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
import math
import os
import sys
import time
import csv
from datetime import datetime

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
    verify_last_batch: bool = False,
) -> tuple:
    """Train for one epoch. Returns (avg_loss, accuracy).

    If verify_last_batch=True, calls verify_gradient_sync on the very last
    batch AFTER loss.backward() but BEFORE optimizer.step() — so it checks
    the real gradients that the optimizer is about to consume.
    """
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

        # ── On the last batch of the final epoch, verify real grad sync ──
        # if verify_last_batch and batch_idx == len(train_loader) - 1:
        #     if rank == 0:
        #         print("\n[Verify] Gradient sync check on REAL last training batch "
        #               "(after backward, before step)...")
        #     verify_gradient_sync(model, rank, dist.get_world_size())

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


def print_gradient_stats(model: DDP, epoch: int, rank: int, max_params: int = 10):
    """
    Print gradient values for the first `max_params` named parameters (rank 0 only).
    Called at the end of each epoch. Matches the style of verify_gradient_sync:
    first 4 elements of each gradient + how many more exist.
    Uses the gradients still attached from the final batch's backward().
    """
    if rank != 0:
        return
    print(f"\n[Epoch {epoch}] Gradients (first {max_params} params):")
    printed = 0
    for name, param in model.named_parameters():
        if printed >= max_params:
            break
        if param.grad is None:
            print(f"  {name:60s}  grad=None")
        else:
            g = param.grad.detach().cpu()
            n = g.numel()
            sample = g.flatten()[:4].tolist()
            extra  = n - len(sample)
            val_str = (f"{sample} ... ({extra} more elements)"
                       if extra > 0 else f"{sample}")
            print(f"  {name:60s}  grad={val_str}")
        printed += 1
    print(flush=True)


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
        _local_rank = mpi_rank % torch.cuda.device_count()
        torch.cuda.set_device(_local_rank)
        dist.init_process_group(
            backend=config.backend,
            device_id=torch.device(f"cuda:{_local_rank}"),
        )
    else:
        dist.init_process_group(backend=config.backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ── Setup logging and CSV (rank 0 only) ──────────────────────────────
    csv_file   = None
    csv_writer = None
    if rank == 0:
        algo_str = config.comm_algorithm or "builtin"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = (
            f"{config.model_name}_{config.num_epochs}_"
            f"{config.backend}_{algo_str}.log"
        )
        log_file = open(log_filename, 'w')
        sys.stdout = Tee(sys.__stdout__, log_file)

        os.makedirs("results", exist_ok=True)
        csv_path   = (f"results/{config.model_name}_{config.backend}_"
                      f"{algo_str}_{timestamp}.csv")
        csv_file   = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch', 'lr', 'train_loss', 'train_acc',
            'val_loss', 'val_acc', 'epoch_time_s',
            'model', 'backend', 'algorithm', 'world_size',
            'batch_size', 'num_epochs',
        ])

    # Seed for reproducible model init (DDP broadcasts rank 0's params anyway)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Standard paper setting: let cuDNN benchmark and pick the fastest kernels.
    torch.backends.cudnn.benchmark = True

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

    # Model — pass cifar_stem explicitly from config
    model = create_model(
        config.model_name, config.num_classes, cifar_stem=config.cifar_stem
    ).to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        # find_unused_parameters=False: ResNet50 uses all parameters every forward.
        # Setting True triggers an extra autograd graph traversal every iteration.
        find_unused_parameters=False,
        # static_graph=True: tells DDP the autograd graph never changes.
        # Buckets are built once and never rebuilt, so we don't need a dummy
        # backward pass to force a rebuild.  This also makes bucket-to-hook
        # dispatch permanently stable — no risk of hooks being silently
        # dropped after a rebuild.
        static_graph=True,
    )

    if rank == 0:
        print(f"[Setup] Model on {device}, wrapped in DDP (static_graph=True)")

    # ── Communication hook ───────────────────────────────────────────────
    # get_comm_hook returns (hook, state) or None.
    # The ring hook requires a RingAllreduceState as its state so that
    # persistent GPU buffers can be reused across backward passes — this
    # is what fixes the Cray MPICH GPU-direct RDMA stale-registration bug.
    comm_hook_result = get_comm_hook(config.backend, config.comm_algorithm)
    if comm_hook_result is not None:
        comm_hook, comm_state = comm_hook_result
        model.register_comm_hook(state=comm_state, hook=comm_hook)
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

    # ── BUG FIX: LR warmup ──────────────────────────────────────────────
    # ResNet50 from scratch on 32×32 images diverges with lr=0.01 from step 1
    # because the randomly-initialised deep network produces large gradient
    # norms that, combined with momentum=0.9, cause weight explosion.
    # A linear warmup over the first epoch lets BN statistics stabilise and
    # gradient norms settle before the full learning rate kicks in.
    #
    # The scheduler chains warmup → cosine annealing:
    #   Epoch 1..warmup_epochs:  lr ramps 0 → config.learning_rate linearly
    #   Epoch warmup_epochs+1..: cosine annealing to 0
    warmup_epochs = getattr(config, 'warmup_epochs', min(5, max(1, config.num_epochs // 10)))

    def lr_lambda(step: int) -> float:
        if step < warmup_epochs:
            return 1e-3 + (1.0 - 1e-3) * step / max(warmup_epochs, 1)
        progress = (step - warmup_epochs) / max(config.num_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
    epoch_times = []  # track per-epoch wall time

    for epoch in range(1, config.num_epochs + 1):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch}] Training... (lr={cur_lr:.6f})")

        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, rank, device,
            verify_last_batch=(epoch == config.num_epochs),
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        epoch_lr = optimizer.param_groups[0]['lr']   # LR used this epoch (before step)
        scheduler.step()
        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)

        if rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, "
                  f"Acc: {train_acc:.2f}%  |  "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%  |  "
                  f"Time: {epoch_elapsed:.1f}s")
            csv_writer.writerow([
                epoch, f"{epoch_lr:.8f}",
                f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{val_loss:.6f}",  f"{val_acc:.4f}",
                f"{epoch_elapsed:.2f}",
                config.model_name, config.backend, algo_str,
                world_size, config.batch_size, config.num_epochs,
            ])
            csv_file.flush()

            # NaN/Inf detection — catches exploding gradients from bad hooks
            if not torch.isfinite(torch.tensor(train_loss)):
                print(f"  WARNING: train_loss is {train_loss} — "
                      f"possible gradient explosion from hook bug. "
                      f"Check that hook divides by world_size exactly once.")

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

        # ── Per-epoch gradient stats (rank 0 only, first 10 params) ─────
        # print_gradient_stats(model, epoch, rank, max_params=10)

    elapsed = time.time() - start_time

    # ── Final summary ────────────────────────────────────────────────────
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"Done in {elapsed:.1f}s | Best val acc: {best_val_acc:.2f}%")
        if epoch_times:
            import statistics
            print(f"Epoch times: mean={statistics.mean(epoch_times):.1f}s  "
                  f"min={min(epoch_times):.1f}s  "
                  f"max={max(epoch_times):.1f}s")
        print("=" * 80)

    if csv_file is not None:
        csv_file.close()

    if config.backend == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
    dist.destroy_process_group()