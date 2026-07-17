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
import json
import statistics
import socket
from datetime import datetime

from config import TrainingConfig, create_model, get_data_loaders
from communication_strategy import (
    get_comm_hook,
    open_first_bucket_compute_range,
    reset_bucket_compute_markers,
    set_profiling_step,
    summarize_hook_timing,
)

ENABLE_NVTX_PROFILING = os.getenv("DDP_PROFILE_NVTX", "0") == "1"

# ── FIX 3: per-iteration timing ────────────────────────────────────────────
# Barrier-before-backward isolates load imbalance from communication cost, but
# it suppresses backward/allreduce overlap. DEFAULT OFF. Enable it only for the
# one-off comm-breakdown run; never for headline throughput numbers.
ENABLE_PROFILE_BARRIER = os.getenv("DDP_PROFILE_BARRIER", "0") == "1"

# Write per-iteration JSONL from every rank (straggler / variance analysis).
# OFF by default: at 64 ranks this is ~75 MB per run. The per-epoch CSV medians
# are computed in memory regardless, so nothing in the CSV depends on this.
# Turn on (DDP_ITER_LOG=1) only for a few representative configs.
ENABLE_ITER_LOG = os.getenv("DDP_ITER_LOG", "0") == "1"


def _nvtx_range_push(msg: str) -> None:
    if ENABLE_NVTX_PROFILING and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(msg)


def _nvtx_range_pop() -> None:
    if ENABLE_NVTX_PROFILING and torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


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


# ── FIX 1: weight decay parameter groups ───────────────────────────────────
def build_param_groups(model, weight_decay: float, wd_on_bn_bias: bool):
    """
    Split parameters into decay / no-decay groups.

    wd_on_bn_bias=False  (B1, recommended):
        BatchNorm gamma/beta and ALL biases get weight_decay=0.0.
        Everything else (conv + linear weight matrices) gets `weight_decay`.
        Rule: any parameter with ndim <= 1 is BN gamma, BN beta, or a bias.

    wd_on_bn_bias=True   (B4, == the original behaviour):
        Every parameter gets `weight_decay`. Returns a single group.

    Returns a list of param-group dicts for the optimizer.
    """
    if wd_on_bn_bias:
        return [{
            "params": [p for p in model.parameters() if p.requires_grad],
            "weight_decay": weight_decay,
        }]

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def train_epoch(
    model: DDP,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,                      # FIX 2: stepped per-iteration
    epoch: int,
    rank: int,
    device: torch.device,
    verify_last_batch: bool = False,
    grad_clip=None,
    iter_log_path=None,             # FIX 3
) -> tuple:
    """Train for one epoch. Returns (avg_loss, accuracy, iter_stats).

    iter_stats is a dict of per-iteration timing summaries for this epoch.

    Timing note: torch.cuda.Event.record() is a non-blocking stream enqueue.
    elapsed_time() is NEVER called inside the loop (that would force a host
    sync); events are buffered and drained once after the loop. This adds zero
    synchronization relative to the original code, which already syncs twice
    per iteration via loss.item() and .sum().item().
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    epoch_t0 = time.time()
    last_t = epoch_t0
    last_print_batch = 0

    # FIX 3: buffered timing state — no elapsed_time() calls inside the loop.
    ev_buf = []      # list of (e_start, e_fwd, e_bar0, e_bar1, e_bwd, e_opt)
    data_ms = []     # host-side loader wait per iteration

    t_data_start = time.perf_counter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_ms.append((time.perf_counter() - t_data_start) * 1e3)

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        set_profiling_step(epoch, batch_idx)

        e_start = torch.cuda.Event(enable_timing=True)
        e_fwd   = torch.cuda.Event(enable_timing=True)
        e_bar0  = torch.cuda.Event(enable_timing=True)
        e_bar1  = torch.cuda.Event(enable_timing=True)
        e_bwd   = torch.cuda.Event(enable_timing=True)
        e_opt   = torch.cuda.Event(enable_timing=True)

        _nvtx_range_push(f"iteration epoch={epoch} batch={batch_idx}")
        try:
            e_start.record()

            _nvtx_range_push(f"forward epoch={epoch} batch={batch_idx}")
            try:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            finally:
                _nvtx_range_pop()

            e_fwd.record()

            # Optional: isolate load imbalance from comm. OFF by default.
            e_bar0.record()
            if ENABLE_PROFILE_BARRIER:
                torch.cuda.synchronize()
                dist.barrier()
            e_bar1.record()

            _nvtx_range_push(f"backward epoch={epoch} batch={batch_idx}")
            try:
                reset_bucket_compute_markers()
                open_first_bucket_compute_range()
                loss.backward()
            finally:
                reset_bucket_compute_markers()
                _nvtx_range_pop()

            e_bwd.record()

            # ── On the last batch of the final epoch, verify real grad sync ──
            # if verify_last_batch and batch_idx == len(train_loader) - 1:
            #     if rank == 0:
            #         print("\n[Verify] Gradient sync check on REAL last training batch "
            #               "(after backward, before step)...")
            #     verify_gradient_sync(model, rank, dist.get_world_size())

            # Gradient clipping — after allreduce (backward complete), before optimizer.
            # Needed for deep models (ResNeXt101+) where gradient norms at init are
            # large enough to cause explosion even at warmup LR via momentum accumulation.
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            _nvtx_range_push(f"optimizer_step epoch={epoch} batch={batch_idx}")
            try:
                optimizer.step()
                scheduler.step()          # FIX 2: per-iteration cosine/warmup
            finally:
                _nvtx_range_pop()

            e_opt.record()
        finally:
            _nvtx_range_pop()

        ev_buf.append((e_start, e_fwd, e_bar0, e_bar1, e_bwd, e_opt))

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if rank == 0 and (batch_idx + 1) % 50 == 0:
            accuracy = 100.0 * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%, lr: {cur_lr:.6f}")

        t_data_start = time.perf_counter()

    # ── FIX 3: drain the event buffer ONCE, after the loop ─────────────────
    torch.cuda.synchronize()

    records = []
    for i, (e_start, e_fwd, e_bar0, e_bar1, e_bwd, e_opt) in enumerate(ev_buf):
        t_fwd = e_start.elapsed_time(e_fwd)
        t_bar = e_bar0.elapsed_time(e_bar1)
        t_bwd = e_bar1.elapsed_time(e_bwd)      # compute + exposed allreduce
        t_opt = e_bwd.elapsed_time(e_opt)
        t_gpu = e_start.elapsed_time(e_opt)
        records.append({
            "epoch": epoch, "iter": i, "rank": rank,
            "t_data_ms":    round(data_ms[i], 4),
            "t_fwd_ms":     round(t_fwd, 4),
            "t_barrier_ms": round(t_bar, 4),
            "t_bwd_ms":     round(t_bwd, 4),
            "t_opt_ms":     round(t_opt, 4),
            "t_gpu_ms":     round(t_gpu, 4),
            "t_iter_ms":    round(t_gpu + data_ms[i], 4),
        })

    if iter_log_path is not None and records:
        with open(iter_log_path, 'a') as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _med(key):
        return statistics.median([r[key] for r in records]) if records else 0.0

    iter_stats = {
        "t_iter_median_ms":    _med("t_iter_ms"),
        "t_fwd_median_ms":     _med("t_fwd_ms"),
        "t_bwd_median_ms":     _med("t_bwd_ms"),
        "t_opt_median_ms":     _med("t_opt_ms"),
        "t_data_median_ms":    _med("t_data_ms"),
        "t_barrier_median_ms": _med("t_barrier_ms"),
        # Train-only epoch time: excludes validation, checkpointing, logging.
        "t_epoch_train_s":     sum(r["t_iter_ms"] for r in records) / 1e3,
        "num_iters":           len(records),
    }

    return total_loss / len(train_loader), 100.0 * correct / total, iter_stats


def validate(model, val_loader, criterion, device):
    """Validate on the FULL unsharded test set. Contains NO collective."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # weight by sample count so the final divide is exact
            total_loss += criterion(outputs, targets).item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / total, 100.0 * correct / total


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
        # NCCL uses env:// init.
        # On Slurm systems like Frontier, use Slurm rank metadata.
        # On mpiexec-based systems like Polaris, fall back to MPI rank metadata.
        if "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])

            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")

            # With --gpus-per-task=1, each task typically sees only one GPU.
            local_rank = 0
            torch.cuda.set_device(local_rank)

            print(
                f"[PRE-INIT] rank={rank}/{world_size} "
                f"host={socket.gethostname()} "
                f"local_rank={local_rank} "
                f"current_device={torch.cuda.current_device()} "
                f"device_count={torch.cuda.device_count()} "
                f"visible={os.environ.get('ROCR_VISIBLE_DEVICES')} "
                f"device_name={torch.cuda.get_device_name(torch.cuda.current_device())}",
                flush=True,
            )

            dist.init_process_group(
                backend=config.backend,
                device_id=torch.device(f"cuda:{local_rank}"),
            )

            dist.barrier()
            print(
                f"[POST-INIT] rank={dist.get_rank()}/{dist.get_world_size()} "
                f"host={socket.gethostname()} "
                f"current_device={torch.cuda.current_device()} "
                f"visible={os.environ.get('ROCR_VISIBLE_DEVICES')} "
                f"device_name={torch.cuda.get_device_name(torch.cuda.current_device())}",
                flush=True,
            )
            dist.barrier()


        else:
            from mpi4py import MPI
            mpi_rank = MPI.COMM_WORLD.Get_rank()
            mpi_world_size = MPI.COMM_WORLD.Get_size()
            os.environ["RANK"] = str(mpi_rank)
            os.environ["WORLD_SIZE"] = str(mpi_world_size)
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
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

    # ── FIX 1 / FIX 2 knobs (env-var fallback, no config.py change needed) ──
    wd_on_bn_bias = str(getattr(
        config, "wd_on_bn_bias", os.getenv("WD_ON_BN_BIAS", "false")
    )).lower() in ("1", "true", "yes")

    nesterov = str(getattr(
        config, "nesterov", os.getenv("NESTEROV", "true")
    )).lower() in ("1", "true", "yes")

    # ── Setup logging and CSV (rank 0 only) ──────────────────────────────
    csv_file   = None
    csv_writer = None

    algo_str = config.comm_algorithm or "builtin"
    if config.comm_algorithm and "zfp" in config.comm_algorithm:
        algo_str = f"{algo_str}_rate{config.zfp_rate:g}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # FIX 3: per-rank iteration log (every rank writes — needed for straggler analysis)
    iter_log_path = None
    if ENABLE_ITER_LOG:
        os.makedirs("results/iters", exist_ok=True)
        iter_log_path = (
            f"results/iters/{config.model_name}_{config.backend}_{algo_str}_"
            f"ws{world_size}_bs{config.batch_size}_rank{rank}.jsonl"
        )
        # truncate any stale file from a previous run
        open(iter_log_path, 'w').close()

    if rank == 0:
        log_filename = (
            f"{config.model_name}_{config.num_epochs}_"
            f"{config.backend}_{algo_str}_"
            f"ws{world_size}_bs{config.batch_size}_"
            f"gb{config.batch_size * world_size}.log"
        )
        log_file = open(log_filename, 'w')
        sys.stdout = Tee(sys.__stdout__, log_file)

        os.makedirs("results", exist_ok=True)
        csv_path   = (f"results/{config.model_name}_{config.backend}_"
                      f"{algo_str}_"
                      f"ws{world_size}_bs{config.batch_size}_"
                      f"gb{config.batch_size * world_size}_"
                      f"{timestamp}.csv")
        csv_file   = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch', 'lr', 'train_loss', 'train_acc',
            'val_loss', 'val_acc', 'epoch_time_s',
            # FIX 3: new timing columns
            'epoch_train_time_s', 't_iter_median_ms', 't_fwd_median_ms',
            't_bwd_median_ms', 't_opt_median_ms', 't_data_median_ms',
            't_barrier_median_ms', 'num_iters',
            'model', 'backend', 'algorithm', 'zfp_rate', 'world_size',
            'batch_size', 'global_batch_size', 'num_epochs',
            # FIX 1: record the ablation setting in the results
            'wd_on_bn_bias', 'nesterov',
        ])

    # Seed for reproducible model init (DDP broadcasts rank 0's params anyway)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Standard paper setting: let cuDNN benchmark and pick the fastest kernels.
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        # If the GPU name contains 'amd' or 'gfx' (common for AMD architectures)
        if 'amd' in device_name or 'gfx' in device_name:
            torch.backends.cudnn.benchmark = False
        else:
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

    # ── Pretrained weights cache location (explicit name) ───────────────────
    # TorchVision pretrained weights are cached under: $TORCH_HOME/hub/checkpoints/
    pretrained_cache_root = os.environ.get(
        "PRETRAINED_WEIGHTS_CACHE",
        os.path.join(config.data_dir, "pretrained_weights_cache"),
    )
    os.makedirs(pretrained_cache_root, exist_ok=True)
    os.environ["TORCH_HOME"] = pretrained_cache_root
    if rank == 0:
        print(f"[Cache] Pretrained weights cache (TORCH_HOME) = {pretrained_cache_root}")

    # Ensure only rank 0 can download pretrained weights; others wait.
    if config.pretrained and rank != 0:
        dist.barrier()

    model = create_model(
        config.model_name,
        config.num_classes,
        cifar_stem=config.cifar_stem,
        pretrained=config.pretrained,
        init_cifar_stem_from_pretrained_center=config.init_cifar_stem_from_pretrained_center,
    ).to(device)

    if config.pretrained and rank == 0:
        dist.barrier()

    model = DDP(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
        static_graph=False,
        # bucket_cap_mb=30,                 # try 1, 5, 10, 25, 50
        gradient_as_bucket_view=False,   # try False first
    )

    if rank == 0:
        print(f"[Setup] Model on {device}, wrapped in DDP (static_graph=True)")

    # ── Communication hook ───────────────────────────────────────────────
    # get_comm_hook returns (hook, state) or None.
    # The ring hook requires a RingAllreduceState as its state so that
    # persistent GPU buffers can be reused across backward passes — this
    # is what fixes the Cray MPICH GPU-direct RDMA stale-registration bug.
    result = get_comm_hook(
        config.backend,
        config.comm_algorithm,
        zfp_rate=config.zfp_rate,
    )
    if result is not None:
        comm_hook, comm_state = result
        model.register_comm_hook(state=comm_state, hook=comm_hook)
        if rank == 0:
            print(f"[Setup] Hook registered: {config.backend}/"
                  f"{config.comm_algorithm}")
    else:
        if rank == 0:
            print(f"[Setup] No hook — using DDP built-in "
                  f"{config.backend.upper()} allreduce")

    # ── Optimizer ────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # FIX 1: BN gamma/beta and biases excluded from weight decay unless
    # wd_on_bn_bias=True (which reproduces the original single-group behaviour).
    param_groups = build_param_groups(model, config.weight_decay, wd_on_bn_bias)
    optimizer_name = str(getattr(
        config, "optimizer", os.getenv("OPTIMIZER", "sgd")
    )).lower()

    param_groups = build_param_groups(model, config.weight_decay, wd_on_bn_bias)

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,   # per-group value overrides this
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            param_groups,
            lr=config.learning_rate,
            momentum=config.momentum,
            nesterov=(nesterov and config.momentum > 0),
            weight_decay=config.weight_decay,   # per-group value overrides this
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    if rank == 0:
        print(f"[Optim] {config.optimizer.upper()}: lr={config.learning_rate} "
              f"momentum={config.momentum} "
              f"nesterov={nesterov} wd={config.weight_decay} "
              f"wd_on_bn_bias={wd_on_bn_bias}")
        for gi, g in enumerate(optimizer.param_groups):
            n = sum(p.numel() for p in g['params'])
            print(f"[Optim]   group {gi}: {n:,} params, wd={g['weight_decay']}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, train_sampler = get_data_loaders(
        config, rank, world_size
    )
    if rank == 0:
        print(f"[Data] Train batches: {len(train_loader)}, "
              f"Val batches: {len(val_loader)}\n")

    # ── FIX 2: per-ITERATION scheduler ───────────────────────────────────
    # Built after the loader so steps_per_epoch is known. drop_last=True makes
    # len(train_loader) identical on every rank, so the schedule stays in sync.
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * config.num_epochs
    warmup_steps    = steps_per_epoch * config.warmup_epochs

    if config.scheduler == "constant":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    else:
        def lr_lambda(step: int) -> float:
            # step = global iteration index (0, 1, 2, ...), NOT epoch index.
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(max(progress, 0.0), 1.0)   # land exactly on 0 at the end
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if rank == 0:
        print(f"[Sched] {config.scheduler}: steps/epoch={steps_per_epoch}, "
              f"total_steps={total_steps}, warmup_steps={warmup_steps} "
              f"({config.warmup_epochs} epochs)\n")

    # ── Training loop ────────────────────────────────────────────────────
    best_val_acc = 0.0
    start_time = time.time()
    epoch_times = []        # per-epoch wall time (unchanged: train + val)
    epoch_train_times = []  # FIX 3: train-only epoch time (excludes validation)

    for e in range(config.num_epochs):  # e = 0,1,2,...,num_epochs-1
        epoch = e + 1  # for human-readable prints/logs

        train_sampler.set_epoch(e)  # DistributedSampler expects an int; using e is standard

        # LR at the START of the epoch (it now changes every iteration).
        epoch_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            print(f"\n[Epoch {epoch}] Training... (lr at epoch start={epoch_lr:.6f})")

        epoch_start = time.time()
        _nvtx_range_push(f"epoch {epoch}")
        try:
            train_loss, train_acc, iter_stats = train_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                epoch,  # keep passing the human epoch to your prints/NVTX markers
                rank, device,
                verify_last_batch=(epoch == config.num_epochs),
                grad_clip=config.grad_clip,
                iter_log_path=iter_log_path,
            )
        finally:
            _nvtx_range_pop()

        # ── Validation: rank 0 only, on the FULL unsharded test set ──────────
        # val_loader has no DistributedSampler, so every rank would compute an
        # identical number (DDP keeps weights in sync). Running it on rank 0 only
        # avoids P-way redundant work.
        #
        # This is safe ONLY because validate() contains no collective — ranks
        # 1..P-1 can skip it without deadlocking. If you ever add a dist.* call
        # inside validate(), this WILL hang.
        #
        # The barrier re-aligns ranks so a late rank 0 doesn't show up as a
        # phantom straggler in iteration 0 of the next epoch (which would
        # silently inflate t_bwd on every other rank).
        if rank == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = float('nan'), float('nan')
        dist.barrier()

        # NOTE: scheduler.step() is NOT called here any more — it is stepped
        # once per iteration inside train_epoch(). Calling it here as well
        # would advance the schedule steps_per_epoch+1 times per epoch.

        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        epoch_train_times.append(iter_stats["t_epoch_train_s"])

        if rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, "
                  f"Acc: {train_acc:.2f}%  |  "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%  |  "
                  f"Time: {epoch_elapsed:.1f}s")
            print(f"[Epoch {epoch}] t_iter median: "
                  f"{iter_stats['t_iter_median_ms']:.2f} ms  "
                  f"(fwd {iter_stats['t_fwd_median_ms']:.2f} | "
                  f"bwd+comm {iter_stats['t_bwd_median_ms']:.2f} | "
                  f"opt {iter_stats['t_opt_median_ms']:.2f} | "
                  f"data {iter_stats['t_data_median_ms']:.2f})  |  "
                  f"train-only epoch: {iter_stats['t_epoch_train_s']:.1f}s")

            csv_writer.writerow([
                epoch, f"{epoch_lr:.8f}",
                f"{train_loss:.6f}", f"{train_acc:.4f}",
                f"{val_loss:.6f}",  f"{val_acc:.4f}",
                f"{epoch_elapsed:.2f}",
                f"{iter_stats['t_epoch_train_s']:.3f}",
                f"{iter_stats['t_iter_median_ms']:.4f}",
                f"{iter_stats['t_fwd_median_ms']:.4f}",
                f"{iter_stats['t_bwd_median_ms']:.4f}",
                f"{iter_stats['t_opt_median_ms']:.4f}",
                f"{iter_stats['t_data_median_ms']:.4f}",
                f"{iter_stats['t_barrier_median_ms']:.4f}",
                iter_stats['num_iters'],
                config.model_name, config.backend, algo_str,
                f"{config.zfp_rate:g}",
                world_size, config.batch_size,
                config.batch_size * world_size,
                config.num_epochs,
                wd_on_bn_bias, nesterov,
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

            print(f"[Epoch {epoch}] Summarizing hook timing...", flush=True)
            summarize_hook_timing(epoch)
            print(f"[Epoch {epoch}] Hook timing summary done", flush=True)

        # ── Per-epoch gradient stats (rank 0 only, first 10 params) ─────
        # print_gradient_stats(model, epoch, rank, max_params=10)

    elapsed = time.time() - start_time

    # ── Final summary ────────────────────────────────────────────────────
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"Done in {elapsed:.1f}s | Best val acc: {best_val_acc:.2f}%")
        if epoch_times:
            print(f"Epoch times (train+val): mean={statistics.mean(epoch_times):.1f}s  "
                  f"min={min(epoch_times):.1f}s  "
                  f"max={max(epoch_times):.1f}s")
        # FIX 3: scaling-relevant numbers, epoch 0 discarded (MIOpen autotune,
        # allocator warmup, first-touch page faults).
        if len(epoch_train_times) > 1:
            warm = epoch_train_times[1:]
            print(f"Train-only epoch time (epochs 2..N): "
                  f"median={statistics.median(warm):.2f}s  "
                  f"min={min(warm):.2f}s  max={max(warm):.2f}s")
            print(f"  -> use this for STRONG scaling (fixed global batch)")
            print(f"  -> use t_iter_median_ms from the CSV for WEAK scaling "
                  f"(fixed local batch)")
        print("=" * 80)

    if csv_file is not None:
        csv_file.close()

    if config.backend == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()
    dist.destroy_process_group()