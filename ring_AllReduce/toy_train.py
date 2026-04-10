"""
Toy DDP training script for testing ring allreduce hooks.

Rank r produces gradient (r+1) * ones(dim).
After SUM allreduce and division by world_size:
    param.grad = (world_size+1)/2 * ones(dim)

With W=2: expected = 1.5
With W=4: expected = 2.5

Usage:
    mpirun -np 2 python toy_train.py --hook dist_ring --bench-iters 100
    mpirun -np 2 python toy_train.py --hook cpp_ring  --bench-iters 100
    mpirun -np 2 python toy_train.py --hook none      --bench-iters 100
"""

import os
import sys
import math
import argparse
import threading
import statistics
import time as time_module
import functools

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Toy DDP ring allreduce test")
    parser.add_argument(
        "--hook",
        choices=["none", "dist_ring", "cpp_ring"],
        default="none",
    )
    parser.add_argument("--grad-dim", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--bench-iters", type=int, default=0,
        help="Standalone allreduce benchmark iterations (0 = skip)",
    )
    parser.add_argument(
        "--bench-size", type=int, default=1_000_000,
        help="Elements in benchmark tensor (default 1M float32 = 3.8 MB)",
    )
    parser.add_argument(
        "--train-iters", type=int, default=20,
        help="Iterations for the total training timer (default 20)",
    )
    return parser.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────

class ToyModel(nn.Module):
    """Single linear layer y = Wx, weights initialized to zero."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


def make_toy_batch(rank, dim, device):
    """
    Returns (x, y) such that dLoss/dW = (rank+1) * ones(dim).

    With W=0: output=0, loss = MSELoss(0, target) = target^2
    dLoss/dW = 2*(0-target)*ones(dim) = (rank+1)*ones(dim)
    => target = -(rank+1)/2
    """
    x = torch.ones(1, dim, device=device)
    y = torch.tensor([[-(rank + 1) / 2.0]], dtype=torch.float32, device=device)
    return x, y


# ── Hook timing wrapper ───────────────────────────────────────────────────────

class TimedHook:
    """
    Wraps a DDP communication hook to measure the time of the FIRST
    allreduce call that fires during loss.backward().

    Uses torch.cuda.Event for GPU-accurate timing.

    For SYNC hooks:
        start recorded before hook, end recorded after hook returns.
        This correctly measures the full allreduce time because the hook
        blocks until communication is done.

    For ASYNC hooks:
        start recorded before hook, end recorded inside the Future callback
        which fires only after the background thread calls fut.set_result().
        This correctly measures the full communication time including the
        time the background thread spent in MPI/dist calls.

        We attach a callback to the returned Future via fut.add_done_callback()
        which fires when fut.set_result() is called on the background thread.
    """
    def __init__(self, hook_fn):
        self.hook_fn = hook_fn
        self.first_call_ms = None
        self._called = False
        self._start_event = None
        # copy __name__, __qualname__, __annotations__, __doc__, __module__
        # DDP checks several of these in _check_comm_hook
        functools.update_wrapper(self, hook_fn)

    def __call__(self, state, bucket):
        if not self._called:
            self._called = True

            import time as _time

            if True:
                # use wall clock for both sync and async hooks to avoid
                # CUDA Event cross-thread "invalid resource handle" errors.
                # cuda.Event cannot be used across threads — if the end
                # event is recorded on the background thread while the
                # start event was created on the main thread, CUDA crashes.
                # Wall clock (perf_counter) is thread-safe and gives
                # millisecond accuracy sufficient for comm timing.
                start_wall = _time.perf_counter()
                fut = self.hook_fn(state, bucket)

                if fut.done():
                    # SYNC hook — measure immediately
                    self.first_call_ms = (_time.perf_counter() - start_wall) * 1000.0
                else:
                    # ASYNC hook — attach callback that fires on background
                    # thread after fut.set_result() is called
                    def _on_done(f):
                        self.first_call_ms = (
                            _time.perf_counter() - start_wall
                        ) * 1000.0
                    fut.add_done_callback(_on_done)

            return fut
        return self.hook_fn(state, bucket)


# ── Gradient verification ─────────────────────────────────────────────────────

def verify_gradients(model, rank, world_size, dim):
    """Check that every rank has the expected averaged gradient."""
    expected = (world_size + 1) / 2.0
    expected_sum = world_size * (world_size + 1) / 2.0

    grad = model.module.linear.weight.grad
    if grad is None:
        print(f"[Rank {rank}] ERROR: gradient is None", flush=True)
        return

    vals = grad.detach().cpu().flatten().tolist()
    dist.barrier()
    for r in range(world_size):
        if rank == r:
            print(
                f"\n[Rank {rank}]"
                f"\n  actual   grad[:8] = {[f'{v:.6f}' for v in vals[:8]]}"
                f"\n  expected (after hook div) = {expected:.6f}"
                f"\n  expected sum (before div) = {expected_sum:.6f}",
                flush=True,
            )
            ok = all(abs(v - expected) < 1e-4 for v in vals)
            print(f"  Status: {'PASS v' if ok else 'FAIL x'}", flush=True)
        dist.barrier()


# ── Standalone benchmark ──────────────────────────────────────────────────────

def benchmark_allreduce(hook_fn, hook_name, tensor_size, n_iters,
                        n_warmup, rank, device):
    """
    Measure raw allreduce time on a random tensor, completely outside DDP.
    Uses torch.cuda.Event for GPU-accurate timing.
    Reports max across ranks (slowest rank = bottleneck).
    """
    tensor = torch.randn(tensor_size, dtype=torch.float32, device=device)
    size_mb = tensor_size * 4 / 1024 / 1024

    dist.barrier()
    for _ in range(n_warmup):
        hook_fn(tensor.clone())
        torch.cuda.synchronize(device)

    times_ms = []
    dist.barrier()
    for _ in range(n_iters):
        t = tensor.clone()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        hook_fn(t)
        e.record()
        torch.cuda.synchronize(device)
        times_ms.append(s.elapsed_time(e))

    stats = {
        "mean": statistics.mean(times_ms),
        "median": statistics.median(times_ms),
        "min": min(times_ms),
    }

    if rank == 0:
        print(f"\n  [{hook_name}]  tensor: {size_mb:.1f} MB  "
              f"({tensor_size:,} float32)  |  {n_iters} iters  "
              f"({n_warmup} warmup)")

    for stat_name, val in stats.items():
        t = torch.tensor(val, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"    {stat_name:<6} : {t.item():.3f} ms")

    med_t = torch.tensor(stats["median"], device=device)
    dist.all_reduce(med_t, op=dist.ReduceOp.MAX)
    if rank == 0:
        world_size = dist.get_world_size()
        factor = 2.0 * (world_size - 1) / world_size
        bw = (factor * size_mb / 1024) / (med_t.item() / 1000)
        print(f"    algbw  : {bw:.2f} GB/s  (2*(W-1)/W * size / median)")


# ── Training loop timer ───────────────────────────────────────────────────────

def timed_training_loop(model, n_iters, rank, device, dim):
    """
    Run n_iters of forward+backward+step and return total wall-clock time.

    Uses time.perf_counter (CPU wall clock) not cuda.Event because we want
    to capture TOTAL time including CPU waiting for async hooks to resolve.
    A barrier before/after ensures all ranks start and stop together.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # warmup
    for _ in range(3):
        optimizer.zero_grad()
        x, y = make_toy_batch(rank, dim, device)
        criterion(model(x), y).backward()
        optimizer.step()

    torch.cuda.synchronize(device)
    dist.barrier()
    t_start = time_module.perf_counter()

    for _ in range(n_iters):
        optimizer.zero_grad()
        x, y = make_toy_batch(rank, dim, device)
        criterion(model(x), y).backward()
        optimizer.step()

    torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time_module.perf_counter() - t_start

    t = torch.tensor(elapsed, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    dist.init_process_group(backend="mpi")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"Toy Ring Allreduce Test")
        print(f"Ranks: {world_size} | Hook: {args.hook} | Grad dim: {args.grad_dim}")
        print("=" * 60)
        print(f"\nEach rank r produces gradient (r+1) per element.")
        print(f"Expected sum  : {world_size*(world_size+1)/2:.1f}")
        print(f"Expected /W   : {(world_size+1)/2:.4f}  (hook divides by {world_size})\n")

    # ── Model + DDP ───────────────────────────────────────────────────────
    torch.manual_seed(0)
    model = ToyModel(args.grad_dim).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    # ── Register hook ─────────────────────────────────────────────────────
    timed_hook = None

    if args.hook == "dist_ring":
        from hooks.ring_allreduce_dist import ring_allreduce_dist_hook_async
        timed_hook = TimedHook(ring_allreduce_dist_hook_async)
        ddp_model.register_comm_hook(state=None, hook=timed_hook)
        if rank == 0:
            print("[Hook] dist primitives ring allreduce (async)")

    elif args.hook == "cpp_ring":
        try:
            from hooks.ring_allreduce_cpp_hook import (
                ring_allreduce_cpp_hook_async,
                check_mpi_thread_safety,
            )
            if rank == 0:
                check_mpi_thread_safety()
            timed_hook = TimedHook(ring_allreduce_cpp_hook_async)
            ddp_model.register_comm_hook(state=None, hook=timed_hook)
            if rank == 0:
                print("[Hook] C++/CUDA MPI ring allreduce (async)")
        except ImportError as e:
            if rank == 0:
                print(f"[Hook] ERROR: {e}")
            dist.destroy_process_group()
            sys.exit(1)
    else:
        if rank == 0:
            print("[Hook] None - DDP built-in MPI allreduce")

    # ── One forward-backward to verify + measure first allreduce ─────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.0)

    if rank == 0:
        print("\n--- Forward-backward pass ---")

    optimizer.zero_grad()
    x, y = make_toy_batch(rank, args.grad_dim, device)
    loss = criterion(ddp_model(x), y)
    if rank == 0:
        print(f"Loss: {loss.item():.6f}")
    loss.backward()

    if timed_hook is not None and rank == 0:
        import time as _t
        for _ in range(200):
            if timed_hook.first_call_ms is not None:
                break
            _t.sleep(0.01)
        ms = timed_hook.first_call_ms
        if ms is not None:
            print(f"\n[Timing] First allreduce inside DDP backward: {ms:.3f} ms")
            print(f"         Method: wall clock (perf_counter), thread-safe")
            print(f"         Sync hooks:  hook entry to hook return")
            print(f"         Async hooks: hook entry to fut.set_result() on bg thread")
        else:
            print(f"\n[Timing] First allreduce: still pending after 2s timeout")

    if rank == 0:
        print("\n--- Gradient verification ---")
    verify_gradients(ddp_model, rank, world_size, args.grad_dim)

    # ── Total training timer ──────────────────────────────────────────────
    if rank == 0:
        print(f"\n--- Training loop timer ({args.train_iters} iters, 3 warmup) ---")

    total_time = timed_training_loop(
        ddp_model, args.train_iters, rank, device, args.grad_dim
    )

    if rank == 0:
        print(f"  Total wall time : {total_time:.4f} s")
        print(f"  Per iteration   : {total_time/args.train_iters*1000:.3f} ms")
        print(f"  Throughput      : {args.train_iters/total_time:.1f} iters/s")

    # ── Standalone benchmark ──────────────────────────────────────────────
    if args.bench_iters > 0:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"Standalone allreduce benchmark ({args.bench_iters} iters, 10 warmup)")
            print(f"{'=' * 60}")

        if args.hook == "dist_ring":
            from hooks.ring_allreduce_dist import _ring_allreduce_sum
            benchmark_allreduce(
                _ring_allreduce_sum, "dist_ring",
                args.bench_size, args.bench_iters, 10, rank, device,
            )
        elif args.hook == "cpp_ring":
            try:
                import ring_allreduce_cuda_ext as _ext
                benchmark_allreduce(
                    _ext.ring_allreduce, "cpp_ring (CUDA-aware MPI)",
                    args.bench_size, args.bench_iters, 10, rank, device,
                )
            except ImportError:
                if rank == 0:
                    print("[Bench] C++ extension not available.")
        else:
            def _builtin(t):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            benchmark_allreduce(
                _builtin, f"DDP built-in ({dist.get_backend()})",
                args.bench_size, args.bench_iters, 10, rank, device,
            )

        if rank == 0:
            print()

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("\nDone.\n")


if __name__ == "__main__":
    main()