"""
Communication Strategy & AllReduce Hooks

═══════════════════════════════════════════════════════════════════════════════
BUG HISTORY
═══════════════════════════════════════════════════════════════════════════════

BUG 1 (FIXED): Cray MPICH GPU-direct RDMA stale tag cache
──────────────────────────────────────────────────────────
recv_bufs were receiving stale data because Cray MPICH caches RDMA buffer
registrations per tag. Reusing tag=N on batch 2 with the same GPU address
caused the runtime to replay the cached DMA rather than posting a fresh one.

FIX: global monotonically increasing _tag_counter. Every P2POp gets a unique
tag across the entire training run. Cray MPICH always sees a fresh tag and
always creates a new RDMA registration.

BUG 2 (FIXED): B0 hook silenced after the first backward pass
─────────────────────────────────────────────────────────────
SYMPTOM: [HOOK | R0|B0|C1] appeared exactly once. From batch 2 onward, B0's
hook was never called. B1–B5 fired correctly every batch.

ROOT CAUSE — two compounding issues:

  1. DDP bucket rebuild after first backward.
     PyTorch DDP profiles gradient arrival order during the first backward and
     rebuilds bucket assignments to match real timing. This changes bucket
     sizes (numel). RingAllreduceState keys per-bucket state by numel, so after
     rebuild the old B0 state is orphaned and a new one is created. But more
     fundamentally, rebuild resets DDP's internal hook dispatch for B0 in a way
     that interacts badly with a pre-resolved or near-instantly-resolved future.

  2. Pre-resolved / near-instantly-resolved future returned by the hook.
     DDP's C++ Reducer queues finalize_backward() as an autograd engine callback
     that runs after the full backward pass. finalize_backward() calls
     future_work->wait() for each bucket in order (B0 first). If B0's future is
     already resolved (scalar dummy all_reduce completes in microseconds),
     finalize_backward processes B0 synchronously and immediately, potentially
     before DDP's internal state machine has finished re-registering B0's
     autograd hooks for the next iteration post-rebuild.

FIX 1: static_graph=True in DDP constructor (in ddp_training.py).
     Tells DDP the autograd graph never changes. Buckets are built once and
     never rebuilt. B0's hook registration is permanent.

FIX 2 (superseded): return a genuine async future tied to a full-size tensor.
     This was needed when the hook was registered before the first backward,
     because bucket rebuild caused a timing window where the last post-rebuild
     bucket was silently dropped. The token all_reduce was a synchronous
     barrier surrogate to paper over the mismatch.

FIX 2 (current): dummy backward in ddp_training.py before register_comm_hook.
     Triggers the DDP bucket rebuild explicitly before the hook is registered.
     The hook sees the final stable bucket layout from the start. The token
     all_reduce is no longer needed and has been removed — it was itself
     introducing spurious synchronous collectives that corrupted Cray MPICH
     GPU-direct RDMA ordering on every backward pass.

═══════════════════════════════════════════════════════════════════════════════
DDP HOOK CONTRACT
═══════════════════════════════════════════════════════════════════════════════

When NO hook is registered, DDP SUM-allreduces then divides by world_size.
When a hook IS registered, DDP does NOT divide — the hook must produce the
averaged gradient. Our hooks divide by world_size after the SUM.

The hook MUST return a future that:
  - Comes from dist.all_reduce(..., async_op=True).get_future()  [real Work]
  - Resolves to bucket.buffer() — the original buffer tensor, not a clone

═══════════════════════════════════════════════════════════════════════════════
API
═══════════════════════════════════════════════════════════════════════════════

    result = get_comm_hook(backend, algorithm)
    if result is not None:
        hook, state = result
        model.register_comm_hook(state=state, hook=hook)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import threading

import torch
import torch.distributed as dist

# ── Diagnostic knob ──────────────────────────────────────────────────────────
# ── Verbosity knobs ──────────────────────────────────────────────────────────
# FULL_LOG_CALLS : first N calls get the full per-chunk trace + verify.
# LIGHT_LOG_CALLS: calls N+1..M get a cheap one-line summary (norm + sample)
#                  so you can watch whether the gradient is healthy over time
#                  without the massive per-step trace that slows training.
#                  Set to 0 to silence everything after the full-log window.
FULL_LOG_CALLS  = 0    # heavy trace with per-step P2P + verify (as before)
LIGHT_LOG_CALLS = 0    # lightweight one-liner after that.  0 = off.
# ─────────────────────────────────────────────────────────────────────────────

# ── Global monotonic tag counter ─────────────────────────────────────────────
# Each P2POp gets a unique tag. Never reused → no RDMA cache reuse.
# One counter per process (MPI rank = OS process, no sharing needed).
_tag_counter: int = 0

def _next_tag() -> int:
    global _tag_counter
    t = _tag_counter
    _tag_counter += 1
    return t
# ─────────────────────────────────────────────────────────────────────────────


# ── Tensor snippet formatter ──────────────────────────────────────────────────
def _fmt(t: torch.Tensor) -> str:
    """Print first 4 elements plus how many more exist."""
    n      = t.numel()
    sample = t.flatten()[:4].tolist()
    extra  = n - len(sample)
    return f"{sample} ... ({extra} more elements)" if extra > 0 else f"{sample}"
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT HOOKS
# ═══════════════════════════════════════════════════════════════════════════════

def _default_allreduce_hook(
    state: Optional[object],
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Baseline — dist.all_reduce with pre-division."""
    tensor = bucket.buffer()
    tensor.div_(dist.get_world_size())
    fut = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True).get_future()
    return fut.then(lambda f: f.value()[0])

nccl_default_allreduce_hook = _default_allreduce_hook
mpi_default_allreduce_hook  = _default_allreduce_hook


# ═══════════════════════════════════════════════════════════════════════════════
# RING ALLREDUCE — state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RingBucketState:
    """Persistent GPU buffers for one DDP bucket."""
    flat:       Optional[torch.Tensor] = None
    send_bufs:  List[torch.Tensor]     = field(default_factory=list)
    recv_bufs:  List[torch.Tensor]     = field(default_factory=list)
    cnts:       List[int]              = field(default_factory=list)
    displs:     List[int]              = field(default_factory=list)
    bucket_id:  int                    = 0
    call_count: int                    = 0
    ready:      bool                   = False
    token:      Optional[torch.Tensor] = None  # unused — token allreduce removed

    def initialize(self, tensor: torch.Tensor, world_size: int,
                   bucket_id: int) -> None:
        numel      = tensor.numel()
        base_chunk = math.ceil(numel / world_size)

        total = 0
        self.cnts = []
        for i in range(world_size):
            c = min(base_chunk, numel - total)
            self.cnts.append(max(c, 0))
            total += self.cnts[-1]

        self.displs = [0]
        for i in range(1, world_size):
            self.displs.append(self.displs[-1] + self.cnts[i - 1])

        self.flat      = torch.empty(numel, dtype=tensor.dtype, device=tensor.device)
        self.send_bufs = [
            torch.empty(self.cnts[i], dtype=tensor.dtype, device=tensor.device)
            for i in range(world_size)
        ]
        self.recv_bufs = [
            torch.empty(self.cnts[i], dtype=tensor.dtype, device=tensor.device)
            for i in range(world_size)
        ]
        self.bucket_id = bucket_id
        self.ready     = True


@dataclass
class RingAllreduceState:
    """
    Per-bucket ring state, keyed by DDP bucket index.

    Keying by bucket.index() (not numel) is important: after DDP's first-backward
    bucket rebuild, bucket sizes may change. Keying by index means each logical
    bucket position always maps to the same state slot, and initialize() is called
    again if the tensor shape changed (detected by numel mismatch vs flat buffer).
    """
    buckets: Dict[int, RingBucketState] = field(default_factory=dict)

    def get_or_init(self, bucket_index: int,
                    tensor: torch.Tensor, world_size: int) -> RingBucketState:
        bs = self.buckets.get(bucket_index)
        if bs is None or bs.flat is None or bs.flat.numel() != tensor.numel():
            # First call for this bucket, or numel changed after DDP rebuild —
            # (re)initialize the persistent GPU buffers.
            if bs is not None and bs.flat is not None:
                pass  # numel changed after DDP bucket rebuild — reinitializing
                # old_numel = bs.flat.numel()
                # print(f"[RING STATE] bucket_index={bucket_index} reinitializing: "
                #       f"numel changed {old_numel} → {tensor.numel()} "
                #       f"(DDP bucket rebuild detected)", flush=True)
            bs = RingBucketState()
            bs.initialize(tensor, world_size, bucket_index)
            self.buckets[bucket_index] = bs
        return bs


# ═══════════════════════════════════════════════════════════════════════════════
# RING ALLREDUCE — algorithm
# ═══════════════════════════════════════════════════════════════════════════════

def _cuda_sync(t: torch.Tensor) -> None:
    """Drain the current CUDA stream so every preceding kernel has finished.

    GPU-direct RDMA (Cray MPICH with MPICH_GPU_SUPPORT_ENABLED=1) reads GPU
    memory via PCIe DMA, which bypasses CUDA stream ordering.  A CUDA copy_()
    into a send buffer is an async kernel; if MPI initiates the DMA before
    that kernel lands, it sends stale data.  This one-liner guarantees the
    buffer is ready before MPI touches it.

    Why C1/C2 appeared correct without this: the full-log code path calls
    .tolist() (device→host copy = implicit stream sync) right before every
    batch_isend_irecv.  C3+ skipped logging → no sync → stale reads →
    MISMATCH.
    """
    if t.is_cuda:
        torch.cuda.current_stream(device=t.device).synchronize()

def _ring_allreduce_sum(tensor: torch.Tensor, bstate: RingBucketState,
                        log: bool, call_n: int = 0) -> None:
    """
    In-place ring allreduce with monotonically increasing tags.

    Every P2POp gets a fresh unique tag via _next_tag(). This guarantees
    Cray MPICH creates a new RDMA registration for every operation and
    never reuses a stale cached mapping.

    All ranks call _next_tag() the same number of times in the same order
    within each hook invocation, so send/recv tags always match.
    """
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size == 1:
        return

    pfx = f"[R{rank}|B{bstate.bucket_id}|C{call_n}]"

    bstate.flat.copy_(tensor.flatten())

    chunks = [
        bstate.flat[bstate.displs[i]: bstate.displs[i] + bstate.cnts[i]]
        for i in range(world_size)
    ]

    src = (rank - 1 + world_size) % world_size
    dst = (rank + 1) % world_size

    if log:
        print(f"{pfx} flat after copy from tensor: {_fmt(bstate.flat)}", flush=True)
        for i in range(world_size):
            owned = " <-- OWNED" if i == rank else ""
            print(f"{pfx}   chunk[{i}] displ={bstate.displs[i]} "
                  f"cnt={bstate.cnts[i]}{owned}: {_fmt(chunks[i])}", flush=True)

    # ── Phase 1: Reduce-scatter ───────────────────────────────────────────────
    for step in range(world_size - 1):
        send_idx = (rank - step - 1 + world_size) % world_size
        recv_idx = (rank - step - 2 + world_size) % world_size

        # Fresh unique tag — Cray MPICH will never reuse a cached registration
        tag = _next_tag()

        bstate.send_bufs[send_idx].copy_(chunks[send_idx])
        _cuda_sync(bstate.send_bufs[send_idx])

        if log:
            print(f"{pfx} P1 step={step} tag={tag} "
                  f"SENDING chunk[{send_idx}]→rank{dst}: {_fmt(bstate.send_bufs[send_idx])} | "
                  f"EXPECTING recv chunk[{recv_idx}]←rank{src}", flush=True)

        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.irecv, bstate.recv_bufs[recv_idx], src, tag=tag),
            dist.P2POp(dist.isend, bstate.send_bufs[send_idx], dst, tag=tag),
        ])
        for req in reqs:
            req.wait()

        before = chunks[recv_idx][:4].tolist() if log else None
        chunks[recv_idx].add_(bstate.recv_bufs[recv_idx])

        if log:
            print(f"{pfx} P1 step={step} RECVD chunk[{recv_idx}]←rank{src}: "
                  f"{_fmt(bstate.recv_bufs[recv_idx])}", flush=True)
            print(f"{pfx}   chunk[{recv_idx}] before_add={before} "
                  f"after_add={chunks[recv_idx][:4].tolist()} "
                  f"... ({bstate.cnts[recv_idx] - 4} more elements)", flush=True)

    if log:
        print(f"{pfx} === AFTER Phase 1 (reduce-scatter) ===", flush=True)
        for i in range(world_size):
            owned = " <-- fully reduced" if i == rank else ""
            print(f"{pfx}   chunk[{i}]{owned}: {_fmt(chunks[i])}", flush=True)

    # ── Phase 2: Allgather ────────────────────────────────────────────────────
    for step in range(world_size - 1):
        send_idx = (rank - step     + world_size) % world_size
        recv_idx = (rank - step - 1 + world_size) % world_size

        tag = _next_tag()

        bstate.send_bufs[send_idx].copy_(chunks[send_idx])
        _cuda_sync(bstate.send_bufs[send_idx])

        if log:
            print(f"{pfx} P2 step={step} tag={tag} "
                  f"SENDING chunk[{send_idx}]→rank{dst}: {_fmt(bstate.send_bufs[send_idx])} | "
                  f"EXPECTING recv chunk[{recv_idx}]←rank{src}", flush=True)

        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.irecv, bstate.recv_bufs[recv_idx], src, tag=tag),
            dist.P2POp(dist.isend, bstate.send_bufs[send_idx], dst, tag=tag),
        ])
        for req in reqs:
            req.wait()

        chunks[recv_idx].copy_(bstate.recv_bufs[recv_idx])

        if log:
            print(f"{pfx} P2 step={step} RECVD chunk[{recv_idx}]←rank{src}: "
                  f"{_fmt(bstate.recv_bufs[recv_idx])}", flush=True)
            print(f"{pfx}   chunk[{recv_idx}] now={chunks[recv_idx][:4].tolist()} "
                  f"... ({bstate.cnts[recv_idx] - 4} more elements)", flush=True)

    if log:
        print(f"{pfx} === AFTER Phase 2 (allgather) ===", flush=True)
        for i in range(world_size):
            print(f"{pfx}   chunk[{i}]: {_fmt(chunks[i])}", flush=True)

    tensor.copy_(bstate.flat.view_as(tensor))
    tensor.div_(world_size)
    _cuda_sync(tensor)   # ensure div_ has landed before hook resolves the future


def _ring_allreduce_hook(
    state:  RingAllreduceState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    tensor       = bucket.buffer()
    world_size   = dist.get_world_size()
    rank         = dist.get_rank()
    numel        = tensor.numel()
    bucket_index = bucket.index()   # DDP's logical bucket index, stable across rebuild

    bstate = state.get_or_init(bucket_index, tensor, world_size)
    bstate.call_count += 1
    call_n = bstate.call_count

    do_full_log  = (FULL_LOG_CALLS  > 0 and call_n <= FULL_LOG_CALLS)
    do_light_log = (not do_full_log and LIGHT_LOG_CALLS > 0
                    and call_n <= LIGHT_LOG_CALLS)

    # # Unconditional — fires on every hook invocation regardless of log knobs.
    # # Use this to verify all buckets fire on every backward pass.
    # print(f"[HOOK | R{rank}|B{bstate.bucket_id}|C{call_n}|numel={numel}]",
    #       flush=True)

    # ── Snapshot for verification (full or light) ────────────────────────
    if do_full_log or do_light_log:
        pre = tensor.clone()

    if do_full_log:
        tid = threading.get_ident()
        print(f"\n[RING | R{rank}|B{bstate.bucket_id}|C{call_n}] "
              f"thread={tid} global_tag_before={_tag_counter} numel={numel}", flush=True)
        print(f"  [R{rank}] INPUT tensor: {_fmt(tensor)}", flush=True)
        for i in range(world_size):
            print(f"  [R{rank}] chunk[{i}]: displ={bstate.displs[i]} cnt={bstate.cnts[i]}", flush=True)
        print(f"  [R{rank}] flat.data_ptr={hex(bstate.flat.data_ptr())}", flush=True)
        for i in range(world_size):
            print(f"  [R{rank}] send_bufs[{i}].data_ptr={hex(bstate.send_bufs[i].data_ptr())} "
                  f"recv_bufs[{i}].data_ptr={hex(bstate.recv_bufs[i].data_ptr())}", flush=True)

    _ring_allreduce_sum(tensor, bstate, log=do_full_log, call_n=call_n)

    # ── Post-allreduce verification ──────────────────────────────────────
    # (disabled — re-enable by setting FULL_LOG_CALLS / LIGHT_LOG_CALLS > 0)
    # if do_full_log or do_light_log:
    #     ref = pre.clone()
    #     dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    #     ref.div_(world_size)
    #     max_err = (tensor - ref).abs().max().item()
    #     rel_err = max_err / (ref.norm().item() + 1e-12)
    #     match   = "✓ MATCH" if max_err < 1e-4 else "✗ MISMATCH"

    # if do_full_log:
    #     print(f"  [R{rank}|B{bstate.bucket_id}|C{call_n}] global_tag_after={_tag_counter}", flush=True)
    #     print(f"  [VERIFY | R{rank}|B{bstate.bucket_id}|C{call_n}] "
    #           f"max_abs_err={max_err:.3e}  rel={rel_err:.3e}  {match}", flush=True)
    #     if max_err >= 1e-4:
    #         err_pos = (tensor - ref).abs().argmax().item()
    #         print(f"  [R{rank}] ring     : {_fmt(tensor.detach())} worst_err_pos={err_pos}", flush=True)
    #         print(f"  [R{rank}] allreduce: {_fmt(ref.detach())}", flush=True)

    # elif do_light_log:
    #     in_norm  = pre.norm().item()
    #     out_norm = tensor.norm().item()
    #     is_nan   = "NaN!" if not torch.isfinite(tensor).all() else ""
    #     print(f"  [LIGHT | R{rank}|B{bstate.bucket_id}|C{call_n}] "
    #           f"in_norm={in_norm:.4e} out_norm={out_norm:.4e} "
    #           f"sample={_fmt(tensor)} {match} {is_nan}", flush=True)
    #     if max_err >= 1e-4:
    #         print(f"  [LIGHT | R{rank}|B{bstate.bucket_id}|C{call_n}] "
    #               f"*** MISMATCH *** max_abs_err={max_err:.3e}", flush=True)

    # Return a pre-resolved future containing tensor (= bucket.buffer()).
    # DDP's finalize_backward calls future_work->wait() then parseHookResult
    # which calls our .then() and gets tensor back. populate_bucket_views_out
    # then creates views into tensor — which already holds the ring result.
    #
    # Belt-and-suspenders: ensure ALL CUDA work on tensor is complete before
    # DDP is allowed to consume it.  _ring_allreduce_sum already syncs at
    # the end, but the verification path (clone, all_reduce, abs, max, norm)
    # also launches CUDA kernels on tensor-derived data.  This final sync
    # makes the future contract watertight regardless of code path.
    _cuda_sync(tensor)
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(tensor)
    return fut


nccl_ring_allreduce_hook = _ring_allreduce_hook
mpi_ring_allreduce_hook  = _ring_allreduce_hook


# ═══════════════════════════════════════════════════════════════════════════════
# RECURSIVE DOUBLING
# ═══════════════════════════════════════════════════════════════════════════════

def _recursive_doubling_allreduce_hook(
    state:  Optional[object],
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    tensor     = bucket.buffer()

    if world_size == 1:
        fut = torch.futures.Future()
        fut.set_result(tensor)
        return fut

    pow2      = 1 << (world_size.bit_length() - 1)
    remainder = world_size - pow2
    buf       = tensor.clone()
    recv_buf  = torch.empty_like(tensor)
    participating = True

    if remainder > 0:
        if rank < remainder:
            dist.recv(recv_buf, src=pow2 + rank)
            buf.add_(recv_buf)
        elif rank >= pow2:
            dist.send(buf.contiguous(), dst=rank - pow2)
            participating = False

    if participating:
        for step in range(int(math.log2(pow2))):
            partner  = rank ^ (1 << step)
            send_req = dist.isend(buf.contiguous(), dst=partner)
            dist.recv(recv_buf, src=partner)
            send_req.wait()
            buf.add_(recv_buf)

    if remainder > 0:
        if rank < remainder:
            dist.send(buf.contiguous(), dst=pow2 + rank)
        elif rank >= pow2:
            dist.recv(buf, src=rank - pow2)

    tensor.copy_(buf)
    tensor.div_(world_size)

    fut = torch.futures.Future()
    fut.set_result(tensor)
    return fut

nccl_recursive_doubling_allreduce_hook = _recursive_doubling_allreduce_hook
mpi_recursive_doubling_allreduce_hook  = _recursive_doubling_allreduce_hook


# ═══════════════════════════════════════════════════════════════════════════════
# HOOK FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

_HOOK_REGISTRY = {
    ("nccl", "default"):            nccl_default_allreduce_hook,
    ("nccl", "ring"):               nccl_ring_allreduce_hook,
    ("nccl", "recursive_doubling"): nccl_recursive_doubling_allreduce_hook,
    ("mpi",  "default"):            mpi_default_allreduce_hook,
    ("mpi",  "ring"):               mpi_ring_allreduce_hook,
    ("mpi",  "recursive_doubling"): mpi_recursive_doubling_allreduce_hook,
}


def get_comm_hook(
    backend:   str,
    algorithm: Optional[str] = None,
) -> Optional[tuple]:
    """
    Return (hook, state) for DDP's register_comm_hook, or None.

    Usage:
        result = get_comm_hook(backend, algorithm)
        if result is not None:
            hook, state = result
            model.register_comm_hook(state=state, hook=hook)
    """
    if algorithm is None:
        return None
    key = (backend, algorithm)
    if key not in _HOOK_REGISTRY:
        available = [f"{b}/{a}" for b, a in _HOOK_REGISTRY]
        raise ValueError(
            f"Unknown hook: {backend}/{algorithm}. Available: {available}"
        )
    hook  = _HOOK_REGISTRY[key]
    state = RingAllreduceState() if algorithm == "ring" else None
    return hook, state


def list_available_hooks() -> list:
    return list(_HOOK_REGISTRY.keys())