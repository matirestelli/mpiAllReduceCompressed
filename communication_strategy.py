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
import ctypes
import ctypes.util
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import os
import threading

import torch
import torch.distributed as dist

try:
    from zfp_cuda import (
        ZfpCompressionConfig,
        compress_into as _zfp_compress_into,
        decompress_into as _zfp_decompress_into,
        max_output_bytes as _zfp_max_output_bytes,
    )
    _ZFP_IMPORT_ERROR = None
except Exception as exc:
    ZfpCompressionConfig = None
    _zfp_compress_into = None
    _zfp_decompress_into = None
    _zfp_max_output_bytes = None
    _ZFP_IMPORT_ERROR = exc

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

ENABLE_NVTX_PROFILING = os.getenv("DDP_PROFILE_NVTX", "0") == "1"

_NVTX_LIB = None
_NVTX_RANGE_START = None
_NVTX_RANGE_END = None

if ENABLE_NVTX_PROFILING:
    _nvtx_path = ctypes.util.find_library("nvToolsExt")
    if _nvtx_path is not None:
        try:
            _NVTX_LIB = ctypes.CDLL(_nvtx_path)
            _NVTX_RANGE_START = _NVTX_LIB.nvtxRangeStartA
            _NVTX_RANGE_START.argtypes = [ctypes.c_char_p]
            _NVTX_RANGE_START.restype = ctypes.c_uint64
            _NVTX_RANGE_END = _NVTX_LIB.nvtxRangeEnd
            _NVTX_RANGE_END.argtypes = [ctypes.c_uint64]
            _NVTX_RANGE_END.restype = None
        except Exception:
            _NVTX_LIB = None
            _NVTX_RANGE_START = None
            _NVTX_RANGE_END = None

# Current training step — updated by ddp_training.py each batch via set_profiling_step().
_current_epoch: int = 0
_current_batch: int = 0


def set_profiling_step(epoch: int, batch: int) -> None:
    global _current_epoch, _current_batch
    _current_epoch, _current_batch = epoch, batch


def _nvtx_range_push(msg: str) -> None:
    if ENABLE_NVTX_PROFILING and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(msg)


def _nvtx_range_pop() -> None:
    if ENABLE_NVTX_PROFILING and torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def _nvtx_process_range_start(msg: str) -> Optional[int]:
    """Start a process-wide NVTX range that may be ended from another thread."""
    if not ENABLE_NVTX_PROFILING or _NVTX_RANGE_START is None:
        return None
    return int(_NVTX_RANGE_START(msg.encode("utf-8", errors="replace")))


def _nvtx_process_range_end(range_id: Optional[int]) -> None:
    if range_id is not None and _NVTX_RANGE_END is not None:
        _NVTX_RANGE_END(ctypes.c_uint64(range_id))


# ── Per-bucket compute-window NVTX ranges ────────────────────────────────────
# _bucket_fire_seq      = sequential bucket firing index within this backward.
#                         B=0 is the first bucket whose hook fires, B=N last.
# _known_num_buckets    = learned from GradBucket.is_last(), then reused.
# _compute_range_open   = True when a "compute:B=k" range is on the stack.
_bucket_fire_seq: int = 0
_known_num_buckets: Optional[int] = None
_compute_range_open: bool = False


def reset_bucket_compute_markers() -> None:
    """Close any leftover compute range from the previous backward. Call before loss.backward()."""
    global _bucket_fire_seq, _compute_range_open
    if _compute_range_open:
        _nvtx_range_pop()
        _compute_range_open = False
    _bucket_fire_seq = 0


def open_first_bucket_compute_range() -> None:
    """Open compute:B=0 on the main thread. Call after pushing the 'backward' NVTX range."""
    global _compute_range_open
    if ENABLE_NVTX_PROFILING:
        _nvtx_range_push("compute:B=0")
        _compute_range_open = True


def _profiled_hook(label: str, hook_fn, state, bucket: dist.GradBucket):
    global _bucket_fire_seq, _known_num_buckets, _compute_range_open
    tensor = bucket.buffer()
    numel = tensor.numel()
    epoch, batch = _current_epoch, _current_batch

    # Use the actual hook firing order inside this backward.  This avoids relying
    # on bucket.index() in PyTorch builds where it always reports 0, and avoids
    # collisions when two buckets have the same numel.
    b_idx = _bucket_fire_seq
    _bucket_fire_seq += 1
    is_last_bucket = False
    bucket_is_last = getattr(bucket, "is_last", None)
    if callable(bucket_is_last):
        is_last_bucket = bool(bucket_is_last())
    elif _known_num_buckets is not None:
        is_last_bucket = b_idx == _known_num_buckets - 1
    if is_last_bucket:
        _known_num_buckets = b_idx + 1

    # Close the compute:B=k range opened at backward start or after B=k-1's hook.
    # Runs on the main thread → appears in the same NVTX row as "backward".
    if _compute_range_open:
        _nvtx_range_pop()
        _compute_range_open = False

    # hook_called: microseconds wide for async hooks (just dispatches the op);
    # spans the full comm for synchronous hooks (ring, recursive_doubling).
    _nvtx_range_push(
        f"hook_called:{label} B={b_idx} epoch={epoch} batch={batch} numel={numel}"
    )
    try:
        fut = hook_fn(state, bucket)
    finally:
        _nvtx_range_pop()

    # Open compute:B=k+1 for the next bucket.
    # Skip the push after the last bucket; otherwise the range is closed by the
    # next bucket hook, measuring bucket-to-bucket backward computation.
    if ENABLE_NVTX_PROFILING and not is_last_bucket:
        _nvtx_range_push(f"compute:B={b_idx + 1}")
        _compute_range_open = True

    # comm_done fires in finalize_backward() when DDP awaits the future.
    # Gap between hook_called end and comm_done start = comm overlapped with
    # gradient computation (free overlap). Zero gap for synchronous hooks.
    def _on_done(f):
        _nvtx_range_push(
            f"comm_done:{label} B={b_idx} epoch={epoch} batch={batch}"
        )
        result = f.value()
        _nvtx_range_pop()
        return result

    return fut.then(_on_done)

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
    backend = dist.get_backend()
    rank = dist.get_rank()
    numel = tensor.numel()
    tensor.div_(dist.get_world_size())
    _nvtx_range_push(
        f"allreduce_launched backend={backend} rank={rank} "
        f"bucket={bucket.index()} numel={numel}"
    )
    _nvtx_range_pop()
    range_id = _nvtx_process_range_start(
        f"async_allreduce:{backend} rank={rank} bucket={bucket.index()} numel={numel}"
    )
    fut = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True).get_future()

    def _finish(f):
        _nvtx_process_range_end(range_id)
        return f.value()[0]

    return fut.then(_finish)

# Hook alias: baseline PyTorch-style allreduce using dist.all_reduce.
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


def _cuda_device_sync(t: torch.Tensor) -> None:
    """Full-device sync for external CUDA libraries not stream-ordered with PyTorch."""
    if t.is_cuda:
        torch.cuda.synchronize(device=t.device)


def _require_zfp_cuda() -> None:
    if _ZFP_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ZFP CUDA support is unavailable. Build/runtime error while importing "
            "`zfp_cuda.py`. Ensure libzfp with CUDA support is installed and "
            "`ZFP_HOME` is set if needed."
        ) from _ZFP_IMPORT_ERROR

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
    _nvtx_range_push(f"ring_reduce_scatter:B{bstate.bucket_id} rank={rank}")
    for step in range(world_size - 1):
        send_idx = (rank - step - 1 + world_size) % world_size
        recv_idx = (rank - step - 2 + world_size) % world_size

        # Fresh unique tag — Cray MPICH will never reuse a cached registration
        tag = _next_tag()

        bstate.send_bufs[send_idx].copy_(chunks[send_idx])
        _nvtx_range_push(f"cuda_sync_pre_mpi:B{bstate.bucket_id} phase=1 step={step}")
        _cuda_sync(bstate.send_bufs[send_idx])
        _nvtx_range_pop()

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

    _nvtx_range_pop()  # ring_reduce_scatter

    # ── Phase 2: Allgather ────────────────────────────────────────────────────
    _nvtx_range_push(f"ring_allgather:B{bstate.bucket_id} rank={rank}")
    for step in range(world_size - 1):
        send_idx = (rank - step     + world_size) % world_size
        recv_idx = (rank - step - 1 + world_size) % world_size

        tag = _next_tag()

        bstate.send_bufs[send_idx].copy_(chunks[send_idx])
        _nvtx_range_push(f"cuda_sync_pre_mpi:B{bstate.bucket_id} phase=2 step={step}")
        _cuda_sync(bstate.send_bufs[send_idx])
        _nvtx_range_pop()

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

    _nvtx_range_pop()  # ring_allgather

    tensor.copy_(bstate.flat.view_as(tensor))
    tensor.div_(world_size)
    _nvtx_range_push(f"cuda_sync_post_ring:B{bstate.bucket_id} rank={rank}")
    _cuda_sync(tensor)   # ensure div_ has landed before hook resolves the future
    _nvtx_range_pop()

# Hook: plain ring allreduce with no compression.
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

    _nvtx_range_push(f"bucket_grads_ready:B{bucket_index} rank={rank} call={call_n}")
    _nvtx_range_pop()
    _nvtx_range_push(
        f"comm_hook:ring rank={rank} bucket={bucket_index} call={call_n} numel={numel}"
    )
    try:
        _ring_allreduce_sum(tensor, bstate, log=do_full_log, call_n=call_n)
    finally:
        _nvtx_range_pop()

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


# Hook aliases: plain ring allreduce with no compression.
nccl_ring_allreduce_hook = _ring_allreduce_hook
mpi_ring_allreduce_hook  = _ring_allreduce_hook


# ═══════════════════════════════════════════════════════════════════════════════
# RECURSIVE DOUBLING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RecursiveDoublingBucketState:
    """Persistent GPU buffers for one DDP bucket."""
    flat:       Optional[torch.Tensor] = None
    tmp:        Optional[torch.Tensor] = None
    bucket_id:  int                    = 0
    call_count: int                    = 0

    def initialize(self, tensor: torch.Tensor, bucket_id: int) -> None:
        numel = tensor.numel()
        self.flat = torch.empty(numel, dtype=tensor.dtype, device=tensor.device)
        self.tmp = torch.empty_like(self.flat)
        self.bucket_id = bucket_id


@dataclass
class RecursiveDoublingAllreduceState:
    """Per-bucket recursive doubling state, keyed by DDP bucket index."""
    buckets: Dict[int, RecursiveDoublingBucketState] = field(default_factory=dict)

    def get_or_init(self, bucket_index: int,
                    tensor: torch.Tensor) -> RecursiveDoublingBucketState:
        bs = self.buckets.get(bucket_index)
        if bs is None or bs.flat is None or bs.flat.numel() != tensor.numel():
            bs = RecursiveDoublingBucketState()
            bs.initialize(tensor, bucket_index)
            self.buckets[bucket_index] = bs
        return bs


def _recursive_doubling_allreduce_sum(
    tensor: torch.Tensor,
    bstate: RecursiveDoublingBucketState,
    log: bool,
    call_n: int = 0,
) -> None:
    """
    In-place recursive doubling allreduce with monotonically increasing tags.

    The non-power-of-two case follows the MPICH structure:
      1. peel off the first 2*rem ranks into even/odd pairs,
      2. recursive-double across the surviving power-of-two ranks,
      3. send the final result back to the peeled even ranks.

    All ranks consume the same number of _next_tag() calls in the same order,
    even when a rank is inactive for a phase, so future hook invocations stay
    tag-aligned across the job.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size == 1:
        return

    pfx = f"[R{rank}|B{bstate.bucket_id}|C{call_n}]"

    assert bstate.flat is not None
    assert bstate.tmp is not None
    work = bstate.flat
    tmp = bstate.tmp

    work.copy_(tensor.flatten())

    if log:
        print(f"{pfx} flat after copy from tensor: {_fmt(work)}", flush=True)

    pof2 = 1
    while pof2 <= world_size:
        pof2 <<= 1
    pof2 >>= 1
    rem = world_size - pof2

    newrank = -1

    _nvtx_range_push(f"recursive_doubling_peel:B{bstate.bucket_id} rank={rank}")
    peel_tag = _next_tag()
    if rank < 2 * rem:
        partner = rank + 1 if rank % 2 == 0 else rank - 1
        _nvtx_range_push(f"cuda_sync_pre_mpi:B{bstate.bucket_id} phase=peel")
        _cuda_sync(work)
        _nvtx_range_pop()
        if rank % 2 == 0:
            if log:
                print(f"{pfx} peel tag={peel_tag} SENDING -> rank{partner}: {_fmt(work)}",
                      flush=True)
            req = dist.isend(work, dst=partner, tag=peel_tag)
            req.wait()
            newrank = -1
        else:
            if log:
                print(f"{pfx} peel tag={peel_tag} EXPECTING <- rank{partner}",
                      flush=True)
            req = dist.irecv(tmp, src=partner, tag=peel_tag)
            req.wait()
            work.add_(tmp)
            newrank = rank // 2
            if log:
                print(f"{pfx} peel tag={peel_tag} RECVD <- rank{partner}: {_fmt(tmp)}",
                      flush=True)
                print(f"{pfx}   after peel add: {_fmt(work)}", flush=True)
    else:
        newrank = rank - rem
    _nvtx_range_pop()

    _nvtx_range_push(f"recursive_doubling_main:B{bstate.bucket_id} rank={rank}")
    mask = 1
    while mask < pof2:
        step_tag = _next_tag()
        if newrank != -1:
            newdst = newrank ^ mask
            dst = (newdst << 1) + 1 if newdst < rem else newdst + rem

            _nvtx_range_push(
                f"cuda_sync_pre_mpi:B{bstate.bucket_id} phase=main mask={mask}"
            )
            _cuda_sync(work)
            _nvtx_range_pop()

            if log:
                print(f"{pfx} main mask={mask} tag={step_tag} "
                      f"SENDRECV <-> rank{dst}: {_fmt(work)}", flush=True)

            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.irecv, tmp, dst, tag=step_tag),
                dist.P2POp(dist.isend, work, dst, tag=step_tag),
            ])
            for req in reqs:
                req.wait()

            work.add_(tmp)

            if log:
                print(f"{pfx} main mask={mask} tag={step_tag} "
                      f"RECVD <- rank{dst}: {_fmt(tmp)}", flush=True)
                print(f"{pfx}   after main add: {_fmt(work)}", flush=True)
        mask <<= 1
    _nvtx_range_pop()

    _nvtx_range_push(f"recursive_doubling_finalize:B{bstate.bucket_id} rank={rank}")
    final_tag = _next_tag()
    if rank < 2 * rem:
        partner = rank - 1 if rank % 2 else rank + 1
        _nvtx_range_push(f"cuda_sync_pre_mpi:B{bstate.bucket_id} phase=final")
        _cuda_sync(work)
        _nvtx_range_pop()
        if rank % 2:
            if log:
                print(f"{pfx} final tag={final_tag} SENDING -> rank{partner}: {_fmt(work)}",
                      flush=True)
            req = dist.isend(work, dst=partner, tag=final_tag)
            req.wait()
        else:
            if log:
                print(f"{pfx} final tag={final_tag} EXPECTING <- rank{partner}",
                      flush=True)
            req = dist.irecv(work, src=partner, tag=final_tag)
            req.wait()
            if log:
                print(f"{pfx} final tag={final_tag} RECVD <- rank{partner}: {_fmt(work)}",
                      flush=True)
    _nvtx_range_pop()

    tensor.copy_(work.view_as(tensor))
    tensor.div_(world_size)
    _nvtx_range_push(f"cuda_sync_post_recursive_doubling:B{bstate.bucket_id} rank={rank}")
    _cuda_sync(tensor)
    _nvtx_range_pop()

# Hook: plain recursive-doubling allreduce with no compression.
def _recursive_doubling_allreduce_hook(
    state:  RecursiveDoublingAllreduceState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    tensor = bucket.buffer()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    numel = tensor.numel()
    bucket_index = bucket.index()

    bstate = state.get_or_init(bucket_index, tensor)
    bstate.call_count += 1
    call_n = bstate.call_count

    do_full_log = (FULL_LOG_CALLS > 0 and call_n <= FULL_LOG_CALLS)
    do_light_log = (not do_full_log and LIGHT_LOG_CALLS > 0
                    and call_n <= LIGHT_LOG_CALLS)

    if do_full_log or do_light_log:
        pre = tensor.clone()

    if do_full_log:
        tid = threading.get_ident()
        print(f"\n[RECURSIVE_DOUBLING | R{rank}|B{bstate.bucket_id}|C{call_n}] "
              f"thread={tid} global_tag_before={_tag_counter} numel={numel}", flush=True)
        print(f"  [R{rank}] INPUT tensor: {_fmt(tensor)}", flush=True)
        print(f"  [R{rank}] flat.data_ptr={hex(bstate.flat.data_ptr())}", flush=True)
        print(f"  [R{rank}] tmp.data_ptr={hex(bstate.tmp.data_ptr())}", flush=True)

    _nvtx_range_push(f"bucket_grads_ready:B{bucket_index} rank={rank} call={call_n}")
    _nvtx_range_pop()
    _nvtx_range_push(
        f"comm_hook:recursive_doubling rank={rank} bucket={bucket_index} call={call_n} numel={numel}"
    )
    try:
        _recursive_doubling_allreduce_sum(
            tensor,
            bstate,
            log=do_full_log,
            call_n=call_n,
        )
    finally:
        _nvtx_range_pop()

    # if do_full_log or do_light_log:
    #     ref = pre.clone()
    #     dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    #     ref.div_(world_size)
    #     max_err = (tensor - ref).abs().max().item()
    #     rel_err = max_err / (ref.norm().item() + 1e-12)
    #     match = "✓ MATCH" if max_err < 1e-4 else "✗ MISMATCH"
    #
    # if do_full_log:
    #     print(f"  [R{rank}|B{bstate.bucket_id}|C{call_n}] global_tag_after={_tag_counter}",
    #           flush=True)
    #     print(f"  [VERIFY | R{rank}|B{bstate.bucket_id}|C{call_n}] "
    #           f"max_abs_err={max_err:.3e}  rel={rel_err:.3e}  {match}", flush=True)
    #     if max_err >= 1e-4:
    #         err_pos = (tensor - ref).abs().argmax().item()
    #         print(f"  [R{rank}] recursive_doubling: {_fmt(tensor.detach())} "
    #               f"worst_err_pos={err_pos}", flush=True)
    #         print(f"  [R{rank}] allreduce: {_fmt(ref.detach())}", flush=True)
    #
    # elif do_light_log:
    #     in_norm = pre.norm().item()
    #     out_norm = tensor.norm().item()
    #     is_nan = "NaN!" if not torch.isfinite(tensor).all() else ""
    #     print(f"  [LIGHT | R{rank}|B{bstate.bucket_id}|C{call_n}] "
    #           f"in_norm={in_norm:.4e} out_norm={out_norm:.4e} "
    #           f"sample={_fmt(tensor)} {match} {is_nan}", flush=True)
    #     if max_err >= 1e-4:
    #         print(f"  [LIGHT | R{rank}|B{bstate.bucket_id}|C{call_n}] "
    #               f"*** MISMATCH *** max_abs_err={max_err:.3e}", flush=True)

    _cuda_sync(tensor)
    fut = torch.futures.Future()
    fut.set_result(tensor)
    return fut

# Hook aliases: plain recursive-doubling allreduce with no compression.
nccl_recursive_doubling_allreduce_hook = _recursive_doubling_allreduce_hook
mpi_recursive_doubling_allreduce_hook  = _recursive_doubling_allreduce_hook


# ═══════════════════════════════════════════════════════════════════════════════
# ZFP-COMPRESSED RING ALLREDUCE — naive point-to-point baseline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ZfpRingBucketState:
    """Persistent buffers for the paper's naive P2P ZFP baseline on ring."""
    flat:            Optional[torch.Tensor] = None
    cnts:            List[int]              = field(default_factory=list)
    displs:          List[int]              = field(default_factory=list)
    send_comp:       List[torch.Tensor]     = field(default_factory=list)
    recv_comp:       List[torch.Tensor]     = field(default_factory=list)
    recv_decomp:     List[torch.Tensor]     = field(default_factory=list)
    send_sizes:      List[torch.Tensor]     = field(default_factory=list)
    recv_sizes:      List[torch.Tensor]     = field(default_factory=list)
    bucket_id:       int                    = 0
    call_count:      int                    = 0
    ready:           bool                   = False

    def initialize(self, tensor: torch.Tensor, world_size: int, bucket_id: int) -> None:
        _require_zfp_cuda()
        cfg = ZfpCompressionConfig()
        numel = tensor.numel()
        base_chunk = math.ceil(numel / world_size)

        total = 0
        self.cnts = []
        for _ in range(world_size):
            c = min(base_chunk, numel - total)
            self.cnts.append(max(c, 0))
            total += self.cnts[-1]

        self.displs = [0]
        for i in range(1, world_size):
            self.displs.append(self.displs[-1] + self.cnts[i - 1])

        self.flat = torch.empty(numel, dtype=tensor.dtype, device=tensor.device)
        self.send_comp = []
        self.recv_comp = []
        self.recv_decomp = []
        self.send_sizes = []
        self.recv_sizes = []

        for cnt in self.cnts:
            recv_tensor = torch.empty(cnt, dtype=tensor.dtype, device=tensor.device)
            self.recv_decomp.append(recv_tensor)
            probe = torch.empty(cnt, dtype=tensor.dtype, device=tensor.device)
            max_bytes = _zfp_max_output_bytes(probe, cfg.rate)
            self.send_comp.append(torch.empty(max_bytes, dtype=torch.uint8, device=tensor.device))
            self.recv_comp.append(torch.empty(max_bytes, dtype=torch.uint8, device=tensor.device))
            self.send_sizes.append(torch.zeros(1, dtype=torch.int64, device=tensor.device))
            self.recv_sizes.append(torch.zeros(1, dtype=torch.int64, device=tensor.device))

        self.bucket_id = bucket_id
        self.ready = True


@dataclass
class ZfpRingAllreduceState:
    buckets: Dict[int, ZfpRingBucketState] = field(default_factory=dict)

    def get_or_init(self, bucket_index: int,
                    tensor: torch.Tensor, world_size: int) -> ZfpRingBucketState:
        bs = self.buckets.get(bucket_index)
        if bs is None or bs.flat is None or bs.flat.numel() != tensor.numel():
            bs = ZfpRingBucketState()
            bs.initialize(tensor, world_size, bucket_index)
            self.buckets[bucket_index] = bs
        return bs


def _ring_allreduce_zfp_sum(
    tensor: torch.Tensor,
    bstate: ZfpRingBucketState,
    log: bool,
    call_n: int = 0,
) -> None:
    """
    Naive point-to-point ZFP baseline for ring allreduce.

    This intentionally preserves the inefficiencies called out in the paper:
      1. compress before every send,
      2. decompress after every receive,
      3. reduce on decompressed values,
      4. recompress already-received data again during phase 2 allgather.

    This is *not* the paper's later collective-level online compression scheme.
    """
    _require_zfp_cuda()
    cfg = ZfpCompressionConfig()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size == 1:
        return

    pfx = f"[R{rank}|B{bstate.bucket_id}|C{call_n}]"
    assert bstate.flat is not None
    bstate.flat.copy_(tensor.flatten())
    chunks = [
        bstate.flat[bstate.displs[i]: bstate.displs[i] + bstate.cnts[i]]
        for i in range(world_size)
    ]

    src = (rank - 1 + world_size) % world_size
    dst = (rank + 1) % world_size

    # Phase 1: standard ring reduce-scatter.
    # Baseline behavior: each outgoing chunk is freshly compressed right before
    # the send, and each incoming payload is decompressed before the local add.
    _nvtx_range_push(f"zfp_ring_reduce_scatter:B{bstate.bucket_id} rank={rank}")
    for step in range(world_size - 1):
        send_idx = (rank - step - 1 + world_size) % world_size
        recv_idx = (rank - step - 2 + world_size) % world_size

        size_tag = _next_tag()
        data_tag = _next_tag()

        # Naive P2P compression on the sender side.
        _nvtx_range_push(f"zfp_compress:B{bstate.bucket_id} phase=1 step={step}")
        used_send = _zfp_compress_into(chunks[send_idx], bstate.send_comp[send_idx], cfg.rate)
        bstate.send_sizes[send_idx].fill_(used_send)
        _cuda_device_sync(bstate.send_comp[send_idx])
        _nvtx_range_pop()

        if log:
            print(f"{pfx} ZFP P1 step={step} size_tag={size_tag} data_tag={data_tag} "
                  f"send_idx={send_idx} recv_idx={recv_idx} send_bytes={used_send}",
                  flush=True)

        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.irecv, bstate.recv_sizes[recv_idx], src, tag=size_tag),
            dist.P2POp(dist.isend, bstate.send_sizes[send_idx], dst, tag=size_tag),
        ])
        for req in reqs:
            req.wait()

        recv_nbytes = int(bstate.recv_sizes[recv_idx].item())
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.irecv, bstate.recv_comp[recv_idx][:recv_nbytes], src, tag=data_tag),
            dist.P2POp(dist.isend, bstate.send_comp[send_idx][:used_send], dst, tag=data_tag),
        ])
        for req in reqs:
            req.wait()

        # Naive P2P decompression on the receiver side before reduction.
        _nvtx_range_push(f"zfp_decompress:B{bstate.bucket_id} phase=1 step={step}")
        _zfp_decompress_into(
            bstate.recv_comp[recv_idx],
            recv_nbytes,
            bstate.recv_decomp[recv_idx],
            cfg.rate,
        )
        _cuda_device_sync(bstate.recv_decomp[recv_idx])
        _nvtx_range_pop()
        chunks[recv_idx].add_(bstate.recv_decomp[recv_idx])

    _nvtx_range_pop()

    # Phase 2: standard ring allgather.
    # The paper notes this baseline is inefficient because data that was already
    # decompressed and aggregated is compressed again for forwarding.
    _nvtx_range_push(f"zfp_ring_allgather:B{bstate.bucket_id} rank={rank}")
    for step in range(world_size - 1):
        send_idx = (rank - step + world_size) % world_size
        recv_idx = (rank - step - 1 + world_size) % world_size

        size_tag = _next_tag()
        data_tag = _next_tag()

        # Intentional baseline inefficiency: recompress before forwarding.
        _nvtx_range_push(f"zfp_compress:B{bstate.bucket_id} phase=2 step={step}")
        used_send = _zfp_compress_into(chunks[send_idx], bstate.send_comp[send_idx], cfg.rate)
        bstate.send_sizes[send_idx].fill_(used_send)
        _cuda_device_sync(bstate.send_comp[send_idx])
        _nvtx_range_pop()

        if log:
            print(f"{pfx} ZFP P2 step={step} size_tag={size_tag} data_tag={data_tag} "
                  f"send_idx={send_idx} recv_idx={recv_idx} send_bytes={used_send}",
                  flush=True)

        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.irecv, bstate.recv_sizes[recv_idx], src, tag=size_tag),
            dist.P2POp(dist.isend, bstate.send_sizes[send_idx], dst, tag=size_tag),
        ])
        for req in reqs:
            req.wait()

        recv_nbytes = int(bstate.recv_sizes[recv_idx].item())
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.irecv, bstate.recv_comp[recv_idx][:recv_nbytes], src, tag=data_tag),
            dist.P2POp(dist.isend, bstate.send_comp[send_idx][:used_send], dst, tag=data_tag),
        ])
        for req in reqs:
            req.wait()

        # Decompress received payload back into floating-point values.
        _nvtx_range_push(f"zfp_decompress:B{bstate.bucket_id} phase=2 step={step}")
        _zfp_decompress_into(
            bstate.recv_comp[recv_idx],
            recv_nbytes,
            bstate.recv_decomp[recv_idx],
            cfg.rate,
        )
        _cuda_device_sync(bstate.recv_decomp[recv_idx])
        _nvtx_range_pop()
        chunks[recv_idx].copy_(bstate.recv_decomp[recv_idx])

    _nvtx_range_pop()

    tensor.copy_(bstate.flat.view_as(tensor))
    tensor.div_(world_size)
    _cuda_sync(tensor)

# Hook: naive point-to-point ZFP baseline layered on top of ring allreduce.
def _ring_allreduce_zfp_hook(
    state: ZfpRingAllreduceState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """DDP hook for the paper's naive ring + ZFP baseline."""
    tensor = bucket.buffer()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    numel = tensor.numel()
    bucket_index = bucket.index()

    bstate = state.get_or_init(bucket_index, tensor, world_size)
    bstate.call_count += 1
    call_n = bstate.call_count

    do_full_log = (FULL_LOG_CALLS > 0 and call_n <= FULL_LOG_CALLS)

    if do_full_log:
        print(f"\n[RING_ZFP | R{rank}|B{bstate.bucket_id}|C{call_n}] numel={numel} "
              f"global_tag_before={_tag_counter}", flush=True)

    _nvtx_range_push(
        f"comm_hook:ring_zfp rank={rank} bucket={bucket_index} call={call_n} numel={numel}"
    )
    try:
        _ring_allreduce_zfp_sum(tensor, bstate, log=do_full_log, call_n=call_n)
    finally:
        _nvtx_range_pop()

    _cuda_sync(tensor)
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(tensor)
    return fut


# Hook aliases: naive point-to-point ZFP baseline layered on top of ring.
nccl_ring_zfp_allreduce_hook = _ring_allreduce_zfp_hook
mpi_ring_zfp_allreduce_hook  = _ring_allreduce_zfp_hook


# ═══════════════════════════════════════════════════════════════════════════════
# ZFP-COMPRESSED RECURSIVE DOUBLING — naive point-to-point baseline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ZfpRecursiveDoublingBucketState:
    """Persistent buffers for the paper's naive P2P ZFP baseline on recursive doubling."""
    flat:       Optional[torch.Tensor] = None
    tmp:        Optional[torch.Tensor] = None
    send_comp:  Optional[torch.Tensor] = None
    recv_comp:  Optional[torch.Tensor] = None
    send_size:  Optional[torch.Tensor] = None
    recv_size:  Optional[torch.Tensor] = None
    bucket_id:  int                    = 0
    call_count: int                    = 0

    def initialize(self, tensor: torch.Tensor, bucket_id: int) -> None:
        _require_zfp_cuda()
        cfg = ZfpCompressionConfig()
        self.flat = torch.empty(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
        self.tmp = torch.empty_like(self.flat)
        max_bytes = _zfp_max_output_bytes(self.flat, cfg.rate)
        self.send_comp = torch.empty(max_bytes, dtype=torch.uint8, device=tensor.device)
        self.recv_comp = torch.empty(max_bytes, dtype=torch.uint8, device=tensor.device)
        self.send_size = torch.zeros(1, dtype=torch.int64, device=tensor.device)
        self.recv_size = torch.zeros(1, dtype=torch.int64, device=tensor.device)
        self.bucket_id = bucket_id


@dataclass
class ZfpRecursiveDoublingAllreduceState:
    buckets: Dict[int, ZfpRecursiveDoublingBucketState] = field(default_factory=dict)

    def get_or_init(self, bucket_index: int,
                    tensor: torch.Tensor) -> ZfpRecursiveDoublingBucketState:
        bs = self.buckets.get(bucket_index)
        if bs is None or bs.flat is None or bs.flat.numel() != tensor.numel():
            bs = ZfpRecursiveDoublingBucketState()
            bs.initialize(tensor, bucket_index)
            self.buckets[bucket_index] = bs
        return bs


def _zfp_sendrecv_tensor(
    work: torch.Tensor,
    send_comp: torch.Tensor,
    recv_comp: torch.Tensor,
    send_size: torch.Tensor,
    recv_size: torch.Tensor,
    recv_out: torch.Tensor,
    peer: int,
    size_tag: int,
    data_tag: int,
    rate: float,
) -> None:
    """One naive compressed pairwise exchange: compress, exchange, decompress."""
    used_send = _zfp_compress_into(work, send_comp, rate)
    send_size.fill_(used_send)
    _cuda_device_sync(send_comp)

    reqs = dist.batch_isend_irecv([
        dist.P2POp(dist.irecv, recv_size, peer, tag=size_tag),
        dist.P2POp(dist.isend, send_size, peer, tag=size_tag),
    ])
    for req in reqs:
        req.wait()

    recv_nbytes = int(recv_size.item())
    reqs = dist.batch_isend_irecv([
        dist.P2POp(dist.irecv, recv_comp[:recv_nbytes], peer, tag=data_tag),
        dist.P2POp(dist.isend, send_comp[:used_send], peer, tag=data_tag),
    ])
    for req in reqs:
        req.wait()

    _zfp_decompress_into(recv_comp, recv_nbytes, recv_out, rate)
    _cuda_device_sync(recv_out)


def _recursive_doubling_zfp_sum(
    tensor: torch.Tensor,
    bstate: ZfpRecursiveDoublingBucketState,
    log: bool,
    call_n: int = 0,
) -> None:
    """
    Naive point-to-point ZFP baseline for recursive doubling.

    Each recursive-doubling exchange remains structurally identical to the
    original collective, but every send is preceded by ZFP compression and every
    receive is followed by ZFP decompression before local reduction.

    This intentionally introduces the extra compression/decompression and sync
    overheads described in the paper's baseline discussion.
    """
    _require_zfp_cuda()
    cfg = ZfpCompressionConfig()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size == 1:
        return

    pfx = f"[R{rank}|B{bstate.bucket_id}|C{call_n}]"
    assert bstate.flat is not None
    assert bstate.tmp is not None
    assert bstate.send_comp is not None
    assert bstate.recv_comp is not None
    assert bstate.send_size is not None
    assert bstate.recv_size is not None

    work = bstate.flat
    tmp = bstate.tmp
    work.copy_(tensor.flatten())

    pof2 = 1
    while pof2 <= world_size:
        pof2 <<= 1
    pof2 >>= 1
    rem = world_size - pof2
    newrank = -1

    # Phase 0: non-power-of-two peel.
    # Baseline behavior: the peeled exchange itself is also wrapped in
    # compress/send and recv/decompress.
    peel_size_tag = _next_tag()
    peel_data_tag = _next_tag()
    _nvtx_range_push(f"zfp_recursive_doubling_peel:B{bstate.bucket_id} rank={rank}")
    if rank < 2 * rem:
        partner = rank + 1 if rank % 2 == 0 else rank - 1
        if rank % 2 == 0:
            # Even peeled rank: compress and send its full working buffer.
            used_send = _zfp_compress_into(work, bstate.send_comp, cfg.rate)
            bstate.send_size.fill_(used_send)
            _cuda_device_sync(bstate.send_comp)
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, bstate.send_size, partner, tag=peel_size_tag),
            ])
            for req in reqs:
                req.wait()
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, bstate.send_comp[:used_send], partner, tag=peel_data_tag),
            ])
            for req in reqs:
                req.wait()
            newrank = -1
        else:
            # Odd peeled rank: receive compressed payload, decompress, reduce.
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.irecv, bstate.recv_size, partner, tag=peel_size_tag),
            ])
            for req in reqs:
                req.wait()
            recv_nbytes = int(bstate.recv_size.item())
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.irecv, bstate.recv_comp[:recv_nbytes], partner, tag=peel_data_tag),
            ])
            for req in reqs:
                req.wait()
            _zfp_decompress_into(bstate.recv_comp, recv_nbytes, tmp, cfg.rate)
            _cuda_device_sync(tmp)
            work.add_(tmp)
            newrank = rank // 2
            if log:
                print(f"{pfx} peel recv_bytes={recv_nbytes} from rank{partner}", flush=True)
    else:
        newrank = rank - rem
    _nvtx_range_pop()

    # Phase 1: standard recursive-doubling exchanges across the surviving pof2 ranks.
    # Baseline behavior: every pairwise exchange performs a fresh compression
    # before the send and a decompression before the local add.
    _nvtx_range_push(f"zfp_recursive_doubling_main:B{bstate.bucket_id} rank={rank}")
    mask = 1
    while mask < pof2:
        size_tag = _next_tag()
        data_tag = _next_tag()
        if newrank != -1:
            newdst = newrank ^ mask
            dst = (newdst << 1) + 1 if newdst < rem else newdst + rem
            _zfp_sendrecv_tensor(
                work,
                bstate.send_comp,
                bstate.recv_comp,
                bstate.send_size,
                bstate.recv_size,
                tmp,
                dst,
                size_tag,
                data_tag,
                cfg.rate,
            )
            work.add_(tmp)
            if log:
                print(f"{pfx} main mask={mask} peer={dst} recv_bytes={int(bstate.recv_size.item())}",
                      flush=True)
        mask <<= 1
    _nvtx_range_pop()

    # Phase 2: send the final reduced result back to the peeled ranks.
    # This also uses the same naive compress/send and recv/decompress pattern.
    final_size_tag = _next_tag()
    final_data_tag = _next_tag()
    _nvtx_range_push(f"zfp_recursive_doubling_finalize:B{bstate.bucket_id} rank={rank}")
    if rank < 2 * rem:
        partner = rank - 1 if rank % 2 else rank + 1
        if rank % 2:
            used_send = _zfp_compress_into(work, bstate.send_comp, cfg.rate)
            bstate.send_size.fill_(used_send)
            _cuda_device_sync(bstate.send_comp)
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, bstate.send_size, partner, tag=final_size_tag),
            ])
            for req in reqs:
                req.wait()
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, bstate.send_comp[:used_send], partner, tag=final_data_tag),
            ])
            for req in reqs:
                req.wait()
        else:
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.irecv, bstate.recv_size, partner, tag=final_size_tag),
            ])
            for req in reqs:
                req.wait()
            recv_nbytes = int(bstate.recv_size.item())
            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.irecv, bstate.recv_comp[:recv_nbytes], partner, tag=final_data_tag),
            ])
            for req in reqs:
                req.wait()
            _zfp_decompress_into(bstate.recv_comp, recv_nbytes, work, cfg.rate)
            _cuda_device_sync(work)
            if log:
                print(f"{pfx} final recv_bytes={recv_nbytes} from rank{partner}", flush=True)
    _nvtx_range_pop()

    tensor.copy_(work.view_as(tensor))
    tensor.div_(world_size)
    _cuda_sync(tensor)

# Hook: naive point-to-point ZFP baseline layered on top of recursive doubling.
def _recursive_doubling_zfp_hook(
    state: ZfpRecursiveDoublingAllreduceState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """DDP hook for the paper's naive recursive-doubling + ZFP baseline."""
    tensor = bucket.buffer()
    rank = dist.get_rank()
    numel = tensor.numel()
    bucket_index = bucket.index()

    bstate = state.get_or_init(bucket_index, tensor)
    bstate.call_count += 1
    call_n = bstate.call_count
    do_full_log = (FULL_LOG_CALLS > 0 and call_n <= FULL_LOG_CALLS)

    if do_full_log:
        print(f"\n[RECURSIVE_DOUBLING_ZFP | R{rank}|B{bstate.bucket_id}|C{call_n}] "
              f"numel={numel} global_tag_before={_tag_counter}", flush=True)

    _nvtx_range_push(
        f"comm_hook:recursive_doubling_zfp rank={rank} bucket={bucket_index} call={call_n} numel={numel}"
    )
    try:
        _recursive_doubling_zfp_sum(tensor, bstate, log=do_full_log, call_n=call_n)
    finally:
        _nvtx_range_pop()

    _cuda_sync(tensor)
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(tensor)
    return fut


# Hook aliases: naive point-to-point ZFP baseline layered on top of recursive doubling.
nccl_recursive_doubling_zfp_allreduce_hook = _recursive_doubling_zfp_hook
mpi_recursive_doubling_zfp_allreduce_hook  = _recursive_doubling_zfp_hook

# Explicit aliases with "naive" in the name to mirror the paper terminology.
nccl_ring_zfp_naive_allreduce_hook = _ring_allreduce_zfp_hook
mpi_ring_zfp_naive_allreduce_hook = _ring_allreduce_zfp_hook
nccl_recursive_doubling_zfp_naive_allreduce_hook = _recursive_doubling_zfp_hook
mpi_recursive_doubling_zfp_naive_allreduce_hook = _recursive_doubling_zfp_hook


# ═══════════════════════════════════════════════════════════════════════════════
# HOOK FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

_HOOK_REGISTRY = {
    ("nccl", "default"):            nccl_default_allreduce_hook,
    ("nccl", "ring"):               nccl_ring_allreduce_hook,
    ("nccl", "ring_zfp"):           nccl_ring_zfp_allreduce_hook,
    ("nccl", "ring_zfp_naive"):     nccl_ring_zfp_naive_allreduce_hook,
    ("nccl", "recursive_doubling"): nccl_recursive_doubling_allreduce_hook,
    ("nccl", "recursive_doubling_zfp"): nccl_recursive_doubling_zfp_allreduce_hook,
    ("nccl", "recursive_doubling_zfp_naive"): nccl_recursive_doubling_zfp_naive_allreduce_hook,
    ("mpi",  "default"):            mpi_default_allreduce_hook,
    ("mpi",  "ring"):               mpi_ring_allreduce_hook,
    ("mpi",  "ring_zfp"):           mpi_ring_zfp_allreduce_hook,
    ("mpi",  "ring_zfp_naive"):     mpi_ring_zfp_naive_allreduce_hook,
    ("mpi",  "recursive_doubling"): mpi_recursive_doubling_allreduce_hook,
    ("mpi",  "recursive_doubling_zfp"): mpi_recursive_doubling_zfp_allreduce_hook,
    ("mpi",  "recursive_doubling_zfp_naive"): mpi_recursive_doubling_zfp_naive_allreduce_hook,
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
    base_hook = _HOOK_REGISTRY[key]
    if algorithm == "ring":
        state = RingAllreduceState()
    elif algorithm in ("ring_zfp", "ring_zfp_naive"):
        state = ZfpRingAllreduceState()
    elif algorithm == "recursive_doubling":
        state = RecursiveDoublingAllreduceState()
    elif algorithm in ("recursive_doubling_zfp", "recursive_doubling_zfp_naive"):
        state = ZfpRecursiveDoublingAllreduceState()
    else:
        state = None

    def hook(state_obj, bucket):
        return _profiled_hook(f"{backend}:{algorithm}", base_hook, state_obj, bucket)

    return hook, state


def list_available_hooks() -> list:
    return list(_HOOK_REGISTRY.keys())
