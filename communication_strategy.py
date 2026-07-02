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

import time
from concurrent.futures import ThreadPoolExecutor

import os
import threading

import torch
import torch.distributed as dist

try:
    from zfp_api import (
        ZfpCompressionConfig,
        compress_into as _zfp_compress_into,
        decompress_into as _zfp_decompress_into,
        compress_into_current_stream as _zfp_compress_into_current_stream,
        decompress_into_current_stream as _zfp_decompress_into_current_stream,
        max_output_bytes as _zfp_max_output_bytes,
    )
    _ZFP_IMPORT_ERROR = None
except Exception as exc:
    ZfpCompressionConfig = None
    _zfp_compress_into = None
    _zfp_decompress_into = None
    _zfp_compress_into_current_stream = None
    _zfp_decompress_into_current_stream = None
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

# profile variables for measuring communication time, so time spend in the hooks
ENABLE_HOOK_TIMING = os.getenv("DDP_HOOK_TIMING", "0") == "1"
HOOK_TIMING_RANK0_ONLY = os.getenv("DDP_HOOK_TIMING_RANK0_ONLY", "1") == "1"
_HOOK_TIMING_STATS = {}
_HOOK_TAIL_STATS = {}
_HOOK_TIMING_LOCK = threading.Lock()


def _record_hook_work_timing(label: str, work_ms: float) -> None:
    if not ENABLE_HOOK_TIMING:
        return
    if HOOK_TIMING_RANK0_ONLY and dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    try:
        with _HOOK_TIMING_LOCK:
            _HOOK_TIMING_STATS.setdefault(label, []).append(work_ms)
    except Exception as exc:
        print(f"[HOOK_TIMING] failed to record timing: {exc}", flush=True)

def _record_hook_tail_timing(label: str, tail_ms: float) -> None:
    if not ENABLE_HOOK_TIMING:
        return
    if HOOK_TIMING_RANK0_ONLY and dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    try:
        with _HOOK_TIMING_LOCK:
            _HOOK_TAIL_STATS.setdefault(label, []).append(tail_ms)
    except Exception as exc:
        print(f"[HOOK_TIMING] failed to record tail timing: {exc}", flush=True)

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
        
# helper to summarize epoch time to not print each hook time each time
def summarize_hook_timing(epoch: int) -> None:
    if not ENABLE_HOOK_TIMING:
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    acquired = _HOOK_TIMING_LOCK.acquire(blocking=False)
    if not acquired:
        print(
            f"[HOOK_TIMING_SUMMARY] epoch={epoch} skipped: timing lock busy",
            flush=True,
        )
        return

    try:
        work_items = [
            (label, list(values))
            for label, values in _HOOK_TIMING_STATS.items()
            if values
        ]
        tail_items = [
            (label, list(values))
            for label, values in _HOOK_TAIL_STATS.items()
            if values
        ]
        _HOOK_TIMING_STATS.clear()
        _HOOK_TAIL_STATS.clear()
    finally:
        _HOOK_TIMING_LOCK.release()

    if not work_items and not tail_items:
        print(f"[HOOK_TIMING_SUMMARY] epoch={epoch} no hook timing samples", flush=True)
        return

    print(f"\n[HOOK_TIMING_SUMMARY] epoch={epoch}", flush=True)

    for label, values in sorted(work_items):
        work_total = sum(values)
        work_mean = work_total / len(values)

        print(
            f"[HOOK_WORK_SUMMARY] label={label} "
            f"calls={len(values)} "
            f"work_total_ms={work_total:.3f} "
            f"work_mean_ms={work_mean:.3f} "
            f"work_min_ms={min(values):.3f} "
            f"work_max_ms={max(values):.3f}",
            flush=True,
        )

    for label, values in sorted(tail_items):
        tail_total = sum(values)
        tail_mean = tail_total / len(values)

        print(
            f"[HOOK_TAIL_SUMMARY] label={label} "
            f"steps={len(values)} "
            f"tail_total_ms={tail_total:.3f} "
            f"tail_mean_ms={tail_mean:.3f} "
            f"tail_min_ms={min(values):.3f} "
            f"tail_max_ms={max(values):.3f}",
            flush=True,
        )

def _profiled_hook(label: str, hook_fn, state, bucket: dist.GradBucket):
    global _bucket_fire_seq, _known_num_buckets, _compute_range_open
    tensor = bucket.buffer()
    numel = tensor.numel()
    epoch, batch = _current_epoch, _current_batch
    rank = dist.get_rank()
    hook_dispatch_time = time.perf_counter()

    b_idx = _bucket_fire_seq
    _bucket_fire_seq += 1
    is_last_bucket = False
    bucket_is_last = getattr(bucket, "is_last", None)
    if callable(bucket_is_last):
        is_last_bucket = bool(bucket_is_last())

    if is_last_bucket:
        _known_num_buckets = b_idx + 1
    elif _known_num_buckets is not None:
        is_last_bucket = b_idx == _known_num_buckets - 1

    if _compute_range_open:
        _nvtx_range_pop()
        _compute_range_open = False

    _nvtx_range_push(
        f"hook_called:{label} B={b_idx} epoch={epoch} batch={batch} numel={numel}"
    )
    try:
        fut = hook_fn(state, bucket)
    finally:
        _nvtx_range_pop()

    if ENABLE_NVTX_PROFILING and not is_last_bucket:
        _nvtx_range_push(f"compute:B={b_idx + 1}")
        _compute_range_open = True

    def _on_done(f):
        _nvtx_range_push(
            f"comm_done:{label} B={b_idx} epoch={epoch} batch={batch}"
        )
        result = f.value()
        _nvtx_range_pop()
        
        if is_last_bucket:
            tail_ms = (time.perf_counter() - hook_dispatch_time) * 1000.0
            _record_hook_tail_timing(label, tail_ms)
            
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

# ── Background worker for async ring hook ─────────────────────────────────────
# One worker preserves the same P2P/tag order as the synchronous ring hook while
# allowing DDP's main/autograd thread to continue after returning the Future.
_COMM_HOOK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="ddp-comm-hook",
)

def _submit_async_hook_work(
    label: str,
    tensor: torch.Tensor,
    work_fn,
) -> torch.futures.Future[torch.Tensor]:
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    device_index = tensor.device.index if tensor.is_cuda else None

    def _worker() -> None:
        try:
            if device_index is not None:
                torch.cuda.set_device(device_index)

            work_start = time.perf_counter()

            _nvtx_range_push(f"comm_work:{label}")
            try:
                work_fn()
                _cuda_sync(tensor)
            finally:
                _nvtx_range_pop()

            work_ms = (time.perf_counter() - work_start) * 1000.0
            _record_hook_work_timing(label, work_ms)

            fut.set_result(tensor)

        except BaseException as exc:
            fut.set_exception(exc)

    _COMM_HOOK_EXECUTOR.submit(_worker)
    return fut
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

    label = f"{backend}:default"

    tensor.div_(dist.get_world_size())

    _nvtx_range_push(
        f"allreduce_launched backend={backend} rank={rank} "
        f"bucket={bucket.index()} numel={numel}"
    )
    _nvtx_range_pop()

    range_id = _nvtx_process_range_start(
        f"async_allreduce:{backend} rank={rank} bucket={bucket.index()} numel={numel}"
    )

    work_start = time.perf_counter()
    fut = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True).get_future()

    def _finish(f):
        work_ms = (time.perf_counter() - work_start) * 1000.0
        _record_hook_work_timing(label, work_ms)

        _nvtx_process_range_end(range_id)
        return f.value()[0]

    return fut.then(_finish)

# Hook alias: baseline PyTorch-style allreduce using dist.all_reduce.
nccl_default_allreduce_hook = _default_allreduce_hook
mpi_default_allreduce_hook = _default_allreduce_hook


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
    def _work() -> None:
        _ring_allreduce_sum(
            tensor,
            bstate,
            log=do_full_log,
            call_n=call_n,
        )

    return _submit_async_hook_work(f"{dist.get_backend()}:ring", tensor, _work)


# Hook aliases: plain ring allreduce with no compression.
nccl_ring_allreduce_hook = _ring_allreduce_hook
mpi_ring_allreduce_hook  = _ring_allreduce_hook

#ring async hook try 1 

# def _ring_allreduce_async_hook(
#     state: RingAllreduceState,
#     bucket: dist.GradBucket,
# ) -> torch.futures.Future[torch.Tensor]:
#     """
#     Async wrapper for the plain ring allreduce.
#
#     The hook immediately returns an unresolved Future. A single background
#     worker thread runs the existing ring implementation and resolves the Future
#     when communication is complete.
#     """
#     tensor = bucket.buffer()
#     world_size = dist.get_world_size()
#     rank = dist.get_rank()
#     numel = tensor.numel()
#     bucket_index = bucket.index()
#
#     bstate = state.get_or_init(bucket_index, tensor, world_size)
#     bstate.call_count += 1
#     call_n = bstate.call_count
#
#     do_full_log = (FULL_LOG_CALLS > 0 and call_n <= FULL_LOG_CALLS)
#
#     fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
#     device_index = tensor.device.index if tensor.is_cuda else None
#
#     if do_full_log:
#         print(
#             f"\n[RING_ASYNC | R{rank}|B{bstate.bucket_id}|C{call_n}] "
#             f"numel={numel} submit_thread={threading.get_ident()} "
#             f"global_tag_before={_tag_counter}",
#             flush=True,
#         )
#
#     def _worker() -> None:
#         try:
#             if device_index is not None:
#                 torch.cuda.set_device(device_index)
#
#             if do_full_log:
#                 print(
#                     f"[RING_ASYNC | R{rank}|B{bstate.bucket_id}|C{call_n}] "
#                     f"worker_thread={threading.get_ident()}",
#                     flush=True,
#                 )
#
#             work_start = time.perf_counter()
#
#             _ring_allreduce_sum(
#                 tensor,
#                 bstate,
#                 log=do_full_log,
#                 call_n=call_n,
#             )
#
#             _cuda_sync(tensor)
#
#             work_ms = (time.perf_counter() - work_start) * 1000.0
#             _record_hook_work_timing("mpi:ring_async", work_ms)
#
#             fut.set_result(tensor)
#
#         except BaseException as exc:
#             fut.set_exception(exc)
#
#     _RING_ASYNC_EXECUTOR.submit(_worker)
#     return fut
#
# #aliases
# mpi_ring_async_allreduce_hook = _ring_allreduce_async_hook


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
    def _work() -> None:
        _recursive_doubling_allreduce_sum(
            tensor,
            bstate,
            log=do_full_log,
            call_n=call_n,
        )

    return _submit_async_hook_work(
        f"{dist.get_backend()}:recursive_doubling",
        tensor,
        _work,
    )

# Hook aliases: plain recursive-doubling allreduce with no compression.
nccl_recursive_doubling_allreduce_hook = _recursive_doubling_allreduce_hook
mpi_recursive_doubling_allreduce_hook  = _recursive_doubling_allreduce_hook


# ═══════════════════════════════════════════════════════════════════════════════
# ZFP-COMPRESSED RING ALLREDUCE — naive point-to-point baseline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ZfpRingBucketState:
    """Persistent buffers for ZFP ring variants, including online compression."""
    flat:            Optional[torch.Tensor] = None
    cnts:            List[int]              = field(default_factory=list)
    displs:          List[int]              = field(default_factory=list)
    send_comp:       List[torch.Tensor]     = field(default_factory=list)
    recv_comp:       List[torch.Tensor]     = field(default_factory=list)
    recv_decomp:     List[torch.Tensor]     = field(default_factory=list)
    send_sizes:      List[torch.Tensor]     = field(default_factory=list)
    recv_sizes:      List[torch.Tensor]     = field(default_factory=list)

    # Online hook additions.
    streams:         List[torch.cuda.Stream] = field(default_factory=list)
    comp_bytes:      List[int]               = field(default_factory=list)
    phase2_recv_comp: List[torch.Tensor]     = field(default_factory=list)
    phase2_recv_bytes: List[int]             = field(default_factory=list)

    bucket_id:       int                    = 0
    call_count:      int                    = 0
    ready:           bool                   = False
    rate:            float                  = 16.0

    def initialize(self, tensor: torch.Tensor, world_size: int, bucket_id: int, rate: float) -> None:
        _require_zfp_cuda()
        self.rate = rate
        cfg = ZfpCompressionConfig(rate=rate)
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
        self.streams = []
        self.comp_bytes = []
        self.phase2_recv_comp = []
        self.phase2_recv_bytes = []

        max_chunk_bytes = 0
        for cnt in self.cnts:
            recv_tensor = torch.empty(cnt, dtype=tensor.dtype, device=tensor.device)
            self.recv_decomp.append(recv_tensor)

            probe = torch.empty(cnt, dtype=tensor.dtype, device=tensor.device)
            max_bytes = _zfp_max_output_bytes(probe, cfg.rate)
            max_chunk_bytes = max(max_chunk_bytes, max_bytes)

            self.send_comp.append(torch.empty(max_bytes, dtype=torch.uint8, device=tensor.device))
            self.recv_comp.append(torch.empty(max_bytes, dtype=torch.uint8, device=tensor.device))
            self.send_sizes.append(torch.zeros(1, dtype=torch.int64, device=tensor.device))
            self.recv_sizes.append(torch.zeros(1, dtype=torch.int64, device=tensor.device))
            self.comp_bytes.append(max_bytes)

        self.streams = [
            torch.cuda.Stream(device=tensor.device)
            for _ in range(world_size)
        ]

        # Phase 2 forwards previously received compressed payloads directly.
        # These per-step buffers prevent overwriting a buffer still used by an
        # outstanding MPI_Isend.
        self.phase2_recv_comp = [
            torch.empty(max_chunk_bytes, dtype=torch.uint8, device=tensor.device)
            for _ in range(max(world_size - 1, 1))
        ]
        self.phase2_recv_bytes = [0 for _ in range(max(world_size - 1, 1))]

        self.bucket_id = bucket_id
        self.ready = True

@dataclass
class ZfpRingAllreduceState:
    rate: float = 16.0
    buckets: Dict[int, ZfpRingBucketState] = field(default_factory=dict)

    def get_or_init(self, bucket_index: int,
                    tensor: torch.Tensor, world_size: int) -> ZfpRingBucketState:
        bs = self.buckets.get(bucket_index)
        if bs is None or bs.flat is None or bs.flat.numel() != tensor.numel():
            bs = ZfpRingBucketState()
            bs.initialize(tensor, world_size, bucket_index, self.rate)            
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
    cfg = ZfpCompressionConfig(rate=bstate.rate)    
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

    def _work() -> None:
        _ring_allreduce_zfp_sum(
            tensor,
            bstate,
            log=do_full_log,
            call_n=call_n,
        )

    return _submit_async_hook_work(
        f"{dist.get_backend()}:ring_zfp_naive:rate{bstate.rate:g}",
        tensor,
        _work,
    )


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
    rate:       float                  = 16.0

    def initialize(self, tensor: torch.Tensor, bucket_id: int, rate: float) -> None:
        _require_zfp_cuda()
        self.rate = rate
        cfg = ZfpCompressionConfig(rate=rate)
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
    rate: float = 16.0
    buckets: Dict[int, ZfpRecursiveDoublingBucketState] = field(default_factory=dict)

    def get_or_init(self, bucket_index: int,
                    tensor: torch.Tensor) -> ZfpRecursiveDoublingBucketState:
        bs = self.buckets.get(bucket_index)
        if bs is None or bs.flat is None or bs.flat.numel() != tensor.numel():
            bs = ZfpRecursiveDoublingBucketState()
            bs.initialize(tensor, bucket_index, self.rate)
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
    cfg = ZfpCompressionConfig(rate=bstate.rate)    
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

    def _work() -> None:
        _recursive_doubling_zfp_sum(
            tensor,
            bstate,
            log=do_full_log,
            call_n=call_n,
        )

    return _submit_async_hook_work(
        f"{dist.get_backend()}:recursive_doubling_zfp_naive:rate{bstate.rate:g}",
        tensor,
        _work,
    )


def _ring_allreduce_zfp_online_coll_sum(
    tensor: torch.Tensor,
    bstate: ZfpRingBucketState,
    log: bool,
    call_n: int = 0,
) -> None:
    """
    Algorithm 1: Collective-level Online Compression Design for Ring MPI Allreduce.

    This implementation follows the paper's chunk-index formulas exactly.

    Phase 1:
      line 2: si = (rank - i + N) % N
      line 3: ri = (rank - i - 1 + N) % N

    Phase 2:
      line 21: si = (rank - i + 1 + N) % N
      line 22: ri = (rank - i + N) % N

    B is deterministic because the wrapper uses fixed-rate ZFP. Therefore B is
    precomputed per chunk in bstate.comp_bytes and no extra size exchange is
    needed for this online algorithm.
    """
    _require_zfp_cuda()
    cfg = ZfpCompressionConfig(rate=bstate.rate)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size == 1:
        return

    assert bstate.flat is not None
    assert bstate.streams
    assert bstate.comp_bytes

    left = (rank - 1 + world_size) % world_size
    right = (rank + 1) % world_size

    # Paper input S -> local DDP bucket tensor.
    # Paper output R -> bstate.flat.
    #
    # Figure 3: copy sendbuf to recvbuf using device-to-device cudaMemcpy.
    bstate.flat.copy_(tensor.flatten())

    chunks = [
        bstate.flat[bstate.displs[i]: bstate.displs[i] + bstate.cnts[i]]
        for i in range(world_size)
    ]

    # The copy into R was enqueued on the current PyTorch stream. All non-default
    # ZFP streams must wait for it before reading chunks.
    current_stream = torch.cuda.current_stream(device=tensor.device)
    for s in bstate.streams:
        s.wait_stream(current_stream)

    send_reqs = []

    # =========================================================================
    # Lines 1-19: Phase 1, reduce-scatter with collective-level online compression
    # =========================================================================
    for i in range(world_size - 1):
        # Lines 2-3.
        si = (rank - i + world_size) % world_size
        ri = (rank - i - 1 + world_size) % world_size

        tag = _next_tag()
        send_B = bstate.comp_bytes[si]
        recv_B = bstate.comp_bytes[ri]

        # Lines 4-5.
        # First iteration compresses the initially available R_si.
        if i == 0:
            with torch.cuda.stream(bstate.streams[si]):
                used = _zfp_compress_into_current_stream(
                    chunks[si],
                    bstate.send_comp[si],
                    cfg.rate,
                )
            if log and used != send_B:
                print(f"[R{rank}|B{bstate.bucket_id}|C{call_n}] "
                      f"P1 line5 i={i} si={si} used={used} expected={send_B}",
                      flush=True)

        # Line 6.
        # Post receive before waiting for compression, giving compression and
        # MPI_Irecv the opportunity to overlap.
        recv_req = dist.irecv(
            bstate.recv_comp[ri][:recv_B],
            src=left,
            tag=tag,
        )

        if i == 0:
            # Lines 7-8.
            # Wait only for the stream that produced R_tmp_si.
            bstate.streams[si].synchronize()
        else:
            # Lines 9-12.
            #
            # In the paper formula, current si equals the ri from the previous
            # iteration. That chunk was produced by the previous
            # decompress+reduction stream, so wait for it, then compress it.
            bstate.streams[si].synchronize()

            # Line 11.
            with torch.cuda.stream(bstate.streams[si]):
                used = _zfp_compress_into_current_stream(
                    chunks[si],
                    bstate.send_comp[si],
                    cfg.rate,
                )
            if log and used != send_B:
                print(f"[R{rank}|B{bstate.bucket_id}|C{call_n}] "
                      f"P1 line11 i={i} si={si} used={used} expected={send_B}",
                      flush=True)

            # Line 12.
            bstate.streams[si].synchronize()

        # Line 13.
        send_req = dist.isend(
            bstate.send_comp[si][:send_B],
            dst=right,
            tag=tag,
        )
        send_reqs.append(send_req)

        # Line 14.
        recv_req.wait()

        # Lines 15-16.
        # Decompression and reduction are launched on the same stream. CUDA
        # stream ordering makes the reduction wait for decompression without a
        # host-side synchronization.
        with torch.cuda.stream(bstate.streams[ri]):
            _zfp_decompress_into_current_stream(
                bstate.recv_comp[ri],
                recv_B,
                bstate.recv_decomp[ri],
                cfg.rate,
            )
            chunks[ri].add_(bstate.recv_decomp[ri])

        if log:
            print(f"[R{rank}|B{bstate.bucket_id}|C{call_n}] "
                  f"P1 i={i} si={si} ri={ri} tag={tag} Bsend={send_B} Brecv={recv_B}",
                  flush=True)

    # Lines 17-18.
    # Wait for the last reduction stream. Under the paper's formula, this is
    # the chunk index (rank + 1) % N.
    ri = (rank + 1) % world_size
    bstate.streams[ri].synchronize()

    # Line 19.
    for req in send_reqs:
        req.wait()

    send_reqs = []

    # =========================================================================
    # Lines 20-33: Phase 2, allgather with compressed forwarding
    # =========================================================================
    for i in range(world_size - 1):
        # Lines 21-22.
        si = (rank - i + 1 + world_size) % world_size
        ri = (rank - i + world_size) % world_size

        tag = _next_tag()
        send_B = bstate.comp_bytes[si]
        recv_B = bstate.comp_bytes[ri]

        # Lines 23-24.
        # Only the first phase-2 send compresses the local fully reduced chunk.
        # Later sends forward a previously received compressed payload directly.
        if i == 0:
            with torch.cuda.stream(bstate.streams[si]):
                used = _zfp_compress_into_current_stream(
                    chunks[si],
                    bstate.send_comp[si],
                    cfg.rate,
                )
            if log and used != send_B:
                print(f"[R{rank}|B{bstate.bucket_id}|C{call_n}] "
                      f"P2 line24 i={i} si={si} used={used} expected={send_B}",
                      flush=True)

        # Line 25.
        # Receive into a per-step compressed buffer so it can be forwarded later
        # without being overwritten by another receive.
        recv_buf = bstate.phase2_recv_comp[i][:recv_B]
        recv_req = dist.irecv(
            recv_buf,
            src=left,
            tag=tag,
        )
        bstate.phase2_recv_bytes[i] = recv_B

        if i == 0:
            # Lines 26-27.
            bstate.streams[si].synchronize()
            send_buf = bstate.send_comp[si]
        else:
            # Line 28, paper text:
            # MPI_Isend directly sends out the previously received compressed
            # chunk.  Previous step's receive buffer is i-1.
            send_buf = bstate.phase2_recv_comp[i - 1]

        # Line 28.
        send_req = dist.isend(
            send_buf[:send_B],
            dst=right,
            tag=tag,
        )
        send_reqs.append(send_req)

        # Line 29.
        recv_req.wait()

        # Line 30.
        # No reduction in allgather. Decompress received compressed data into R_ri.
        with torch.cuda.stream(bstate.streams[ri]):
            _zfp_decompress_into_current_stream(
                bstate.phase2_recv_comp[i],
                recv_B,
                chunks[ri],
                cfg.rate,
            )

        if log:
            print(f"[R{rank}|B{bstate.bucket_id}|C{call_n}] "
                  f"P2 i={i} si={si} ri={ri} tag={tag} Bsend={send_B} Brecv={recv_B}",
                  flush=True)

    # Lines 31-32.
    for i in range(world_size - 1):
        ri = (rank - i + world_size) % world_size
        bstate.streams[ri].synchronize()

    # Line 33.
    for req in send_reqs:
        req.wait()

    # Final DDP result. Synchronize all non-default streams before copying the
    # completed R buffer back to DDP's bucket tensor.
    for s in bstate.streams:
        s.synchronize()

    tensor.copy_(bstate.flat.view_as(tensor))
    tensor.div_(world_size)
    _cuda_sync(tensor)
    
def _ring_allreduce_zfp_online_coll_hook(
    state: ZfpRingAllreduceState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """DDP hook for Algorithm 1: ring collective-level online ZFP compression."""
    tensor = bucket.buffer()
    rank = dist.get_rank()
    bucket_index = bucket.index()
    world_size = dist.get_world_size()
    numel = tensor.numel()

    bstate = state.get_or_init(bucket_index, tensor, world_size)
    bstate.call_count += 1
    call_n = bstate.call_count

    do_full_log = (FULL_LOG_CALLS > 0 and call_n <= FULL_LOG_CALLS)

    if do_full_log:
        print(f"\n[RING_ZFP_ONLINE_COLL | R{rank}|B{bstate.bucket_id}|C{call_n}] "
              f"numel={numel} global_tag_before={_tag_counter}",
              flush=True)

    def _work() -> None:
        _ring_allreduce_zfp_online_coll_sum(
            tensor,
            bstate,
            log=do_full_log,
            call_n=call_n,
        )

    return _submit_async_hook_work(
        f"{dist.get_backend()}:ring_zfp_online_coll:rate{bstate.rate:g}",
        tensor,
        _work,
    )


@dataclass
class ZfpRecursiveDoublingOnlineBucketState:
    """Persistent buffers for recursive-doubling collective-level online ZFP."""
    flat: Optional[torch.Tensor] = None
    recv_decomp: List[torch.Tensor] = field(default_factory=list)
    send_comp: List[torch.Tensor] = field(default_factory=list)
    recv_comp: List[torch.Tensor] = field(default_factory=list)
    streams: List[torch.cuda.Stream] = field(default_factory=list)
    comp_bytes: int = 0
    bucket_id: int = 0
    call_count: int = 0
    rate: float = 16.0

    def initialize(self, tensor: torch.Tensor, world_size: int, bucket_id: int, rate: float) -> None:
        _require_zfp_cuda()
        self.rate = rate
        cfg = ZfpCompressionConfig(rate=rate)
        numel = tensor.numel()
        self.flat = torch.empty(numel, dtype=tensor.dtype, device=tensor.device)

        pof2 = 1
        while pof2 * 2 <= world_size:
            pof2 *= 2

        num_steps = int(math.log2(pof2)) if pof2 > 1 else 0
        # +2 gives room for peel/finalize plus recursive-doubling steps.
        num_slots = max(num_steps + 2, 1)

        self.comp_bytes = int(_zfp_max_output_bytes(self.flat, cfg.rate))

        self.recv_decomp = [
            torch.empty_like(self.flat)
            for _ in range(num_slots)
        ]
        self.send_comp = [
            torch.empty(self.comp_bytes, dtype=torch.uint8, device=tensor.device)
            for _ in range(num_slots)
        ]
        self.recv_comp = [
            torch.empty(self.comp_bytes, dtype=torch.uint8, device=tensor.device)
            for _ in range(num_slots)
        ]
        self.streams = [
            torch.cuda.Stream(device=tensor.device)
            for _ in range(world_size)
        ]

        self.bucket_id = bucket_id


@dataclass
class ZfpRecursiveDoublingOnlineAllreduceState:
    rate: float = 16.0
    buckets: Dict[int, ZfpRecursiveDoublingOnlineBucketState] = field(default_factory=dict)

    def get_or_init(
        self,
        bucket_index: int,
        tensor: torch.Tensor,
        world_size: int,
    ) -> ZfpRecursiveDoublingOnlineBucketState:
        bs = self.buckets.get(bucket_index)
        if bs is None or bs.flat is None or bs.flat.numel() != tensor.numel():
            bs = ZfpRecursiveDoublingOnlineBucketState()
            bs.initialize(tensor, world_size, bucket_index, self.rate)
            self.buckets[bucket_index] = bs
        return bs
    
def _recursive_doubling_zfp_online_coll_sum(
    tensor: torch.Tensor,
    bstate: ZfpRecursiveDoublingOnlineBucketState,
    log: bool,
    call_n: int = 0,
) -> None:
    """
    Algorithm 2: Collective-level Online Compression for Recursive-Doubling MPI Allreduce.

    This is the MPI-style version. It uses separate dist.irecv/dist.isend calls
    so the receive can be posted while ZFP compression runs on a non-default
    CUDA stream.

    The reduction kernel is launched on the same stream as decompression.
    Sends are waited after the recursive-doubling loop, matching line 39.
    """
    _require_zfp_cuda()
    cfg = ZfpCompressionConfig(rate=bstate.rate)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size == 1:
        return

    assert bstate.flat is not None
    assert bstate.streams
    assert bstate.send_comp
    assert bstate.recv_comp
    assert bstate.recv_decomp

    work = bstate.flat
    work.copy_(tensor.flatten())

    current_stream = torch.cuda.current_stream(device=tensor.device)
    for s in bstate.streams:
        s.wait_stream(current_stream)

    B = bstate.comp_bytes
    send_reqs = []

    # Lines 1-2.
    pof2 = 1
    while pof2 * 2 <= world_size:
        pof2 *= 2
    rem = world_size - pof2

    # Line 3.
    newrank = 0

    # =========================================================================
    # Lines 4-13: non-power-of-two peel.
    # =========================================================================
    peel_tag = _next_tag()

    if rank < 2 * rem:
        if rank % 2 == 0:
            # Lines 5-7:
            # Even rank compresses and sends to rank + 1, then drops out.
            dst = rank + 1
            stream = bstate.streams[dst]

            with torch.cuda.stream(stream):
                used = _zfp_compress_into_current_stream(
                    work,
                    bstate.send_comp[0],
                    cfg.rate,
                )

            if log and used != B:
                print(f"[R{rank}|B{bstate.bucket_id}|C{call_n}] "
                      f"peel send used={used} expected={B}",
                      flush=True)

            stream.synchronize()
            send_req = dist.isend(
                bstate.send_comp[0][:B],
                dst=dst,
                tag=peel_tag,
            )
            send_reqs.append(send_req)
            newrank = -1

        else:
            # Lines 8-11:
            # Odd rank receives compressed data from rank - 1, decompresses,
            # reduces with its own data, and enters the pof2 communicator.
            src = rank - 1
            stream = bstate.streams[src]

            recv_req = dist.irecv(
                bstate.recv_comp[0][:B],
                src=src,
                tag=peel_tag,
            )
            recv_req.wait()

            with torch.cuda.stream(stream):
                _zfp_decompress_into_current_stream(
                    bstate.recv_comp[0],
                    B,
                    bstate.recv_decomp[0],
                    cfg.rate,
                )
                work.add_(bstate.recv_decomp[0])

            stream.synchronize()
            newrank = rank // 2
    else:
        # Lines 12-13.
        newrank = rank - rem

    # =========================================================================
    # Lines 14-39: recursive-doubling over the pof2 active ranks.
    # =========================================================================
    last_reduce_stream = None

    if newrank != -1:
        # Line 15.
        mask = 0x1
        step = 0

        # Line 16.
        while mask < pof2:
            # Lines 17-18.
            newdst = newrank ^ mask
            dst = (newdst * 2 + 1) if newdst < rem else (newdst + rem)

            tag = _next_tag()
            slot = step + 1

            # Lines 21-23:
            # Choose the stream for the following reduction. If there is a next
            # recursive-doubling step, reduce on that next step's stream so the
            # next compression is naturally ordered after reduction. On the last
            # step, use the current peer's stream.
            mask2 = mask << 1
            if mask2 < pof2:
                newdst2 = newrank ^ mask2
                dst2 = (newdst2 * 2 + 1) if newdst2 < rem else (newdst2 + rem)
            else:
                dst2 = dst

            send_stream = bstate.streams[dst]
            reduce_stream = bstate.streams[dst2]
            last_reduce_stream = reduce_stream

            if mask == 0x1:
                # Lines 19-20:
                # First active step compresses the current buffer S/R.
                with torch.cuda.stream(send_stream):
                    used = _zfp_compress_into_current_stream(
                        work,
                        bstate.send_comp[slot],
                        cfg.rate,
                    )

                # Line 24:
                # Post receive while compression is in flight.
                recv_req = dist.irecv(
                    bstate.recv_comp[slot][:B],
                    src=dst,
                    tag=tag,
                )

                # Lines 29-30.
                send_stream.synchronize()

            else:
                # Line 24:
                # Post receive first so receive can overlap with compression.
                recv_req = dist.irecv(
                    bstate.recv_comp[slot][:B],
                    src=dst,
                    tag=tag,
                )

                # Lines 25-28:
                # Wait for the stream that produced the current R, then compress.
                send_stream.synchronize()
                with torch.cuda.stream(send_stream):
                    used = _zfp_compress_into_current_stream(
                        work,
                        bstate.send_comp[slot],
                        cfg.rate,
                    )
                send_stream.synchronize()

            if log and used != B:
                print(f"[R{rank}|B{bstate.bucket_id}|C{call_n}] "
                      f"main step={step} mask={mask} dst={dst} used={used} expected={B}",
                      flush=True)

            # Line 31.
            send_req = dist.isend(
                bstate.send_comp[slot][:B],
                dst=dst,
                tag=tag,
            )
            send_reqs.append(send_req)

            # Line 32.
            recv_req.wait()

            # Lines 33-34:
            # Decompress and reduce on the same stream.
            with torch.cuda.stream(reduce_stream):
                _zfp_decompress_into_current_stream(
                    bstate.recv_comp[slot],
                    B,
                    bstate.recv_decomp[slot],
                    cfg.rate,
                )
                work.add_(bstate.recv_decomp[slot])

            if log:
                print(f"[R{rank}|B{bstate.bucket_id}|C{call_n}] "
                      f"main step={step} mask={mask} dst={dst} dst2={dst2} tag={tag}",
                      flush=True)

            # Line 35.
            mask <<= 1
            step += 1

        # Lines 36-38:
        # Wait for the last reduction kernel.
        if last_reduce_stream is not None:
            last_reduce_stream.synchronize()

        # Line 39.
        for req in send_reqs:
            req.wait()
        send_reqs = []

    else:
        # Inactive peeled even ranks still consume the same main-loop tags so
        # future hook invocations remain tag-aligned across all ranks.
        mask = 0x1
        while mask < pof2:
            _next_tag()
            mask <<= 1

        for req in send_reqs:
            req.wait()
        send_reqs = []

    # =========================================================================
    # Lines 40-45: non-power-of-two finalize.
    # =========================================================================
    final_tag = _next_tag()

    if rank < 2 * rem:
        if rank % 2 == 0:
            # Lines 40-43:
            # Even peeled rank receives the final compressed result from rank+1.
            src = rank + 1
            stream = bstate.streams[src]

            recv_req = dist.irecv(
                bstate.recv_comp[0][:B],
                src=src,
                tag=final_tag,
            )
            recv_req.wait()

            with torch.cuda.stream(stream):
                _zfp_decompress_into_current_stream(
                    bstate.recv_comp[0],
                    B,
                    work,
                    cfg.rate,
                )

            stream.synchronize()

        else:
            # Lines 44-45:
            # Odd active rank sends compressed final result back to rank-1.
            dst = rank - 1
            stream = bstate.streams[dst]

            stream.synchronize()
            with torch.cuda.stream(stream):
                used = _zfp_compress_into_current_stream(
                    work,
                    bstate.send_comp[0],
                    cfg.rate,
                )

            if log and used != B:
                print(f"[R{rank}|B{bstate.bucket_id}|C{call_n}] "
                      f"final send used={used} expected={B}",
                      flush=True)

            stream.synchronize()
            send_req = dist.isend(
                bstate.send_comp[0][:B],
                dst=dst,
                tag=final_tag,
            )
            send_req.wait()

    for s in bstate.streams:
        s.synchronize()

    tensor.copy_(work.view_as(tensor))
    tensor.div_(world_size)
    _cuda_sync(tensor)
    
def _recursive_doubling_zfp_online_coll_hook(
    state: ZfpRecursiveDoublingOnlineAllreduceState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """DDP hook for Algorithm 2: recursive-doubling collective-level online ZFP."""
    tensor = bucket.buffer()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    bucket_index = bucket.index()
    numel = tensor.numel()

    bstate = state.get_or_init(bucket_index, tensor, world_size)
    bstate.call_count += 1
    call_n = bstate.call_count

    do_full_log = (FULL_LOG_CALLS > 0 and call_n <= FULL_LOG_CALLS)

    if do_full_log:
        print(f"\n[RECURSIVE_DOUBLING_ZFP_ONLINE_COLL | R{rank}|B{bstate.bucket_id}|C{call_n}] "
              f"numel={numel} global_tag_before={_tag_counter}",
              flush=True)

    def _work() -> None:
        _recursive_doubling_zfp_online_coll_sum(
            tensor,
            bstate,
            log=do_full_log,
            call_n=call_n,
        )

    return _submit_async_hook_work(
        f"{dist.get_backend()}:recursive_doubling_zfp_online_coll:rate{bstate.rate:g}",
        tensor,
        _work,
    )
   
# Explicit aliases with "naive" in the name to mirror the paper terminology.
nccl_ring_zfp_naive_allreduce_hook = _ring_allreduce_zfp_hook
mpi_ring_zfp_naive_allreduce_hook = _ring_allreduce_zfp_hook
nccl_recursive_doubling_zfp_naive_allreduce_hook = _recursive_doubling_zfp_hook
mpi_recursive_doubling_zfp_naive_allreduce_hook = _recursive_doubling_zfp_hook

mpi_ring_zfp_online_coll_allreduce_hook = _ring_allreduce_zfp_online_coll_hook
mpi_recursive_doubling_zfp_online_coll_allreduce_hook = _recursive_doubling_zfp_online_coll_hook

# ═══════════════════════════════════════════════════════════════════════════════
# HOOK FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

_HOOK_REGISTRY = {
    ("nccl", "default"):            nccl_default_allreduce_hook,
    ("nccl", "ring"):               nccl_ring_allreduce_hook,
    ("nccl", "ring_zfp_naive"):     nccl_ring_zfp_naive_allreduce_hook,
    ("nccl", "recursive_doubling"): nccl_recursive_doubling_allreduce_hook,
    ("nccl", "recursive_doubling_zfp_naive"): nccl_recursive_doubling_zfp_naive_allreduce_hook,
    ("mpi",  "default"):            mpi_default_allreduce_hook,
    ("mpi",  "ring"):               mpi_ring_allreduce_hook,
    ("mpi",  "ring_zfp_naive"):     mpi_ring_zfp_naive_allreduce_hook,
    ("mpi",  "recursive_doubling"): mpi_recursive_doubling_allreduce_hook,
    ("mpi",  "recursive_doubling_zfp_naive"): mpi_recursive_doubling_zfp_naive_allreduce_hook,
    ("mpi",  "ring_zfp_online_coll"): mpi_ring_zfp_online_coll_allreduce_hook,
    ("mpi", "recursive_doubling_zfp_online_coll"): mpi_recursive_doubling_zfp_online_coll_allreduce_hook,
}


def get_comm_hook(
    backend: str,
    algorithm: Optional[str] = None,
    zfp_rate: float = 16.0,
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
    elif algorithm in ("ring_zfp_naive", "ring_zfp_online_coll"):
        state = ZfpRingAllreduceState(rate=zfp_rate)
    elif algorithm == "recursive_doubling":
        state = RecursiveDoublingAllreduceState()
    elif algorithm == "recursive_doubling_zfp_naive":
        state = ZfpRecursiveDoublingAllreduceState(rate=zfp_rate)
    elif algorithm == "recursive_doubling_zfp_online_coll":
        state = ZfpRecursiveDoublingOnlineAllreduceState(rate=zfp_rate)
    else:
        state = None

    profile_label = f"{backend}:{algorithm}"
    if "zfp" in algorithm:
        profile_label = f"{profile_label}:rate{zfp_rate:g}"

    def hook(state_obj, bucket):
        return _profiled_hook(profile_label, base_hook, state_obj, bucket)

    return hook, state


def list_available_hooks() -> list:
    return list(_HOOK_REGISTRY.keys())
