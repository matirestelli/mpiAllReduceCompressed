"""
Communication Strategy & AllReduce Hooks

Implements communication hooks for both NCCL and MPI backends.
Supports multiple AllReduce algorithms (default, ring, recursive doubling).

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURAL DECISION: torch.distributed vs raw MPI/NCCL
═══════════════════════════════════════════════════════════════════════════════

WHY WE USE torch.distributed PRIMITIVES (dist.send / dist.recv / dist.all_reduce)
INSTEAD OF raw mpi4py or raw NCCL calls:

1. THE HOOK CONTRACT: DDP's register_comm_hook gives us a GradBucket whose
   .buffer() is a CUDA tensor. The hook must return a Future[torch.Tensor].
   Everything lives in PyTorch's tensor world.

2. torch.distributed IS ALREADY the thin wrapper you want:
   - When backend="mpi", dist.send/recv calls MPI_Send/MPI_Recv under the hood
     (through ProcessGroupMPI). With CUDA-aware MPI, these pass GPU pointers
     directly — no CPU staging.
   - When backend="nccl", dist.send/recv calls ncclSend/ncclRecv under the hood
     (through ProcessGroupNCCL).
   So dist.send(tensor, dst) IS the raw MPI_Send or ncclSend — just type-safe.

3. WHY NOT raw mpi4py?
   - mpi4py works with numpy arrays or buffer-protocol objects.
   - To use it with CUDA tensors you'd need: tensor -> dlpack -> mpi4py buffer,
     or cupy interop. This is fragile, adds overhead, and bypasses PyTorch's
     stream synchronization.
   - You'd also lose compatibility: the same hook couldn't work for both
     backends.

4. WHY NOT raw NCCL C API (via ctypes/pybind)?
   - You'd need to manage ncclComm_t handles, CUDA streams, and synchronization
     manually. PyTorch's ProcessGroupNCCL already does this correctly.
   - No benefit: dist.send/recv on NCCL IS ncclSend/ncclRecv with proper stream
     management.

CONCLUSION: torch.distributed point-to-point ops are the right abstraction.
They dispatch to the raw backend (MPI or NCCL) while keeping tensor types,
device placement, and stream sync correct. Using them means the SAME hook
code works for both backends — which is exactly what a benchmarking tool wants.

The performance difference between dist.send() and a raw MPI_Send() is
negligible (one Python->C++ dispatch). The actual communication time dominates.

═══════════════════════════════════════════════════════════════════════════════
DDP HOOK CONTRACT — WHAT THE HOOK RECEIVES AND MUST RETURN
═══════════════════════════════════════════════════════════════════════════════

When NO hook is registered, DDP's C++ Reducer does:
  1. SUM-allreduce the raw gradients
  2. Divide by world_size internally (bucket.gradients.div_(div_factor_))

When a hook IS registered, DDP does NOT divide — the hook is fully responsible
for producing the AVERAGED gradient. From the PyTorch docs:
  "Grad bucket's tensors will not be predivided by world_size. User is
   responsible to divide by the world_size in case of operations like allreduce."

PyTorch's own reference allreduce_hook does:
  tensor.div_(world_size)  THEN  dist.all_reduce(tensor, SUM)

Our custom hooks must do the same: divide by world_size either before or
after the allreduce. We choose to divide AFTER the SUM (mathematically
equivalent: sum(g_i) / N == mean(g_i)).
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.distributed as dist
from typing import Callable, Optional
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT HOOKS (reference / baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def _default_allreduce_hook(
    state: Optional[object],
    bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Default AllReduce hook — works for both NCCL and MPI backends.

    Mirrors PyTorch's own allreduce_hook: divides by world_size then SUMs.
    Useful as a baseline for benchmarking custom hooks.
    """
    tensor = bucket.buffer()
    world_size = dist.get_world_size()

    # Divide BEFORE allreduce to avoid overflow (matches PyTorch's approach)
    tensor.div_(world_size)

    fut = dist.all_reduce(
        tensor, op=dist.ReduceOp.SUM, async_op=True
    ).get_future()

    def extract(fut):
        return fut.value()[0]

    return fut.then(extract)


# Expose with backend-specific names for the factory
nccl_default_allreduce_hook = _default_allreduce_hook
mpi_default_allreduce_hook = _default_allreduce_hook


# ═══════════════════════════════════════════════════════════════════════════════
# RING ALLREDUCE
# ═══════════════════════════════════════════════════════════════════════════════
#
# ALGORITHM
# ─────────
# Ring AllReduce operates on a logical ring: rank r's neighbors are
#   left  = (r - 1) % P
#   right = (r + 1) % P
#
# The gradient tensor is split into P equal-sized chunks.
#
# PHASE 1: REDUCE-SCATTER  (P - 1 steps)
# ────────────────────────────────────────
# Goal: after this phase, rank r "owns" one chunk that is the full
# reduction (sum) of that chunk across ALL ranks.
#
# Iteration s = 0, 1, ..., P-2:
#   send_idx = (r - s)     % P   ← chunk I send to my right neighbor
#   recv_idx = (r - s - 1) % P   ← chunk I receive from my left neighbor
#
#   1) rank r sends   chunks[send_idx] → right
#   2) rank r recvs   recv_buf          ← left
#   3) rank r does    chunks[recv_idx] += recv_buf   (accumulate)
#
# Why this works — trace for P=4, rank 0:
#   s=0: send chunk[0], recv into chunk[3], chunk[3] += recv  → chunk[3] has sum of rank0+rank3
#   s=1: send chunk[3], recv into chunk[2], chunk[2] += recv  → chunk[2] has sum of rank0+rank3+rank2
#   s=2: send chunk[2], recv into chunk[1], chunk[1] += recv  → chunk[1] has FULL sum of all 4 ranks
#
# After P-1 steps, rank r has the fully reduced chunk at index (r+1)%P.
#
# PHASE 2: ALLGATHER  (P - 1 steps)
# ──────────────────────────────────
# Goal: propagate each rank's fully-reduced chunk to all other ranks.
#
# Iteration s = 0, 1, ..., P-2:
#   send_idx = (r - s + 1) % P   ← chunk that's already correct, propagate it
#   recv_idx = (r - s)     % P   ← slot to overwrite with correct data
#
#   1) rank r sends   chunks[send_idx] → right
#   2) rank r recvs   recv_buf          ← left
#   3) rank r does    chunks[recv_idx] = recv_buf   (overwrite, not accumulate!)
#
# After P-1 steps, every rank has the identical fully-reduced tensor.
#
# COMPLEXITY
# ──────────
# Data per rank:  2 * (P-1)/P * N   (bandwidth-optimal)
# Latency:        2 * (P-1) messages
# Best for: large tensors where bandwidth dominates.
#
# ═══════════════════════════════════════════════════════════════════════════════

def _ring_allreduce_hook(
    state: Optional[object],
    bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Ring AllReduce communication hook.

    Works identically for MPI and NCCL backends — torch.distributed handles
    the dispatch. All operations stay on GPU.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor = bucket.buffer()

    if world_size == 1:
        fut = torch.futures.Future()
        fut.set_result(tensor)
        return fut

    # ── Flatten and pad to make evenly divisible ─────────────────────────
    numel = tensor.numel()
    padded_size = math.ceil(numel / world_size) * world_size

    if padded_size != numel:
        padded = torch.zeros(
            padded_size, dtype=tensor.dtype, device=tensor.device
        )
        padded[:numel] = tensor.reshape(-1)
    else:
        padded = tensor.reshape(-1)

    chunk_size = padded_size // world_size
    chunks = list(padded.chunk(world_size))  # list of views into padded

    # Ring neighbors
    left  = (rank - 1) % world_size
    right = (rank + 1) % world_size

    # Pre-allocate receive buffer (one chunk)
    recv_buf = torch.empty(
        chunk_size, dtype=tensor.dtype, device=tensor.device
    )

    # ── Phase 1: Reduce-Scatter ──────────────────────────────────────────
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        # .contiguous() ensures a contiguous send buffer for MPI/NCCL
        send_req = dist.isend(chunks[send_idx].contiguous(), dst=right)
        recv_req = dist.irecv(recv_buf, src=left)

        send_req.wait()
        recv_req.wait()

        # Accumulate received partial sum
        chunks[recv_idx].add_(recv_buf)

    # ── Phase 2: Allgather ───────────────────────────────────────────────
    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size

        send_req = dist.isend(chunks[send_idx].contiguous(), dst=right)
        recv_req = dist.irecv(recv_buf, src=left)

        send_req.wait()
        recv_req.wait()

        # Overwrite — this chunk is already fully reduced
        chunks[recv_idx].copy_(recv_buf)

    # ── Copy result back if we padded ────────────────────────────────────
    if padded_size != numel:
        tensor.copy_(padded[:numel].view_as(tensor))
    # else: tensor was modified in-place through the padded/chunks views

    # ── Divide by world_size to produce the AVERAGE ──────────────────────
    # The hook contract requires us to return averaged gradients.
    tensor.div_(world_size)

    fut = torch.futures.Future()
    fut.set_result(tensor)
    return fut


nccl_ring_allreduce_hook = _ring_allreduce_hook
mpi_ring_allreduce_hook  = _ring_allreduce_hook


# ═══════════════════════════════════════════════════════════════════════════════
# RECURSIVE DOUBLING
# ═══════════════════════════════════════════════════════════════════════════════
#
# ALGORITHM
# ─────────
# Operates in log2(P) rounds. In round k (k = 0, 1, ..., log2(P)-1):
#   distance = 2^k
#   partner  = rank XOR distance
#
#   1) rank sends its entire buffer to partner
#   2) rank receives partner's entire buffer
#   3) rank += received buffer
#
# After log2(P) rounds, every rank holds the full sum.
#
# WHY XOR?
# ────────
# XOR pairing ensures:
#   - Round 0 (dist=1): 0↔1, 2↔3, 4↔5, ...  (adjacent pairs)
#   - Round 1 (dist=2): 0↔2, 1↔3, 4↔6, ...  (stride-2 pairs)
#   - Round 2 (dist=4): 0↔4, 1↔5, 2↔6, ...  (stride-4 pairs)
# Each rank communicates with every other rank exactly once across all rounds.
# After round k, every rank has the partial sum of 2^(k+1) ranks.
#
# EXAMPLE (P=4):
# ──────────────
# Initial: rank0=A, rank1=B, rank2=C, rank3=D
#
# Round 0 (dist=1, XOR partner):
#   0↔1: both get A+B    2↔3: both get C+D
#
# Round 1 (dist=2, XOR partner):
#   0↔2: both get (A+B)+(C+D)    1↔3: both get (A+B)+(C+D)
#
# Result: all ranks have A+B+C+D. Done in 2 steps (log2(4)=2).
#
# NON-POWER-OF-2 HANDLING
# ───────────────────────
# If P is not a power of 2, we use a preliminary reduction:
#   pow2 = largest power of 2 ≤ P
#   remainder = P - pow2
#
#   Pre-step:  ranks [pow2 .. pow2+remainder-1] send their data to
#              ranks [0 .. remainder-1] and sit out.
#   Main:      recursive doubling on pow2 participating ranks.
#   Post-step: ranks [0 .. remainder-1] send results back to their partners.
#
# COMPLEXITY
# ──────────
# Data per rank:  N * log2(P)   (NOT bandwidth-optimal)
# Latency:        log2(P) messages  (latency-optimal)
# Best for: small tensors where latency dominates.
#
# ═══════════════════════════════════════════════════════════════════════════════

def _recursive_doubling_allreduce_hook(
    state: Optional[object],
    bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    Recursive Doubling AllReduce communication hook.

    Works for both MPI and NCCL backends. Handles non-power-of-2 world sizes
    with a preliminary fold step.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    tensor = bucket.buffer()

    if world_size == 1:
        fut = torch.futures.Future()
        fut.set_result(tensor)
        return fut

    # ── Find largest power of 2 ≤ world_size ─────────────────────────────
    pow2 = 1
    while pow2 * 2 <= world_size:
        pow2 *= 2
    remainder = world_size - pow2  # extra ranks beyond power-of-2

    # Working buffer (clone to accumulate into without corrupting original
    # until we're ready)
    buf = tensor.clone()
    recv_buf = torch.empty_like(tensor)

    participating = True  # does this rank participate in main phase?

    # ── Pre-step: fold extra ranks into the power-of-2 set ───────────────
    # Extra ranks (pow2 .. pow2+remainder-1) send to low ranks (0 .. remainder-1).
    if remainder > 0:
        if rank < remainder:
            # I'm a low rank — receive from my extra partner and accumulate
            partner = pow2 + rank
            dist.recv(recv_buf, src=partner)
            buf.add_(recv_buf)
        elif rank >= pow2:
            # I'm an extra rank — send my data to my low partner, then sit out
            partner = rank - pow2
            dist.send(buf.contiguous(), dst=partner)
            participating = False
        # else: rank in [remainder .. pow2-1] — participates, no pre-step work

    # ── Main recursive doubling (on pow2 participating ranks) ────────────
    if participating:
        num_steps = int(math.log2(pow2))

        for step in range(num_steps):
            distance = 1 << step
            partner = rank ^ distance  # XOR to find partner

            # Sendrecv: isend + blocking recv avoids deadlock
            # Both partners send simultaneously, both receive simultaneously
            send_req = dist.isend(buf.contiguous(), dst=partner)
            dist.recv(recv_buf, src=partner)
            send_req.wait()

            # Accumulate partner's data
            buf.add_(recv_buf)

    # ── Post-step: send results back to extra ranks ──────────────────────
    if remainder > 0:
        if rank < remainder:
            partner = pow2 + rank
            dist.send(buf.contiguous(), dst=partner)
        elif rank >= pow2:
            partner = rank - pow2
            dist.recv(buf, src=partner)

    # ── Write result back and average ──────────────────────────────────
    tensor.copy_(buf)

    # ── Divide by world_size to produce the AVERAGE ──────────────────────
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
    ("nccl", "default"):             nccl_default_allreduce_hook,
    ("nccl", "ring"):                nccl_ring_allreduce_hook,
    ("nccl", "recursive_doubling"):  nccl_recursive_doubling_allreduce_hook,
    ("mpi",  "default"):             mpi_default_allreduce_hook,
    ("mpi",  "ring"):                mpi_ring_allreduce_hook,
    ("mpi",  "recursive_doubling"):  mpi_recursive_doubling_allreduce_hook,
}


def get_comm_hook(
    backend: str,
    algorithm: Optional[str] = None
) -> Optional[Callable]:
    """
    Factory function to get the appropriate communication hook.

    Args:
        backend: "nccl" or "mpi"
        algorithm: None → no hook (DDP built-in, fastest path, no Python overhead)
                   "default" → our default hook (baseline with Python-dispatch cost)
                   "ring" → ring allreduce
                   "recursive_doubling" → recursive doubling allreduce

    Returns:
        Hook function, or None (meaning: don't register any hook).
    """
    if algorithm is None:
        # Let DDP use its built-in C++ allreduce path — no Python hook overhead.
        # This is the control / fastest-possible baseline.
        return None

    key = (backend, algorithm)
    if key not in _HOOK_REGISTRY:
        available = [f"{b}/{a}" for b, a in _HOOK_REGISTRY]
        raise ValueError(
            f"Unknown hook: {backend}/{algorithm}. Available: {available}"
        )

    return _HOOK_REGISTRY[key]


def list_available_hooks() -> list:
    """List all registered (backend, algorithm) pairs."""
    return list(_HOOK_REGISTRY.keys())