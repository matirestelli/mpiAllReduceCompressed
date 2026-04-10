"""
Ring AllReduce communication hook — pure dist primitives.

Implements the same two-phase ring algorithm as the C++ MPI code:
  Phase 1: Reduce-scatter  (nranks-1 iterations of send/recv + local add)
  Phase 2: Allgather       (nranks-1 iterations of send/recv + copy)

All tensors stay on GPU throughout. Uses dist.batch_isend_irecv for
safe simultaneous send+recv on NCCL.

The hook performs a SUM allreduce. DDP divides by world_size afterward,
so the final param.grad is the average — DO NOT divide inside this hook.

Future is resolved synchronously (already-completed Future returned).
For overlap with backward pass, wrap in a background thread — see
ring_allreduce_dist_hook_async below.
"""

import math
import threading
import torch
import torch.distributed as dist


# ── Core ring algorithm ───────────────────────────────────────────────────────

def _ring_allreduce_sum(tensor: torch.Tensor) -> None:
    """
    In-place ring allreduce (SUM then divide) on a 1-D GPU tensor.
    Pads to make evenly divisible, runs the ring, trims padding.
    Divides by world_size at the end — produces the AVERAGE.
    DDP does not divide after a custom hook, so we must do it here.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size == 1:
        return  # nothing to do

    numel = tensor.numel()

    # ── Pad so numel is divisible by world_size ───────────────────────────
    chunk_size = math.ceil(numel / world_size)
    padded_size = chunk_size * world_size

    if padded_size != numel:
        # allocate padded buffer and copy tensor into it
        padded = torch.zeros(
            padded_size, dtype=tensor.dtype, device=tensor.device
        )
        padded[:numel] = tensor.flatten()
    else:
        padded = tensor.flatten()

    # views into padded — each is a contiguous chunk of chunk_size elements
    # chunks[i] is the slice "owned" by rank i
    chunks = [padded[i * chunk_size:(i + 1) * chunk_size]
              for i in range(world_size)]

    # ── Ring neighbors (fixed for all iterations) ─────────────────────────
    # src = left neighbor  (we RECEIVE from here)
    # dst = right neighbor (we SEND to here)
    src = (rank - 1 + world_size) % world_size
    dst = (rank + 1) % world_size

    # ── Receive buffer (reused every iteration) ───────────────────────────
    recv_buf = torch.empty(
        chunk_size, dtype=tensor.dtype, device=tensor.device
    )

    # ── Phase 1: Reduce-Scatter ───────────────────────────────────────────
    # Each iteration:
    #   send_idx — which chunk to send this step
    #   recv_idx — which chunk we will receive and accumulate into
    #
    # After nranks-1 iterations, chunks[rank] holds the fully reduced sum
    # for rank's own segment.
    for step in range(world_size - 1):
        send_idx = (rank - step + world_size) % world_size
        recv_idx = (rank - step - 1 + world_size) % world_size

        # post recv FIRST (critical — avoids deadlock)
        # then send
        send_tensor = chunks[send_idx].contiguous()
        ops = [
            dist.P2POp(dist.irecv, recv_buf, src),
            dist.P2POp(dist.isend, send_tensor, dst),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        # local reduction — equivalent to MPI_Reduce_local
        chunks[recv_idx].add_(recv_buf)

    # ── Phase 2: Allgather ────────────────────────────────────────────────
    # Each iteration:
    #   send_idx — chunk to send (already fully reduced)
    #   recv_idx — where to write the received fully-reduced chunk
    #
    # After nranks-1 iterations, every rank has all fully-reduced chunks.
    for step in range(world_size - 1):
        # the fully reduced chunk for this step starts at (rank+1) offset
        send_idx = (rank - step + 1 + world_size) % world_size
        recv_idx = (rank - step + world_size) % world_size

        send_tensor = chunks[send_idx].contiguous()
        ops = [
            dist.P2POp(dist.irecv, recv_buf, src),
            dist.P2POp(dist.isend, send_tensor, dst),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        # overwrite — this chunk is already fully reduced, no accumulation
        chunks[recv_idx].copy_(recv_buf)

    # ── Copy result back into original tensor (trim padding if needed) ────
    if padded_size != numel:
        tensor.copy_(padded[:numel].view_as(tensor))
    elif padded.data_ptr() != tensor.data_ptr():
        tensor.copy_(padded.view_as(tensor))

    # ── Divide by world_size to produce the average ───────────────────────
    # DDP does NOT divide automatically after a custom hook — the hook
    # is fully responsible for producing the final averaged gradient.
    tensor.div_(world_size)


# ── Synchronous hook (simple, no overlap) ────────────────────────────────────

def ring_allreduce_dist_hook(
    state: object,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    DDP communication hook — synchronous ring allreduce via dist primitives.

    Returns an already-resolved Future. No compute/communication overlap.
    Use this for correctness testing.
    """
    tensor = bucket.buffer()
    _ring_allreduce_sum(tensor)

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(tensor)
    return fut


# ── Async hook (background thread, enables overlap) ──────────────────────────

def ring_allreduce_dist_hook_async(
    state: object,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    DDP communication hook — async ring allreduce via dist primitives.

    Launches the ring allreduce on a background thread and returns a
    pending Future immediately, allowing DDP to overlap the backward pass
    of the next bucket with this bucket's communication.

    Requires MPI_THREAD_MULTIPLE if using MPI backend.
    Safe with NCCL backend.
    """
    tensor = bucket.buffer()
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()

    def _do_work():
        try:
            _ring_allreduce_sum(tensor)
            fut.set_result(tensor)
        except Exception as e:
            fut.set_exception(e)

    t = threading.Thread(target=_do_work, daemon=True)
    t.start()
    return fut