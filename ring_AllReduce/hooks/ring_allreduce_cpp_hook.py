"""
DDP communication hook wrapping the C++/CUDA ring allreduce extension.

This module provides two hooks:
    ring_allreduce_cpp_hook         — synchronous (no overlap)
    ring_allreduce_cpp_hook_async   — async via background thread (with overlap)

The C++ extension (ring_allreduce_cuda_ext) must be compiled first:
    cd hooks && python setup.py install

The hook performs a SUM allreduce. DDP divides by world_size afterward.
DO NOT divide inside the hook.

Background thread notes:
    - MPI must be initialized with MPI_THREAD_MULTIPLE for thread safety
    - mpi4py initializes with MPI_THREAD_MULTIPLE by default
    - Verify with: from mpi4py import MPI; assert MPI.Query_thread() == MPI.THREAD_MULTIPLE
"""

import os
import sys
import threading
import torch
import torch.distributed as dist

# Add the hooks directory to sys.path so Python can find the compiled .so
# regardless of where the script is run from
_hooks_dir = os.path.dirname(os.path.abspath(__file__))
if _hooks_dir not in sys.path:
    sys.path.insert(0, _hooks_dir)


# ── Import compiled extension ─────────────────────────────────────────────────

try:
    import ring_allreduce_cuda_ext as _ext
    _EXTENSION_AVAILABLE = True
except ImportError:
    _EXTENSION_AVAILABLE = False


def _check_extension():
    if not _EXTENSION_AVAILABLE:
        raise ImportError(
            "ring_allreduce_cuda_ext is not installed. "
            "Run: cd hooks && python setup.py install"
        )


# ── Synchronous hook ──────────────────────────────────────────────────────────

def ring_allreduce_cpp_hook(
    state: object,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    DDP communication hook — synchronous C++/CUDA ring allreduce.

    Calls the compiled C++ extension which runs the full ring algorithm
    (MPI_Isend, MPI_Irecv, MPI_Waitall, CUDA elementwise add, MPI_Allgatherv)
    entirely in C++ with CUDA-aware MPI.

    The main thread blocks until the C++ call returns.
    Returns an already-resolved Future — no compute/communication overlap.

    Use this for:
        - correctness testing
        - clusters without MPI_THREAD_MULTIPLE support
    """
    _check_extension()

    tensor = bucket.buffer()

    # ensure contiguous GPU tensor before passing to C++
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # entire ring runs in C++ — blocks until done
    _ext.ring_allreduce(tensor)

    # DDP does not divide after a custom hook — divide here
    tensor.div_(dist.get_world_size())

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(tensor)
    return fut


# ── Async hook (background thread) ───────────────────────────────────────────

def ring_allreduce_cpp_hook_async(
    state: object,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    DDP communication hook — async C++/CUDA ring allreduce.

    Launches the C++ ring allreduce on a background thread and returns
    a pending Future immediately. DDP can overlap the backward pass of
    the next gradient bucket with this bucket's ring communication.

    The background thread explicitly sets the CUDA device before calling
    the C++ extension — CUDA contexts are thread-local so without this
    the background thread has no context and GPU pointers are invalid.

    Requirements:
        - MPI initialized with MPI_THREAD_MULTIPLE
        - Verify: from mpi4py import MPI; assert MPI.Query_thread() == MPI.THREAD_MULTIPLE
    """
    _check_extension()

    tensor = bucket.buffer()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # capture device index on the main thread while we have the CUDA context
    device_index = tensor.device.index

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()

    def _do_work():
        try:
            # initialize CUDA context on this background thread
            # without this, CUDA operations have no context and crash
            # with "illegal memory access"
            torch.cuda.set_device(device_index)

            _ext.ring_allreduce(tensor)
            tensor.div_(dist.get_world_size())
            fut.set_result(tensor)
        except Exception as e:
            fut.set_exception(e)

    t = threading.Thread(target=_do_work, daemon=True)
    t.start()

    return fut


# ── Thread safety check ───────────────────────────────────────────────────────

def check_mpi_thread_safety() -> bool:
    """
    Verify MPI was initialized with MPI_THREAD_MULTIPLE.
    Call this before using ring_allreduce_cpp_hook_async.

    Returns True if thread-safe, False otherwise.
    """
    try:
        from mpi4py import MPI
        level = MPI.Query_thread()
        if level == MPI.THREAD_MULTIPLE:
            return True
        else:
            level_names = {
                MPI.THREAD_SINGLE:     "THREAD_SINGLE",
                MPI.THREAD_FUNNELED:   "THREAD_FUNNELED",
                MPI.THREAD_SERIALIZED: "THREAD_SERIALIZED",
                MPI.THREAD_MULTIPLE:   "THREAD_MULTIPLE",
            }
            print(
                f"WARNING: MPI thread level is {level_names.get(level, level)}, "
                f"need THREAD_MULTIPLE for async hook. "
                f"Using async hook may cause race conditions or crashes."
            )
            return False
    except ImportError:
        print("WARNING: mpi4py not available, cannot check MPI thread safety.")
        return False