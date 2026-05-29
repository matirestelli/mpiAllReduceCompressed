# CLAUDE.md — Project Context for Claude Code

---

## How Claude Code Must Behave in This Project

**Before writing or changing any code:**
1. Explain what you plan to do and why in the chat first
2. Wait for explicit approval ("yes", "go ahead", "do it") before touching any file
3. Never create documentation files, guides, reports, or explanation files — answer questions in the chat only
4. When showing code changes, show only the specific diff/section being changed, not the entire file
5. Never rewrite a whole file unless explicitly asked to — make surgical edits

**When debugging:**
- Propose one hypothesis at a time, explain the reasoning, ask which to pursue
- Do not add logging speculatively — ask what specific data is needed first
- Do not add logging to files that already have enough diagnostic output

---

## What This Project Is

A **benchmarking tool for distributed training communication algorithms** on the Polaris supercomputer at ALCF (Argonne Leadership Computing Facility). The goal is to implement, test, and benchmark custom AllReduce communication hooks for PyTorch DDP (Distributed Data Parallel) and compare them against each other and the DDP built-in baseline. Current hook inventory:

- **`default`** — DDP's built-in `dist.all_reduce`, wrapped for NVTX timing (baseline)
- **`ring`** — custom ring allreduce via P2P primitives (reduce-scatter + P2P allgather), no compression
- **`recursive_doubling`** — MPICH recursive-doubling algorithm (handles non-power-of-two), no compression
- **`ring_zfp` / `ring_zfp_naive`** — naive P2P ZFP compression baseline on ring: compress every send, decompress every recv, re-compress during allgather (paper's explicit baseline)
- **`recursive_doubling_zfp` / `recursive_doubling_zfp_naive`** — same naive ZFP baseline applied to recursive doubling
- **C++/CUDA extension** (`ring_allreduce_cuda.cu`) — direct GPU port of the advisor's C++ ring implementation (separate from the DDP hook system)

---

## Hardware and Software Environment

- **Cluster**: Polaris at ALCF (Argonne National Laboratory)
- **MPI**: Cray MPICH with `MPICH_GPU_SUPPORT_ENABLED=1` — enables **GPU-direct RDMA** (NIC DMA-transfers directly from GPU memory, no CPU staging)
- **GPUs**: 4× NVIDIA A100 per node, one MPI rank per GPU
- **PyTorch**: Built from source with custom CUDA-aware MPI support
- **Launch command**:
  ```
  mpirun -np 4 --ppn 4 --depth=8 --cpu-bind depth -env MPICH_GPU_SUPPORT_ENABLED=1 python interface.py
  ```
- **Backend**: `mpi` (not `nccl`)
- **Python env**: `2025-09-28/base` conda env on Polaris

---

## Project File Structure

```
mpiAllReduceCompressed/
├── interface.py              # Entry point — configure TrainingConfig and call train()
├── config.py                 # TrainingConfig dataclass, create_model(), get_data_loaders()
├── ddp_training.py           # Main training loop, DDP init, hook registration
├── communication_strategy.py # ALL AllReduce hooks live here — THE main file
├── zfp_cuda.py               # Thin Python wrapper — JIT-builds the C++ extension on first import
├── zfp_cuda_extension.cpp    # C++ extension wrapping libzfp's CUDA execution policy (cuZFP)
├── ring_allreduce_dist.py    # Original standalone ring hook (pure dist primitives, no DDP)
├── ring_allreduce_cuda.cu    # C++/CUDA extension — direct GPU port of advisor's C++ code
├── ring_allreduce_cpp_hook.py# Python DDP wrapper for the compiled C++ extension
├── setup.py                  # Build script for ring_allreduce_cuda_ext
└── toy_train.py              # Correctness + timing test with analytically predictable gradients
```

**ZFP runtime requirements**: `libzfp` built with `ZFP_WITH_CUDA=ON`. Set `ZFP_HOME=/path/to/zfp/install` before running. The extension is JIT-compiled by `torch.utils.cpp_extension.load` on first import of `zfp_cuda.py`; delete `__pycache__` if the extension source changes.

---

## The Advisor's Reference C++ Implementation (Ground Truth)

The advisor provided `MPICH_Allreduce_ring` in C++. This is the authoritative reference — all Python and CUDA implementations must match its logic exactly. Here is the full algorithm:

```c
int MPICH_Allreduce_ring(const char* sendbuf, char* recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    int rank, nranks, extent;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    MPI_Type_size(datatype, &extent);

    // --- Chunk sizing: ceil(count/nranks), last chunk may be smaller ---
    int *cnts = malloc(nranks * sizeof(int));
    int *displs = malloc(nranks * sizeof(int));
    int total_count = 0;
    for (int i = 0; i < nranks; i++) {
        cnts[i] = (count + nranks - 1) / nranks;
        if (total_count + cnts[i] > count) {
            cnts[i] = count - total_count;
            break;
        } else total_count += cnts[i];
    }
    displs[0] = 0;
    for (int i = 1; i < nranks; i++)
        displs[i] = displs[i-1] + cnts[i-1];

    // Copy sendbuf -> recvbuf if not in-place
    if (sendbuf != MPI_IN_PLACE)
        memcpy(recvbuf, sendbuf, count * extent);

    void *tmpbuf = malloc(count * extent);  // receive staging buffer

    int src = (nranks + rank - 1) % nranks;  // left neighbor
    int dst = (rank + 1) % nranks;            // right neighbor

    // --- Phase 1: Reduce-Scatter (nranks-1 steps) ---
    for (int i = 0; i < nranks - 1; i++) {
        int recv_rank = (nranks + rank - 2 - i) % nranks;
        int send_rank = (nranks + rank - 1 - i) % nranks;

        MPI_Irecv(tmpbuf, cnts[recv_rank], datatype, src, i, comm, &reqs[0]);
        MPI_Isend(recvbuf + displs[send_rank]*extent, cnts[send_rank],
                  datatype, dst, i, comm, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        MPI_Reduce_local(tmpbuf, recvbuf + displs[recv_rank]*extent,
                         cnts[recv_rank], datatype, op);
    }

    // --- Phase 2: Allgather (single collective call) ---
    MPI_Allgatherv(MPI_IN_PLACE, -1, MPI_DATATYPE_NULL,
                   recvbuf, cnts, displs, datatype, comm);

    free(tmpbuf); free(cnts); free(displs);
    return MPI_SUCCESS;
}
```

### Key observations about the C++ reference:
- **Tags in Phase 1**: each step `i` uses tag=`i` explicitly
- **tmpbuf size**: `count * extent` — sized to the FULL buffer, not per-chunk
- **Phase 2** uses a single `MPI_Allgatherv` rather than a P2P loop
- The function returns a **SUM**, not an average — callers divide by nranks if they need the mean

### How the C++/CUDA extension adapts this for GPU:

| C++ reference | CUDA extension (`ring_allreduce_cuda.cu`) |
|---|---|
| `malloc(count * extent)` for tmpbuf | `cudaMalloc` — tmpbuf must live on GPU |
| `memcpy` for in-place copy | `cudaMemcpy DeviceToDevice` |
| `MPI_Reduce_local(tmpbuf, recvbuf+...)` | Custom `elementwise_add_kernel<<<>>>` CUDA kernel |
| `MPI_Allgatherv` unchanged | `MPI_Allgatherv` unchanged — CUDA-aware MPI handles GPU pointers |
| No sync needed (CPU buffers) | `cudaDeviceSynchronize()` before MPI to flush backward-pass kernels |
| `MPI_Isend` / `MPI_Irecv` unchanged | Unchanged — CUDA-aware MPI handles GPU pointers directly |

The CUDA extension is compiled via `python setup.py install` using `torch.utils.cpp_extension.CUDAExtension`. It exposes `ring_allreduce_cuda_ext.ring_allreduce(tensor)` to Python.

---

## The Python Ring Implementation in `communication_strategy.py`

The Python implementation translates the C++ index formulas into `torch.distributed` P2P operations. The index formulas differ slightly in naming convention from the C++ code but are mathematically equivalent:

**C++ Phase 1**: `recv_rank = (nranks + rank - 2 - i) % nranks`, `send_rank = (nranks + rank - 1 - i) % nranks`

**Python Phase 1** (equivalent, step=i):
```python
send_idx = (rank - step - 1 + world_size) % world_size  # = send_rank
recv_idx = (rank - step - 2 + world_size) % world_size  # = recv_rank
```

**Python Phase 2** (P2P loop instead of MPI_Allgatherv):
```python
send_idx = (rank - step     + world_size) % world_size
recv_idx = (rank - step - 1 + world_size) % world_size
```

---

## DDP Hook Contract (Critical — Do Not Get This Wrong)

When **no hook** is registered, DDP's C++ Reducer:
1. SUM-allreduces all gradients
2. Divides by `world_size` internally

When a hook **is** registered, DDP does **NOT** divide. The hook must produce the averaged gradient itself. Every custom hook must call `tensor.div_(world_size)`.

Hook signature:
```python
def my_hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    tensor = bucket.buffer()   # flat 1D GPU tensor of packed gradients for this bucket
    # ... allreduce producing SUM ...
    tensor.div_(world_size)    # produce AVERAGE — DDP will not do this
    fut = torch.futures.Future()
    fut.set_result(tensor)
    return fut
```

`get_comm_hook()` returns `(hook, state)` as a tuple. Registration in `ddp_training.py`:
```python
result = get_comm_hook(config.backend, config.comm_algorithm)
if result is not None:
    comm_hook, comm_state = result
    model.register_comm_hook(state=comm_state, hook=comm_hook)
```

---

## State Object Design

Every hook has a top-level state class (`*AllreduceState`) that holds a `dict[int, *BucketState]` keyed by `bucket.index()`. On first call for each bucket the bucket-level state is initialized with persistent GPU buffers that are never reallocated — fixed GPU addresses are required by Cray MPICH GPU-direct RDMA.

| Top-level state | Bucket state | Persistent buffers |
|---|---|---|
| `RingAllreduceState` | `RingBucketState` | `flat`, `send_bufs[i]`, `recv_bufs[i]` (per-chunk) |
| `RecursiveDoublingAllreduceState` | `RecursiveDoublingBucketState` | `flat`, `tmp` (full tensor) |
| `ZfpRingAllreduceState` | `ZfpRingBucketState` | `flat`, `send_comp[i]`, `recv_comp[i]`, `recv_decomp[i]`, `send_sizes[i]`, `recv_sizes[i]` (per-chunk) |
| `ZfpRecursiveDoublingAllreduceState` | `ZfpRecursiveDoublingBucketState` | `flat`, `tmp`, `send_comp`, `recv_comp`, `send_size`, `recv_size` (full tensor) |

**ZFP size tensors**: `send_sizes[i]` / `recv_sizes[i]` (ring) and `send_size` / `recv_size` (recursive doubling) are 1-element int64 tensors on-device. They carry the actual compressed byte count for each exchange so only true bytes travel over the network. Reading `.item()` after `wait()` does not require a host round-trip.

The hot path for all hooks:
```python
bstate.flat.copy_(tensor.flatten())        # load fresh gradients into persistent buffer
# ... allreduce phases using persistent send/recv bufs ...
tensor.copy_(bstate.flat.view_as(tensor))  # write result back to bucket buffer
tensor.div_(world_size)                    # produce average (DDP does not divide when a hook is registered)
```

---

## Toy Training for Correctness Testing (`toy_train.py`)

**Design**: `nn.Linear(dim, 1, bias=False)` with `weight = 0`. Each rank `r` gets:
- `x = ones(1, dim)`
- `y = -(r+1)/2`

With `W=0`: forward=0, `dL/dW = 2*(0-y)*x = (r+1)*ones(dim)`.

So rank `r` produces gradient `(r+1)` on every element. After correct allreduce:
```
expected = (1 + 2 + ... + world_size) / world_size = (world_size + 1) / 2
```
With 4 ranks: **2.5** on every element of every rank. Closed-form, no floating-point ambiguity. Any other value = the allreduce is wrong.

---

## All Bugs Encountered (Do Not Re-Introduce These)

### Bug 1 — Stale RDMA registration on recv buffers
**Code pattern that triggers it**: `recv_buf = torch.empty(chunk_size, ...)` inside the hook body (new allocation every backward pass).
**Why**: Cray MPICH caches GPU buffer → RDMA registration mappings. New GPU addresses from PyTorch's caching allocator cause stale registrations. DMA writes to wrong memory.
**Fix**: Allocate `recv_bufs` once in `RingBucketState.initialize()`, reuse forever.

### Bug 2 — Single state object for all DDP buckets
**Code pattern that triggers it**: One `RingAllreduceState` for the entire model, `initialized` flag set on first bucket, subsequent buckets hit wrong-sized `flat`.
**Why**: DDP fires the hook once per bucket per backward. ResNet50 has 6 buckets with 6 different sizes.
**Fix**: `dict[int, *BucketState]` keyed by `bucket.index()`. Originally the code keyed by `tensor.numel()` because `bucket.index()` returned 0 for every bucket in this PyTorch build — but this was fixed by adding `static_graph=True` to the DDP constructor and running a dummy backward before `register_comm_hook`. With those two changes, `bucket.index()` is stable and correct. The code now keys by `bucket.index()` and re-initializes the state if `tensor.numel()` has changed (DDP rebuild guard).

### Bug 3 — Stale RDMA registration on send buffers
**Code pattern that triggers it**: `send_tensor = chunks[send_idx].clone()` or `.contiguous()` inside the loop — new allocation every step.
**Why**: Same as Bug 1, on the send side.
**Fix**: `send_bufs[i]` persistent buffers in `RingBucketState`. `bstate.send_bufs[send_idx].copy_(chunks[send_idx])` before each isend.

### Bug 4 — MPI global tag counter drift across buckets
**Code pattern that triggers it**: Using `dist.batch_isend_irecv` or `dist.send`/`dist.recv` without explicit `tag=` argument.
**Why**: `ProcessGroupMPI` auto-assigns tags via a global counter that increments on every P2P op. With 6 buckets firing in different orders across ranks (GPU backward completion times vary), counters drift. Wrong messages match wrong recvs.
**Diagnostic signature**: Some chunks exactly correct, others 100x+ wrong. Wrong values differ across calls (not constant).
**Fix**: Pass explicit `tag=tag` to every `P2POp`.

### Bug 5 — RDMA cache per tag (tags reused across hook calls)
**Code pattern that triggers it**: Explicit tags but with a formula like `bucket_id * 1000 + phase * 100 + step` — the same tag fires again on the next backward pass.
**Why**: Cray MPICH caches RDMA registrations per tag. When tag=2001 fires a second time with the same persistent recv_buf address, MPICH may reuse the stale cached mapping instead of issuing a fresh DMA registration.
**Diagnostic signature**: Wrong values are **constant across multiple calls** (same garbage value every batch).
**Fix**: Global monotonically increasing tag counter — never reuse any tag in the process lifetime:
```python
_tag_counter: int = 0
def _next_tag() -> int:
    global _tag_counter
    t = _tag_counter
    _tag_counter += 1
    return t
```
Up to ~360,000 tags for 100 epochs — well within Cray MPICH's 2^31-1 limit.

---

## Current Status

**Ring and recursive doubling hooks are correct and training converges.** Loss and accuracy with these hooks now match the DDP built-in baseline.

**ZFP hooks (`ring_zfp`, `recursive_doubling_zfp`) were integrated** from `prova_zfp/` into the main `communication_strategy.py`. Key design points:
- Uses on-device `send_sizes`/`recv_sizes` tensors that carry the actual compressed byte count at runtime (not a fixed `compressed_bytes` constant).
- Each P2P step uses two rounds: a size exchange (size_tag) followed by a data exchange (data_tag) using exact-size views of the persistent compressed buffers.
- `get_comm_hook()` correctly instantiates the right state class for all algorithm types.
- `reset_bucket_compute_markers()` and `open_first_bucket_compute_range()` NVTX helpers present.
- `ring_zfp_naive` and `recursive_doubling_zfp_naive` aliases added for paper-terminology clarity.

**ZFP hooks have not yet been run on Polaris.** They need `libzfp` built with `ZFP_WITH_CUDA=ON` and `ZFP_HOME` set. First step is to verify correctness with `toy_train.py` before benchmarking.

---

## Known Rules — Never Violate These

1. **Key state by `bucket.index()`, not `tensor.numel()`** — `bucket.index()` is now stable (fixed by `static_graph=True` + dummy backward before hook registration). State is re-initialized if `tensor.numel()` changes (DDP rebuild guard), but the dict key is always the bucket index.
2. **Never allocate CUDA tensors inside the hook body** (no `torch.empty()`, `.clone()`, `.contiguous()` that creates new storage). Every tensor passed to MPI must have a fixed GPU address for the process lifetime.
3. **Always use explicit `tag=` on every `P2POp`**.
4. **Tags must be globally monotonically increasing** — never reuse a tag.
5. **Hook must call `tensor.div_(world_size)`** — DDP does not do this.
6. **C++ extension needs `cudaDeviceSynchronize()` before MPI** — flushes backward-pass GPU kernels.
7. **ZFP hooks: call `_cuda_device_sync()` after every compress/decompress** — cuZFP uses its own internal CUDA stream, not PyTorch's. A regular stream sync (`_cuda_sync`) is not enough; only `torch.cuda.synchronize()` (device-wide) guarantees cuZFP has finished writing to GPU memory before GPU-direct MPI DMA or a PyTorch kernel reads it.
8. **ZFP hooks: two tags per step, not one** — the size-exchange round (size_tag) and data-exchange round (data_tag) must use separate tags. All ranks — including inactive ones — consume both tags in the same order so the global counter stays aligned. Never combine size and data into one message.
9. **Delete `__pycache__`** after editing any `.py` file before running on cluster.
10. **Never create guide/documentation/report files** — explain everything in chat.
11. **Always get approval before writing or changing code** — explain the plan first.
12. **`contiguous()` is safe only if it does not allocate new storage** — check with `tensor.data_ptr()` if uncertain. For MPI-facing tensors use persistent pre-allocated buffers only.

---

## Timing Methodology

**First allreduce inside DDP** — `TimedHook` wrapper around the hook function:
- Uses `time.perf_counter()` (wall clock), NOT `torch.cuda.Event` — Events cannot cross threads, async hooks run on background threads
- Sync hooks: measured from entry to return
- Async hooks: `fut.add_done_callback()` fires on background thread when `fut.set_result()` is called

**Standalone benchmark** — `benchmark_allreduce()`:
- `torch.cuda.Event` for GPU-accurate timing
- 10 warmup iterations, N timed iterations
- Reports **max across ranks** via `dist.all_reduce(MAX)` — slowest rank is the bottleneck
- Algorithmic bandwidth: `algbw = 2*(W-1)/W * size_GB / time_s`

**Training loop timer** — wall clock with barrier before/after N iterations, max across ranks.

---

## Nsys Profiling Flags (per-algorithm)

The nsys flags in `trainJob_profile_mpi.sh` must differ between algorithms because the default hook uses `dist.all_reduce(async_op=True)` (which becomes an async `MPI_Iallreduce`), while the ring hook uses synchronous P2P MPI ops.

**Current flags (for `default` algorithm):**
```
--trace=cuda,nvtx,osrt
--cuda-event-trace=false
```
`--trace=mpi` is omitted because nsys's async MPI collective tracing segfaults (exit 139) on Cray MPICH when tracking `MPI_Iallreduce` completions that resolve on a background thread. `--cuda-event-trace=false` removes additional overhead that was warned about at startup.

**Flags to restore when profiling `ring` algorithm:**
```
--trace=cuda,mpi,nvtx,osrt
```
Remove `--cuda-event-trace=false` (or leave it — it reduces overhead but loses cross-stream dependency edges). The ring hook's synchronous P2P operations are safe for nsys MPI tracing.

---

## Model Being Trained

**ResNet50 on CIFAR10**, modified stem:
- 7×7 stride-2 conv + maxpool replaced with 3×3 stride-1 conv, no maxpool (preserves 32×32 resolution)
- `num_classes=10`, `batch_size=128` per rank, `effective_batch=512`
- DDP creates **6 buckets** with sizes: 23.5M, 6.8M, 6.6M, 6.6M, 2.4M, 1.1M elements

**Ground truth (built-in DDP, no hook)**:
- Epoch 1 batch 50: Loss ~2.38, Acc ~15%
- Epoch 15: Val acc ~62%, Loss ~1.1

The `data/` directory (CIFAR-10) is gitignored and was lost in the `rm -rf *` incident. Re-download by running any training script once — PyTorch will fetch it automatically if `download=True` is set in `torchvision.datasets.CIFAR10`.
