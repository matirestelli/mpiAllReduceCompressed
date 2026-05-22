mpiexec -np 8 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    python interface.py
```

### 16 GPUs

16 GPUs should be requested as 4 nodes:

```bash
#PBS -l select=4:ngpus=4

mpiexec -np 16 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    python interface.py
```

Use `debug-scaling` for multi-node debug jobs unless the queue policy allows the requested shape in `debug`.

## Communication Algorithms

- `None` / `none`: DDP built-in AllReduce, no custom hook.
- `default`: Built-in DDP AllReduce wrapped for timing/NVTX only.
- `ring`: Custom ring AllReduce using reduce-scatter plus allgather.
- `recursive_doubling`: Recursive doubling AllReduce.
- `ring_zfp_naive`: Ring with ZFP compression at each exchange step.
- `recursive_doubling_zfp_naive`: Recursive doubling with ZFP compression.
- `ring_zfp_online_coll`: Ring with collective-level online ZFP compression.
- `recursive_doubling_zfp_online_coll`: Recursive doubling with collective-level online ZFP compression.

## Data Loader Requirement for Large Per-Rank Batches

For 16 GPUs with `batch_size=128`, CIFAR-10 produces a partial final global batch:

```text
50000 / (16 x 128) = 24.4
```

Built-in DDP can handle this, but custom hooks such as `ring` can deadlock if the final training step is uneven. The training loader must drop the final partial batch:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
```

With `drop_last=True`, the 16-GPU `batch_size=128` run reports:

```text
Train batches: 24
```

instead of:

```text
Train batches: 25
```

Keep validation unchanged unless doing timing-only validation.

## Hook Timing

Hook timing can be enabled from the job script:

```bash
export DDP_HOOK_TIMING=1
export DDP_HOOK_TIMING_RANK0_ONLY=1
```

For MPI launches, pass these through `mpiexec` if needed:

```bash
-env DDP_HOOK_TIMING="${DDP_HOOK_TIMING}"
-env DDP_HOOK_TIMING_RANK0_ONLY="${DDP_HOOK_TIMING_RANK0_ONLY}"
```

When debugging hangs, disable hook timing first:

```bash
export DDP_HOOK_TIMING=0
```

## Known Constraints

### Fixed GPU Buffer Addresses

Cray MPICH caches GPU memory to RDMA registration mappings. MPI-facing tensors should be preallocated and reused. Avoid allocating new GPU tensors inside hook bodies.

### Monotonically Increasing MPI Tags

Custom P2P hooks use globally increasing MPI tags. Do not reuse tags across operations.

### cuZFP Synchronization

ZFP compression and decompression use cuZFP's internal CUDA stream. Synchronize before MPI or PyTorch kernels consume compressed/decompressed buffers.

### `static_graph=True`

DDP uses `static_graph=True` so bucket indices remain stable across backward passes.

### Delete `__pycache__` After Editing Python

Before submitting after code edits:

```bash
rm -rf __pycache__
```

This avoids stale bytecode on compute nodes.