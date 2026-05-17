# AllReduce Benchmarking Framework

A benchmarking tool for distributed deep learning communication algorithms on the [Polaris supercomputer](https://www.alcf.anl.gov/polaris) at ALCF (Argonne Leadership Computing Facility).

## What this is

In Distributed Data Parallel (DDP) training, each GPU holds a full copy of the model and trains independently on its own data partition. After each backward pass, all GPUs must synchronize their gradients through a collective communication primitive called **AllReduce** — every GPU sends its local gradients and receives the globally averaged result. This synchronization step is a well-documented bottleneck in large-scale training, and as models grow and supercomputers scale to exascale, the choice of algorithm and backend matters more and more.

This framework implements and benchmarks multiple AllReduce communication strategies as PyTorch DDP communication hooks. Because all hooks plug into the same training loop on the same model and data, results are directly comparable. The goal is to characterize how algorithm choice, hardware, backend, and model type interact — and to inform which combination performs best for a given workload.

The framework is built and tested on Polaris (NVIDIA A100), with a planned port to [Frontier (OLCF)](https://www.olcf.ornl.gov/frontier/) (AMD MI300X) to enable cross-hardware comparisons.

---

## Hardware

- **Cluster**: Polaris at ALCF
- **GPUs**: 4× NVIDIA A100 per node
- **Interconnect**: Slingshot-11 (HPE Cray)
- **MPI**: Cray MPICH 9.0.1 with GPU-direct RDMA (`MPICH_GPU_SUPPORT_ENABLED=1`)
- **Configuration**: 1 MPI rank per GPU, 4 ranks per node

---

## Software environment

This project runs inside a specific pre-built software stack on Polaris. **Do not mix environments.**

### PyTorch

PyTorch is built from source at:
```
/lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build/src/pytorch
```
This custom build enables CUDA-aware MPI support (`dist.is_mpi_available() == True`). The standard Polaris PyTorch modules do not expose this. Do not substitute with a conda-installed PyTorch.

**Version**: source build from the PyTorch main branch, September 2025.

### Conda environment

```bash
conda activate base   # the 2025-09-28 base environment on Polaris
```

Activated automatically by `envScript3.sh`. Do not use a different environment.

### CUDA

Version 12.9, loaded via:
```bash
module load cuda/12.9
```

### Cray MPICH and GPU-direct RDMA

```bash
module load cray-mpich
module load craype-accel-nvidia80
```

The GPU Transfer Layer (GTL) shared library is preloaded at runtime:
```
/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib/libmpi_gtl_cuda.so
```

GTL enables GPU-direct RDMA: the NIC can DMA directly from GPU memory without CPU staging. All MPI-facing GPU tensors must have **fixed GPU addresses for the entire process lifetime** — see the Known Constraints section.

### ZFP / cuZFP

Required only for the `ring_zfp` and `recursive_doubling_zfp` algorithms.

- `libzfp` must be built with `ZFP_WITH_CUDA=ON`
- Pre-built install: `/eagle/UIC-HPC/mrest/zfp-install`
- `ZFP_HOME` is set automatically by `envScript3.sh`

The Python C++ extension (`zfp_cuda_extension.cpp`) is JIT-compiled by `torch.utils.cpp_extension.load` on the first import of `zfp_cuda.py`. This happens once per node at job start — expect a ~30s compilation delay on the first run. Subsequent runs use the cached build (in `__pycache__`).

Compression rate is controlled by the `DDP_ZFP_RATE` environment variable (default: `8.0` bits/value, i.e. 4× compression for float32).

---

## Setup: first-time data download

CIFAR-10 is **not included in this repository** (gitignored). The `./data/` directory must exist on the filesystem before training.

**Step 1** — temporarily set `download=True` in `config.py` (lines ~114–115):
```python
train_dataset = ds_cls(root=config.data_dir, train=True, download=True, ...)
val_dataset   = ds_cls(root=config.data_dir, train=False, download=True, ...)
```

**Step 2** — run any training script once. PyTorch downloads CIFAR-10 (~170 MB) through the Polaris HTTP proxy (set automatically by `envScript3.sh`).

**Step 3** — revert both lines back to `download=False` to avoid a network check on every subsequent run.

Expected directory structure after download:
```
./data/
└── cifar-10-batches-py/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    └── test_batch
```

---

## Configuring an experiment

All experiment parameters are set in **`interface.py`**. Edit this file before submitting a job — no other file needs to change for a standard experiment.

```python
config = TrainingConfig(

    # ── Model (uncomment one, comment the rest) ───────────────────────────
    # model_name="resnet18",           # ~11M params — fastest, lightweight baseline
    # model_name="resnet50",           # ~25M params — standard CIFAR-10 benchmark
    # model_name="resnet101",          # ~44M params — deeper ResNet
    model_name="resnext101_32x8d",     # ~88M params — grouped convs (32 groups × 8 width)
    # model_name="convnext_tiny",      # ~28M params — modern architecture, no stem fix needed
    # model_name="convnext_small",     # ~50M params — larger ConvNeXt variant

    dataset="cifar10",
    num_classes=10,
    num_epochs=10,

    # ── Batch size ────────────────────────────────────────────────────────
    # 128 is safe for all models on A100-40GB with CIFAR-10 (32×32 input).
    # Effective batch = batch_size × world_size. At 4 GPUs × 128 = 512 effective.
    # Keep effective batch ≤ 2048 for CIFAR-10 without LR scaling.
    # Go to 64 if OOM. See BATCH_SIZE.md for the full decision guide.
    batch_size=128,

    learning_rate=0.01,
    momentum=0.9,
    weight_decay=5e-4,

    backend="mpi",     # "mpi" for Cray MPICH + GPU-direct RDMA (primary)
                       # "nccl" for NCCL via env:// init (custom hooks also supported)

    # ── Communication algorithm (uncomment one, comment the rest) ─────────
    comm_algorithm=None,                        # DDP built-in AllReduce (baseline, no hook)
    # comm_algorithm="default",                 # built-in wrapped with NVTX timing only
    # comm_algorithm="ring",                    # custom ring: reduce-scatter + allgather via P2P
    # comm_algorithm="recursive_doubling",      # MPICH recursive doubling, handles non-power-of-2
    # comm_algorithm="ring_zfp",                # ring + ZFP compression (naive paper baseline)
    # comm_algorithm="ring_zfp_naive",          # alias for ring_zfp
    # comm_algorithm="recursive_doubling_zfp",  # recursive doubling + ZFP compression
    # comm_algorithm="recursive_doubling_zfp_naive",  # alias for recursive_doubling_zfp

    cifar_stem=True,        # replaces 7×7 stride-2 conv+maxpool with 3×3 stride-1 conv
                            # required for resnet/resnext on 32×32 CIFAR images
    data_dir="./data",
    checkpoint_dir="./checkpoints",
    seed=42,
)
```

**Notes on specific settings:**

- `cifar_stem=True` is required for all ResNet and ResNeXt models on CIFAR-10. ConvNeXt models ignore it (they handle small inputs natively).
- ZFP algorithms (`ring_zfp`, `recursive_doubling_zfp`) require `ZFP_HOME` to be set and `libzfp` with CUDA support — handled automatically by `envScript3.sh` on Polaris.
- The MPI backend uses GPU-direct RDMA and is the primary target for custom hooks. NCCL is supported but uses a different init path (env://) and does not use Cray MPICH.

---

## Running experiments on Polaris

### 1. Edit `interface.py`

Choose your model, algorithm, backend, batch size, and number of epochs.

### 2. Delete `__pycache__`

**Always do this before submitting**, especially after editing any `.py` file. Compute nodes load bytecode from `__pycache__` and will silently ignore your edits otherwise.

```bash
rm -rf __pycache__
```

### 3. Submit the job

```bash
qsub trainJob.sh
```

### Available job scripts

| Script | Purpose | Queue | Walltime |
|---|---|---|---|
| `trainJob.sh` | Standard training run via `interface.py` | `debug-scaling` | 1h |
| `trainJob_nccl_nogtl.sh` | NCCL backend, GTL preload disabled | `debug-scaling` | 1h |
| `trainJob_profile_mpi.sh` | Nsys profiling (CUDA + NVTX + MPI) | `debug-scaling` | 1h |
| `trainJob_toy.sh` | Correctness check with toy linear model | `debug-scaling` | 1h |
| `baselineJob.sh` | Baseline run (DDP built-in, no hook) | `debug-scaling` | 1h |

### Checking job status

```bash
qstat -u $USER          # list your running/queued jobs
qstat -f <jobid>        # full details for a specific job
```

Output is written to `train.<jobid>.out` in the working directory.

### Walltime guidance

The `debug-scaling` queue caps at 1 hour. Approximate time per epoch on 4× A100 (CIFAR-10, `batch_size=128`):

| Model | Time / epoch |
|---|---|
| ResNet18 | ~15–20s |
| ResNet50 | ~40–45s |
| ResNet101 | ~60–80s |
| ResNeXt101-32x8d | ~90–120s |

For ResNeXt101 at 10 epochs (~20 min), 1 hour is sufficient. For longer runs switch to `preemptable` or `prod` queues.

### Profiling with Nsys

```bash
qsub trainJob_profile_mpi.sh
```

NVTX ranges mark: iteration, forward pass, backward pass, optimizer step, and per-bucket communication. The nsys flags differ between algorithms:

- **`default` / NCCL algorithms**: `--trace=cuda,nvtx,osrt --cuda-event-trace=false` (omit `--trace=mpi` — nsys's async MPI collective tracing segfaults on Cray MPICH with `MPI_Iallreduce`)
- **`ring` / `recursive_doubling` / ZFP variants**: `--trace=cuda,mpi,nvtx,osrt` (synchronous P2P ops are safe for MPI tracing)

See the comment block inside `trainJob_profile_mpi.sh` to switch between these.

### Correctness check before benchmarking

Before running a new algorithm at scale, verify it with the toy model:

```bash
qsub trainJob_toy.sh
```

`toy_train.py` uses a single `nn.Linear` layer with analytically predictable gradients. With 4 ranks, the correct averaged gradient is exactly **2.5** on every parameter element. Any other value means the AllReduce is broken.

---

## Output files

Each training run produces:

- **`<model>_<epochs>_<backend>_<algo>.log`** — full stdout log (also printed live to console)
- **`results/<model>_<backend>_<algo>_<timestamp>.csv`** — per-epoch metrics table:

| Column | Description |
|---|---|
| `epoch` | Epoch number |
| `lr` | Learning rate used this epoch |
| `train_loss` / `train_acc` | Training loss and accuracy |
| `val_loss` / `val_acc` | Validation loss and accuracy |
| `epoch_time_s` | Wall time for the epoch |
| `model`, `backend`, `algorithm` | Experiment configuration |
| `world_size`, `batch_size`, `num_epochs` | Run parameters |

Best model checkpoint saved to:
```
checkpoints/<model_name>_best.pth
```

---

## Communication algorithms

### `None` — DDP built-in (baseline)
No hook registered. DDP's C++ Reducer performs a `dist.all_reduce` and divides by world size internally. On the MPI backend this becomes `MPI_Iallreduce`; on NCCL it uses NCCL's native collective.

### `ring` — Custom ring AllReduce
Implements the advisor's C++ reference algorithm in PyTorch distributed P2P primitives. Two phases:
1. **Reduce-scatter** (world_size − 1 steps): each rank sends and receives a chunk, accumulating partial sums
2. **Allgather** (world_size − 1 steps): each rank propagates its fully-reduced chunk to all others

Uses persistent GPU buffers and globally monotonically increasing MPI tags. Agnostic to backend (works with MPI and NCCL P2P ops).

### `recursive_doubling` — Recursive doubling AllReduce
Implements the MPICH binary-tree recursive doubling algorithm. At each step, ranks that are a power-of-two apart exchange and reduce their full buffers. Handles non-power-of-two process counts via a peel/finalize step on the odd rank out.

### `ring_zfp` / `ring_zfp_naive` — Ring with ZFP compression
Ring AllReduce where every P2P message is compressed on-GPU using cuZFP fixed-rate mode before sending and decompressed after receiving. Each exchange step uses two rounds: a size exchange (int64 byte count) followed by the compressed payload at the exact received size. This is the naive paper baseline — compresses and decompresses at every step, including during allgather.

### `recursive_doubling_zfp` / `recursive_doubling_zfp_naive` — Recursive doubling with ZFP
Recursive doubling with the same ZFP compression protocol applied to every P2P exchange.

---

## Project structure

```
mpiAllReduceCompressed/
├── interface.py              # Entry point — configure TrainingConfig and call train()
├── config.py                 # TrainingConfig dataclass, create_model(), get_data_loaders()
├── ddp_training.py           # DDP init, training loop, validation, CSV logging
├── communication_strategy.py # All AllReduce hooks — ring, recursive doubling, ZFP variants
├── zfp_cuda.py               # JIT-builds the ZFP C++ extension on first import
├── zfp_cuda_extension.cpp    # C++ extension wrapping libzfp's CUDA execution policy
├── ring_allreduce_dist.py    # Standalone ring hook (pure dist primitives, no DDP)
├── ring_allreduce_cuda.cu    # C++/CUDA GPU port of the advisor's C++ ring reference
├── ring_allreduce_cpp_hook.py# DDP wrapper for the compiled C++ extension
├── setup.py                  # Build script for ring_allreduce_cuda_ext
├── toy_train.py              # Correctness + timing test with analytically predictable gradients
├── trainJob.sh               # PBS job — standard training
├── trainJob_profile_mpi.sh   # PBS job — nsys profiling
├── trainJob_nccl_nogtl.sh    # PBS job — NCCL without GTL preload
├── trainJob_toy.sh           # PBS job — toy correctness check
├── baselineJob.sh            # PBS job — baseline (built-in allreduce)
├── envScript3.sh             # Environment setup (modules, conda, ZFP, MPI, proxy)
├── envScript_nccl.sh         # Environment setup variant for NCCL backend
├── BATCH_SIZE.md             # Batch size selection guide with memory and quality analysis
└── data/                     # CIFAR-10 dataset (gitignored — download separately)
```

---

## Known constraints

**Fixed GPU buffer addresses**
Cray MPICH caches GPU memory → RDMA registration mappings on first use. New GPU addresses from PyTorch's caching allocator cause stale RDMA registrations and DMA writes to wrong memory. All MPI-facing tensors are pre-allocated once in the bucket state initializer and reused across backward passes. Never allocate tensors inside the hook body.

**Monotonically increasing MPI tags**
A global counter increments on every P2P operation and never resets. Reusing any tag (even with the same buffer address) causes Cray MPICH to serve a stale cached RDMA mapping. With ~360k tags for 100 epochs, this is well within MPICH's 2^31−1 limit.

**cuZFP synchronization**
ZFP compression and decompression run on cuZFP's own internal CUDA stream. After every compress or decompress call, `torch.cuda.synchronize()` (device-wide) is required before MPI or a PyTorch kernel reads the result. A regular CUDA stream sync is not sufficient.

**`static_graph=True` on DDP**
Required for `bucket.index()` to be stable across backward passes. Without it, DDP may rebuild buckets and assign different indices, breaking per-bucket state initialization.

**Delete `__pycache__` after editing `.py` files**
Compute nodes load bytecode from the cache. Stale cache means your edits are silently ignored.
