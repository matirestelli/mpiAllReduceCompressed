DDP AllReduce Evaluation Framework

This repository contains the artifact for the paper **"Accelerating MPI
AllReduce Communication with Efficient GPU-Based Compression Schemes on Modern
GPU Clusters"**.  It provides a PyTorch DistributedDataParallel (DDP) training
framework for comparing MPI/NCCL communication hooks, including uncompressed
ring and recursive-doubling AllReduce variants and ZFP-compressed variants.

The framework is designed to answer one question reproducibly: for a fixed
model, batch-size regime, GPU count, and cluster configuration, when does a
custom compressed communication hook improve end-to-end DDP training time?

## Repository Contents

```text
.
|-- interface.py                         # User-facing experiment configuration
|-- config.py                            # Models, data loaders, and TrainingConfig
|-- ddp_training.py                      # DDP training loop, logging, timing
|-- communication_strategy.py            # DDP communication hooks
|-- zfp_cuda.py                          # Python wrapper for cuZFP extension
|-- zfp_cuda_extension.cpp               # C++/CUDA-facing ZFP extension
|-- build_zfp_cuda.sh                    # PBS build script for modified cuZFP
|-- envScript3.sh                        # Polaris environment for CUDA-aware MPI
|-- trainJob.sh                          # PBS sweep template
|-- experiments_results_lr001_GlobalBS128/
|-- experiments_results_lr001_LocalBS128/
`-- plotting/                            # Scripts used to regenerate paper figures
```

Auxiliary notes:

- `BATCH_SIZE.md` explains fixed-global versus fixed-local batch size.
- `TRAINING_STABILITY.md` explains warmup, gradient clipping, and pretrained
  weights for deep CIFAR-10 experiments.
- `logs/` contains debugging notes from hook development.

## Tested Platform

The included job scripts target **Polaris at ALCF**:

- NVIDIA A100 GPUs
- PBS job scheduler
- Cray MPICH with CUDA-aware MPI/GTL
- CUDA 12.9
- PyTorch built with MPI support
- `mpi4py`
- Modified ZFP/cuZFP built with CUDA support

The code is not Polaris-only, but another cluster must provide equivalent
pieces: a GPU-enabled PyTorch build, a distributed backend (`mpi` or `nccl`),
CUDA-aware MPI for the MPI hooks, and a CUDA-enabled ZFP build for compressed
hooks.

## Quick Start

From the repository root:

```bash
cd ddp-allreduce-eval-framework
source envScript3.sh
```

Check that the environment sees CUDA, MPI, and the ZFP install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import torch.distributed as dist; print('MPI available:', dist.is_mpi_available())"
python -c "import zfp_cuda; print('zfp_cuda import OK')"
```

Run a small correctness/training smoke test on 4 GPUs:

```bash
mpiexec -np 4 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    -env LD_PRELOAD="${LD_PRELOAD}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    -env BACKEND=mpi \
    -env COMM_ALGORITHM=default \
    python interface.py
```

Rank 0 writes one `.log` file in the current directory and one CSV file under
`results/`.

## Data and Weights

Experiments use CIFAR-10 by default.  The data loader uses
`download=False`, so the dataset must already exist under:

```text
ddp-allreduce-eval-framework/data/cifar-10-batches-py
```

If the dataset is missing, download CIFAR-10 once in an interactive or login
environment where network access is allowed, or temporarily change the
`download` argument in `config.py` to `True` for setup only.

Some paper configurations use `pretrained=True` in `interface.py`, especially
for `resnext101_32x8d`.  Make sure the corresponding torchvision weights are
available in the local torch cache before submitting jobs on compute nodes
without internet access.

## Build ZFP/cuZFP

The first time you are using this framework on a cluster/system, you should build and install ZFP into a system-specific prefix (do not reuse the present builds).

From the repository root, build ZFP in an out-of-source build directory named build-<newSystem> and install into zfp-install-<newSystem>:

```bash
cd ddp-allreduce-eval-framework/zfp

mkdir -p build-newSystem
cd build-newSystem

cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$PWD/../../zfp-install-newSystem \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

cmake --build . -j
cmake --install .
```

This installs headers and libraries into:

```text
ddp-allreduce-eval-framework/zfp-install-newSystem
```

Configure your environment
In your env script (or before launching jobs), point ZFP_HOME at the install prefix and ensure the runtime loader can find libzfp.so:

```bash
export ZFP_HOME="/path/to/ddp-allreduce-eval-framework/zfp-install-newSystem"
export LD_LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$ZFP_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LIBRARY_PATH:-}"
```

If you compile any components that use CMake, also export:

```bash
export CMAKE_PREFIX_PATH="$ZFP_HOME:${CMAKE_PREFIX_PATH:-}"
```
If import zfp_cuda fails, confirm that ZFP_HOME is correct and that $ZFP_HOME/lib64/libzfp.so (or $ZFP_HOME/lib/libzfp.so) exists in the job environment.

## Configure an Experiment

Most runs are configured in `interface.py` through `TrainingConfig`.

The most important fields are:

```python
TrainingConfig(
    model_name="wide_resnet50_2",      # or "resnext101_32x8d"
    dataset="cifar10",
    num_epochs=20,
    batch_size=32,                    # per-rank batch size
    learning_rate=0.001,
    backend="mpi",
    comm_algorithm=None,               # paper baseline; use "default" for timing wrapper
    zfp_rate=16.0,
    pretrained=True,
    cifar_stem=True,
)
```

Job scripts can override the backend, hook, and ZFP rate without editing
`interface.py`:

```bash
BACKEND=mpi
COMM_ALGORITHM=ring_zfp_online_coll
ZFP_RATE=10
```

These environment variables are read by `apply_job_overrides()` in
`interface.py`.

## Communication Algorithms

Use these values for `COMM_ALGORITHM` or `config.comm_algorithm`:

| Value | Meaning |
|---|---|
| `none` / `None` | No custom hook; DDP built-in AllReduce |
| `default` | DDP built-in AllReduce wrapped for timing/NVTX only |
| `ring` | Custom ring AllReduce using reduce-scatter plus allgather |
| `recursive_doubling` | Custom recursive-doubling AllReduce |
| `ring_zfp_naive` | Ring AllReduce with ZFP compression at each exchange step |
| `recursive_doubling_zfp_naive` | Recursive doubling with ZFP compression |
| `ring_zfp_online_coll` | Ring with collective-level online ZFP compression |
| `recursive_doubling_zfp_online_coll` | Recursive doubling with collective-level online ZFP compression |

For ZFP hooks, set `ZFP_RATE` or `config.zfp_rate`.  The paper results use
rates 16 and 10 for ring online compression, and rates 16 and 8 for recursive
doubling online compression.

## Batch-Size Regimes Used in the Paper

The saved results are organized by batch-size regime:

| Directory | Regime | Per-rank batch size |
|---|---|---|
| `experiments_results_lr001_GlobalBS128/` | Fixed global batch size 128 | `128 / world_size` |
| `experiments_results_lr001_LocalBS128/` | Fixed local batch size 128 | `128` on every rank |

For fixed global batch size 128:

| GPUs | `batch_size` in `interface.py` | Effective batch |
|---:|---:|---:|
| 4 | 32 | 128 |
| 8 | 16 | 128 |
| 16 | 8 | 128 |

For fixed local batch size 128:

| GPUs | `batch_size` in `interface.py` | Effective batch |
|---:|---:|---:|
| 4 | 128 | 512 |
| 8 | 128 | 1024 |
| 16 | 128 | 2048 |

Use the fixed-global setting to reproduce the main `GlobalBS128` figures and
the fixed-local setting to reproduce the `LocalBS128` scaling figures.

## Launch Jobs

### 4 GPUs

Request one Polaris node:

```bash
#PBS -l select=1:ngpus=4
```

Launch:

```bash
mpiexec -np 4 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    -env LD_PRELOAD="${LD_PRELOAD}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    -env BACKEND=mpi \
    -env COMM_ALGORITHM=ring \
    python interface.py
```

### 8 GPUs

Request two Polaris nodes:

```bash
#PBS -l select=2:ngpus=4
```

Launch:

```bash
mpiexec -np 8 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    -env LD_PRELOAD="${LD_PRELOAD}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    -env BACKEND=mpi \
    -env COMM_ALGORITHM=ring_zfp_online_coll \
    -env ZFP_RATE=10 \
    python interface.py
```

### 16 GPUs
Request four Polaris nodes:

```bash
#PBS -l select=4:ngpus=4
```

Launch:

```bash
mpiexec -np 16 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    -env LD_PRELOAD="${LD_PRELOAD}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    -env BACKEND=mpi \
    -env COMM_ALGORITHM=recursive_doubling_zfp_online_coll \
    -env ZFP_RATE=8 \
    python interface.py
```

## Reproduce the Paper Runs

Use `trainJob.sh` as the PBS template.  Edit three things before each sweep:

1. `#PBS -l select=...` for the GPU count.
2. `batch_size` and model settings in `interface.py`.
3. The `EXPERIMENTS` list in `trainJob.sh`.

Recommended hook sweep:

```bash
EXPERIMENTS=(
    "none:"
    "ring:"
    "ring_zfp_naive:16"
    "ring_zfp_online_coll:16"
    "ring_zfp_online_coll:10"
    "recursive_doubling:"
    "recursive_doubling_zfp_naive:16"
    "recursive_doubling_zfp_online_coll:16"
    "recursive_doubling_zfp_online_coll:8"
)
```

Use `none` for the paper baseline.  It produces log filenames containing
`builtin`, which the plotting scripts label as `Baseline`.  Use `default` only
when you specifically want the built-in DDP AllReduce wrapped as a hook for
timing or NVTX instrumentation.

Submit:

```bash
qsub trainJob.sh
```

After each run, move the generated `.log` files into the matching result
directory.  The plotting scripts infer model, GPU count, hook name, and ZFP
rate from the log filenames.

Expected filename pattern:

```text
<model>_<epochs>_<backend>_<algorithm>[_rate<rate>].log
```

Examples:

```text
wide_resnet50_2_20_mpi_builtin.log
wide_resnet50_2_20_mpi_ring_zfp_online_coll_rate10.log
resnext101_32x8d_20_mpi_recursive_doubling_zfp_online_coll_rate8.log
```

## Regenerate Figures from Existing Logs

Run plotting scripts from the repository root unless noted otherwise.
