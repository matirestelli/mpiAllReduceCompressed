# mpiAllReduceCompressed

## Overview

This project implements and benchmarks the paper **"Accelerating MPI AllReduce Communication with Efficient GPU-Based Compression Schemes on Modern GPU Clusters"**.

Target hardware: **Polaris** at ALCF (Argonne Leadership Computing Facility) — 4× NVIDIA A100 per node, Cray MPICH with GPU-direct RDMA.

---

## Communication Hooks

All hooks live in `communication_strategy.py` and are registered via `model.register_comm_hook()` in `ddp_training.py`. Select a hook by setting `comm_algorithm` in `TrainingConfig`.

| Algorithm name | State class | What it does |
|---|---|---|
| `default` | — | Wraps DDP's built-in `dist.all_reduce` (MPI `MPI_Iallreduce` or NCCL). Baseline — no custom routing. |
| `ring` | `RingAllreduceState` | Custom ring allreduce using uncompressed point-to-point MPI sends/recvs. Reduce-scatter phase (nranks−1 steps) then allgather phase (nranks−1 steps). Matches the advisor's C++ reference exactly. |
| `ring_zfp` | `ZfpRingAllreduceState` | Ring allreduce where every P2P message is ZFP-compressed on GPU (cuZFP fixed-rate). Two-round exchange per step: round 1 sends the byte count (int64), round 2 sends the compressed payload at the exact received size. |
| `ring_zfp_naive` | `ZfpRingAllreduceState` | Alias for `ring_zfp`. |
| `recursive_doubling` | `RecursiveDoublingAllreduceState` | Recursive-doubling (binary-tree) allreduce using point-to-point MPI. Handles non-power-of-two ranks via a peel/finalize step. |
| `recursive_doubling_zfp` | `ZfpRecursiveDoublingAllreduceState` | Recursive-doubling allreduce where every P2P message is ZFP-compressed. Same two-round protocol as `ring_zfp`. |
| `recursive_doubling_zfp_naive` | `ZfpRecursiveDoublingAllreduceState` | Alias for `recursive_doubling_zfp`. |

### ZFP Compression Notes

- Compression is performed by **cuZFP** — the official CUDA backend of [libzfp](https://github.com/LLNL/zfp) (`zfp_exec_cuda`, fixed-rate mode).
- Requires libzfp built with `ZFP_WITH_CUDA=ON`. Set `ZFP_HOME` to the install prefix.
- The Python wrapper is `zfp_cuda.py`; the C++ extension source is `zfp_cuda_extension.cpp`.
- Compression rate is controlled by the `DDP_ZFP_RATE` environment variable (default: `8.0` bits/value).
- All MPI-facing compressed buffers are persistent GPU allocations — never reallocated between backward passes, as required by Cray MPICH's RDMA registration cache.

---

## Implementation Scope

1. **Baseline**: Standard MPI/NCCL AllReduce via DDP's built-in collective (`default` hook).

2. **Custom routing algorithms** (implemented from scratch to match the advisor's C++ reference):
   - Ring allreduce (`ring`)
   - Recursive doubling allreduce (`recursive_doubling`)

3. **GPU-compressed variants**: Ring and recursive-doubling with ZFP fixed-rate compression on every P2P message (`ring_zfp`, `recursive_doubling_zfp`).

4. **Benchmarking**: All hooks are benchmarked for communication latency, throughput, compression ratio, and end-to-end training time on ResNet50/CIFAR-10.

---

## Project Files

| File | Purpose |
|---|---|
| `interface.py` | Entry point — configure `TrainingConfig` and call `train()` |
| `config.py` | `TrainingConfig` dataclass, `create_model()`, `get_data_loaders()` |
| `ddp_training.py` | Main training loop, DDP init, hook registration |
| `communication_strategy.py` | **All AllReduce hooks** — ring, recursive doubling, ZFP variants |
| `zfp_cuda.py` | Python wrapper that JIT-builds the cuZFP extension |
| `zfp_cuda_extension.cpp` | C++ PyTorch extension calling libzfp with `zfp_exec_cuda` |
| `ring_allreduce_cuda.cu` | C++/CUDA extension — direct GPU port of advisor's C++ ring code |
| `ring_allreduce_cpp_hook.py` | Python DDP wrapper for the compiled C++ extension |
| `setup.py` | Build script for `ring_allreduce_cuda_ext` |
| `toy_train.py` | Correctness + timing test with analytically predictable gradients |
