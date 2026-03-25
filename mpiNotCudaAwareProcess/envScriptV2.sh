#!/bin/bash
# =============================================================================
# Polaris DDP (NCCL) + Manual mpi4py Communication - CUDA-Aware Setup
# =============================================================================
# --- 1. Safety: clear any stale preloads from previous sessions ---
unset LD_PRELOAD
# --- 2. Modules ---
module purge
module use /soft/modulefiles
# Order matters: compiler -> CUDA -> GPU acceleration
module load nvidia/25.5
module load cuda/12.9
module load craype-accel-nvidia80
module load conda/2025-09-28
# --- 3. CUDA libs in LD_LIBRARY_PATH FIRST (prevents shell breakage) ---
CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# --- 3.5. Compiler flags for CUDA-aware MPI (MPIX_GPU_SUPPORT_CUDA) ---
# This enables PyTorch to recognize Cray MPICH as CUDA-aware during compilation
export CFLAGS="-DMPIX_GPU_SUPPORT_CUDA=1"
export CXXFLAGS="-DMPIX_GPU_SUPPORT_CUDA=1"
# --- 4. Proxy ---
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="localhost,127.0.0.1,*.local,*.alcf.anl.gov,polaris-*,grand.alcf.anl.gov"
# --- 5. Conda ---
conda activate base
# --- 6. CUDA-aware MPI (AFTER CUDA libs are resolvable) ---
export MPICH_GPU_SUPPORT_LEVEL=1
export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_CHECK=0
GTL_LIB=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib/libmpi_gtl_cuda.so
if [[ -f "$GTL_LIB" ]]; then
echo "[INFO] GTL found: $GTL_LIB"
export LD_PRELOAD="$GTL_LIB"
else
echo "[WARN] GTL not found — CUDA-aware MPI will fail"
fi
# --- 7. Threading ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1
# --- 8. Slingshot tuning ---
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=software
export LD_PRELOAD=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3/lib/libmpi_gtl_cuda.so
# --- 9. Sanity check ---
echo "============================================="
echo "  Polaris DDP+NCCL+mpi4py Environment Check"
echo "============================================="
echo "Node        : $(hostname)"
echo "GTL         : $GTL_LIB"
echo "LD_PRELOAD  : $LD_PRELOAD"
echo "CFLAGS      : $CFLAGS"
echo "CXXFLAGS    : $CXXFLAGS"
echo "PyTorch     : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail  : $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"
echo "NCCL support: $(python -c 'import torch; has_nccl = hasattr(torch.cuda, "nccl"); print("Yes" if has_nccl else "No")')"
echo "mpi4py      : $(python -c 'from mpi4py import MPI; msg = MPI.Get_library_version(); print(msg.split(chr(10))[0])')"
echo "mpi4py CUDA : $(mpirun -np 1 python -c 'from mpi4py import MPI; print("Init OK")')"
echo "============================================="