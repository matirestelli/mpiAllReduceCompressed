#!/bin/bash
# =============================================================================
# Polaris DDP + PyTorch (Eagle build) + CUDA-aware Cray MPICH
# =============================================================================

# --- 1. Clear stale preloads ---
unset LD_PRELOAD

# --- 2. Modules (match the build environment) ---
module purge
module use /soft/modulefiles
module load PrgEnv-gnu
module load gcc-native/13
module load cuda/12.9
module load cray-mpich               # provides libmpi + GTL
module load craype-accel-nvidia80    # activates GPU-aware MPI path
module load conda/2025-09-28

# --- 3. CUDA libs ---
CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- 4. ZFP / cuZFP ---
ZFP_HOME=/eagle/UIC-HPC/mrest/zfp-install
export ZFP_HOME
export PATH="$ZFP_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="$ZFP_HOME/lib64:${LD_LIBRARY_PATH}"
export CPATH="$ZFP_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$ZFP_HOME/lib64:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ZFP_HOME:${CMAKE_PREFIX_PATH:-}"


# --- 5. Proxy ---
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="localhost,127.0.0.1,*.local,*.alcf.anl.gov,polaris-*,grand.alcf.anl.gov"

# --- 6. Conda ---
conda activate base
# The Eagle PyTorch editable install registers itself via a .pth file in
# ~/.local/lib/python3.12/site-packages — no PYTHONPATH override needed.
# If for any reason the wrong torch is picked up, uncomment the line below:
export PYTHONPATH="/lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build/src/pytorch:${PYTHONPATH:-}"

# --- 7. CUDA-aware Cray MPICH ---
export MPICH_GPU_SUPPORT_ENABLED=1   # the actual Polaris enable flag
export MPICH_GPU_SUPPORT_LEVEL=1
export CRAY_ACCEL_TARGET=nvidia80

MPI_HOME=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3
export LD_LIBRARY_PATH="$MPI_HOME/lib:${LD_LIBRARY_PATH}"

GTL_LIB="$MPI_HOME/lib/libmpi_gtl_cuda.so"
if [[ -f "$GTL_LIB" ]]; then
    export LD_PRELOAD="$GTL_LIB"
    echo "[INFO] GTL preloaded: $GTL_LIB"
else
    echo "[WARN] GTL not found at $GTL_LIB — CUDA-aware MPI will not work"
fi

# --- 8. Threading ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# --- 9. Slingshot tuning ---
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_RX_MATCH_MODE=software

# --- 10. Sanity check ---
echo "============================================="
echo "  Polaris DDP + Eagle PyTorch + Cray MPICH"
echo "============================================="
echo "Node              : $(hostname)"
echo "GTL               : ${LD_PRELOAD:-<not set>}"
echo "MPICH_GPU_SUPPORT : $MPICH_GPU_SUPPORT_ENABLED"
echo ""
echo "PyTorch version   : $(python -c 'import torch; print(torch.__version__)')"
echo "PyTorch location  : $(python -c 'import torch; print(torch.__file__)')"
echo "CUDA available    : $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"
echo "MPI available     : $(python -c 'import torch.distributed as d; print(d.is_mpi_available())')"
echo "mpi4py            : $(python -c 'from mpi4py import MPI; print(MPI.Get_library_version().split(chr(10))[0])')"
echo "mpi4py rank test  : $(mpirun -np 1 python -c 'from mpi4py import MPI; print("rank", MPI.COMM_WORLD.Get_rank(), "OK")')"
echo "============================================="