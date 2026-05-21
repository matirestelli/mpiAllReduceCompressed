#!/bin/bash
# =============================================================================
# Polaris DDP + PyTorch (Eagle build) + NCCL (NO GTL / NO GPU-aware MPI)
# Use this env for NCCL-backend runs to isolate GTL overhead.
# =============================================================================

# --- 1. Clear stale preloads ---
unset LD_PRELOAD

# --- 2. Modules (match the build environment) ---
module purge
module use /soft/modulefiles
module load PrgEnv-gnu
module load gcc-native/13
module load cuda/12.9
module load cray-mpich               # provides libmpi (rank discovery only)
module load craype-accel-nvidia80    # activates GPU-aware MPI path
module load conda/2025-09-28

# --- 3. CUDA libs ---
CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib
export LD_LIBRARY_PATH="${CUDA_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- 4. Proxy ---
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="localhost,127.0.0.1,*.local,*.alcf.anl.gov,polaris-*,grand.alcf.anl.gov"

# --- 5. ZFP / cuZFP ---
PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
ZFP_HOME="${PROJECT_ROOT}/zfp-install"
export ZFP_HOME

export PATH="$ZFP_HOME/bin:${PATH}"
export LD_LIBRARY_PATH="$ZFP_HOME/lib64:${LD_LIBRARY_PATH}"
export CPATH="$ZFP_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$ZFP_HOME/lib64:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ZFP_HOME:${CMAKE_PREFIX_PATH:-}"

# --- 6. Conda ---
conda activate base
# Custom MPI-aware PyTorch build disabled — using conda PyTorch (has cuDNN).
# Re-enable for MPI backend / ring hook runs:
# export PYTHONPATH="/lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build/src/pytorch:${PYTHONPATH:-}"

# --- 7. MPI lib path (no GTL, no MPICH_GPU_SUPPORT) ---
MPI_HOME=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3
export LD_LIBRARY_PATH="$MPI_HOME/lib:${LD_LIBRARY_PATH}"

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
echo "  Polaris NCCL test — GTL NOT loaded"
echo "============================================="
echo "Node              : $(hostname)"
echo "GTL               : NOT loaded (NCCL test)"
echo "MPICH_GPU_SUPPORT : NOT set (NCCL test)"
echo ""
echo "PyTorch version   : $(python -c 'import torch; print(torch.__version__)')"
echo "PyTorch location  : $(python -c 'import torch; print(torch.__file__)')"
echo "CUDA available    : $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"
echo "cuDNN             : $(python -c 'import torch; print(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "NOT available")')"
echo "MPI available     : $(python -c 'import torch.distributed as d; print(d.is_mpi_available())')"
echo "mpi4py            : $(python -c 'from mpi4py import MPI; print(MPI.Get_library_version().split(chr(10))[0])')"
echo "mpi4py rank test  : $(mpirun -np 1 python -c 'from mpi4py import MPI; print("rank", MPI.COMM_WORLD.Get_rank(), "OK")')"
echo "ZFP_HOME          : ${ZFP_HOME}"
echo "ZFP lib           : $(ls -l ${ZFP_HOME}/lib64/libzfp.so 2>/dev/null || echo MISSING)"
echo "ZFP include       : $(ls -l ${ZFP_HOME}/include/zfp.h 2>/dev/null || echo MISSING)"
echo "============================================="
