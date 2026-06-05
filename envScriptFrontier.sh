#!/bin/bash
# =============================================================================
# Frontier DDP + PyTorch (source build) + GPU-aware Cray MPICH (env setup)
# =============================================================================

unset LD_PRELOAD

module purge
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load cray-mpich
module load rocm/7.1.1
module load craype-accel-amd-gfx90a

# Non-default CPE fix
export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# GPU-aware MPICH
export MPICH_GPU_SUPPORT_ENABLED=1
# (optional; only if you know you need these)
# export MPICH_MAX_THREAD_SAFETY=multiple

# Proxies (compute nodes)
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

# RCCL/Slingshot tuning (from OLCF guide)
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid

export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3   # fallback: hsn0 if hangs

# Threading
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# MIOpen cache on local storage
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
rm -rf "${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# If using a source tree without installing (like your Polaris PYTHONPATH trick):
# export PYTHONPATH="/path/to/pytorch:${PYTHONPATH:-}"

# If using uv venv:
# source /path/to/venv/bin/activate

echo "============================================="
echo " Frontier PyTorch + Cray MPICH environment"
echo "============================================="
echo "Node              : $(hostname)"
echo "MPICH_GPU_SUPPORT : ${MPICH_GPU_SUPPORT_ENABLED:-<unset>}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-<unset>}"
echo "ROCM_PATH         : ${ROCM_PATH:-<unset>}"
echo ""

python - <<'PY'
import torch
import torch.distributed as d
print("PyTorch version   :", torch.__version__)
print("PyTorch location  :", torch.__file__)
print("ROCm available    :", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("MPI available     :", d.is_mpi_available())
try:
    from mpi4py import MPI
    print("mpi4py            :", MPI.Get_library_version().split("\n")[0])
except Exception as e:
    print("mpi4py            : FAILED", e)
PY

echo "============================================="
