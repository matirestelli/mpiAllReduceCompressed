#!/bin/bash
set -euo pipefail
set -x
trap 'echo "FAILED at line $LINENO"; exit 1' ERR

unset LD_PRELOAD
unset PYTHONPATH
export PYTHONNOUSERSITE=1

module purge || true
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load cray-mpich
module load gcc/12
module load rocm/7.1.1
module load libfabric
module load craype-accel-amd-gfx90a
module load miniforge3

# Initialize conda in this shell so conda activate/deactivate works
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1

# RCCL/Slingshot tuning
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid
export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

# Threading
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# proxy
export http_proxy=http://proxy.ccs.ornl.gov:3128
export https_proxy=http://proxy.ccs.ornl.gov:3128
export HTTP_PROXY=http://proxy.ccs.ornl.gov:3128
export HTTPS_PROXY=http://proxy.ccs.ornl.gov:3128

# ZFP (HIP build)
export ZFP_HOME="$HOME/ddp-allreduce-eval-framework/zfp-install-frontier"
export LD_LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$ZFP_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ZFP_HOME:${CMAKE_PREFIX_PATH:-}"

# ---------------------------------------------------------------------
# MIOpen cache settings (Option A: WARMUP / BUILD cache) -- single node
# ---------------------------------------------------------------------
: "${SLURM_JOB_ID:?This script should be sourced/run inside a Slurm job allocation}"

# Persistent baseline cache (Lustre) that you will update AFTER warmup completes
export MIOPEN_BASE="/lustre/orion/gen243/proj-shared/matilderestelli/miopen_cache_baseline"

# Node-local cache (NVMe) for this job
NVME_BASE="/mnt/bb/${USER}"
export MIOPEN_LOCAL="${NVME_BASE}/miopen_cache_${SLURM_JOB_ID}"

# MIOpen paths pointing at the node-local cache
export MIOPEN_USER_DB_PATH="${MIOPEN_LOCAL}/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_LOCAL}/kernel_cache"

mkdir -p "${NVME_BASE}" "${MIOPEN_LOCAL}" "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}"

# Optional: seed local cache from the current baseline so warmup only adds missing entries
if [[ -d "${MIOPEN_BASE}" ]]; then
  echo "=== Seeding local MIOpen cache from baseline (Lustre -> NVMe) ==="
  cp -a "${MIOPEN_BASE}/." "${MIOPEN_LOCAL}/" 2>/dev/null || true
fi

# Fool MIOpen shared-FS detection: point HOME to local NVMe
export ORIGINAL_HOME="${HOME}"
export HOME="${MIOPEN_LOCAL}"
mkdir -p "${HOME}/.cache/miopen" "${HOME}/.config/miopen"

# Warmup/build behavior: do real Find on misses and allow compilation/tuning
export MIOPEN_FIND_MODE=1        # NORMAL Find (bench solvers)
export MIOPEN_FIND_ENFORCE=1     # allow work on cache misses (populate DB/kernel cache)

# (Optional) reduce logging noise during warmup unless debugging
export MIOPEN_ENABLE_LOGGING=0
unset MIOPEN_LOG_LEVEL
unset MIOPEN_LOG_FILE

# NOTE: After your warmup run finishes, you typically copy results back:
#   mkdir -p "${MIOPEN_BASE}"
#   cp -a "${MIOPEN_LOCAL}/." "${MIOPEN_BASE}/"
# Do that in the warmup JOB SCRIPT (after the python run), not here.

# ---------------------------------------------------------------------
# Local conda env unpack (single node)
# ---------------------------------------------------------------------
ENV_TARBALL="${ENV_TARBALL:-/lustre/orion/gen243/proj-shared/matilderestelli/pytorch/conda_env_torch_vision_20260701_hipfix.tar.gz}"

ENV_DIR="/mnt/bb/${USER}/torch_env"
TARBALL_DST="/mnt/bb/${USER}/torch_env.tar.gz"

echo "Using ENV_TARBALL=$ENV_TARBALL"
echo "Using ENV_DIR=$ENV_DIR"

mkdir -p "$NVME_BASE"
cp -f "$ENV_TARBALL" "$TARBALL_DST"

rm -rf "$ENV_DIR"
mkdir -p "$ENV_DIR"
tar --use-compress-program=pigz -xf "$TARBALL_DST" -C "$ENV_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
conda-unpack

# ---------------------------------------------------------------------
# ROCm HIP SONAME compatibility (single node; keep on NVMe)
# ---------------------------------------------------------------------
export ROCM_HOME="${ROCM_HOME:-/opt/rocm-7.1.1}"
export ROCM_COMPAT_DIR="/mnt/bb/${USER}/rocm_compat_${SLURM_JOB_ID}"

ROCM_LIBDIR="$ROCM_HOME/lib"
[[ -d "$ROCM_LIBDIR" ]] || ROCM_LIBDIR="$ROCM_HOME/lib64"

mkdir -p "${ROCM_COMPAT_DIR}"
ln -sf "${ROCM_LIBDIR}/libamdhip64.so.7" "${ROCM_COMPAT_DIR}/libamdhip64.so.7"
ln -sf "${ROCM_LIBDIR}/libamdhip64.so.7" "${ROCM_COMPAT_DIR}/libamdhip64.so.6"
ln -sf "${ROCM_LIBDIR}/libamdhip64.so.7" "${ROCM_COMPAT_DIR}/libamdhip64.so.5"

export LD_LIBRARY_PATH="${ROCM_COMPAT_DIR}:${CONDA_PREFIX}/lib:${ROCM_LIBDIR}:${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - <<'PY'
import torch, torch.distributed as dist
print("PyTorch:", torch.__version__)
print("Location:", torch.__file__)
print("HIP available:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("MPI available:", dist.is_mpi_available())
PY
