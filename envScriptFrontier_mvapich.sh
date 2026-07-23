#!/bin/bash
set -euo pipefail
set -x
trap 'echo "FAILED at line $LINENO"; exit 1' ERR

unset LD_PRELOAD
unset PYTHONPATH
export PYTHONNOUSERSITE=1

module purge || true
module load cpe/26.03
module load PrgEnv-gnu
module load gcc-native/14.2
module use /sw/frontier/ums/modulefiles
module load ums038/basic
module load mvapich-plus/4.0-gnu
module load rocm/7.1.1
module load libfabric
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module load miniforge3/23.11.0-0

source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
conda deactivate || true

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Cray-MPICH-specific; do not use with MVAPICH-Plus
unset MPICH_GPU_SUPPORT_ENABLED
export MV2_USE_ROCM=1
export MV2_SHOW_ENV_INFO=1

# RCCL / Slingshot tuning
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

# Proxy if needed
export http_proxy=http://proxy.ccs.ornl.gov:3128
export https_proxy=http://proxy.ccs.ornl.gov:3128
export HTTP_PROXY=http://proxy.ccs.ornl.gov:3128
export HTTPS_PROXY=http://proxy.ccs.ornl.gov:3128

# ZFP
export ZFP_HOME="$HOME/ddp-allreduce-eval-framework/zfp-install-frontier"
export LD_LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$ZFP_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ZFP_HOME:${CMAKE_PREFIX_PATH:-}"

: "${SLURM_JOB_ID:?This script should be sourced/run inside a Slurm job allocation}"
: "${SLURM_NNODES:?SLURM_NNODES must be set}"

# MIOpen cache setup
export MIOPEN_BASE=/lustre/orion/gen243/proj-shared/matilderestelli/miopen_cache_baseline
NVME_BASE="/mnt/bb/${USER}"
export MIOPEN_LOCAL="${NVME_BASE}/miopen_cache_${SLURM_JOB_ID}"
export MIOPEN_USER_DB_PATH="${MIOPEN_LOCAL}/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_LOCAL}/kernel_cache"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  mkdir -p '$NVME_BASE' '$MIOPEN_LOCAL' '$MIOPEN_USER_DB_PATH' '$MIOPEN_CUSTOM_CACHE_DIR'
"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  if [[ -d '$MIOPEN_BASE' ]]; then
    cp -a '$MIOPEN_BASE/.' '$MIOPEN_LOCAL/' 2>/dev/null || true
  fi
"

export ORIGINAL_HOME="${HOME}"
export HOME="${MIOPEN_LOCAL}"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  mkdir -p '$HOME/.cache/miopen' '$HOME/.config/miopen'
"

unset MIOPEN_FIND_MODE
unset MIOPEN_FIND_ENFORCE

# Packed env
ENV_TARBALL="${ENV_TARBALL:-/lustre/orion/gen243/proj-shared/matilderestelli/pytorch_mvapich/conda_env.tar.gz}"
ENV_DIR="/mnt/bb/${USER}/torch_env"
TARBALL_DST="/mnt/bb/${USER}/torch_env.tar.gz"

echo "Using ENV_TARBALL=$ENV_TARBALL"
echo "Using ENV_DIR=$ENV_DIR"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 mkdir -p "$NVME_BASE"

sbcast -pf "$ENV_TARBALL" "$TARBALL_DST"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 rm -rf "$ENV_DIR"
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 mkdir -p "$ENV_DIR"
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 -c "${SLURM_CPUS_PER_TASK}" \
  tar --use-compress-program=pigz -xf "$TARBALL_DST" -C "$ENV_DIR"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
  conda activate '$ENV_DIR'
  conda-unpack
"

source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

# ROCm compat
export ROCM_HOME=/opt/rocm-7.1.1
export ROCM_COMPAT_DIR="/mnt/bb/${USER}/rocm_compat_${SLURM_JOB_ID}"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  ROCM_HOME='${ROCM_HOME}'
  ROCM_LIBDIR=\"\$ROCM_HOME/lib\"
  [[ -d \"\$ROCM_LIBDIR\" ]] || ROCM_LIBDIR=\"\$ROCM_HOME/lib64\"
  mkdir -p '${ROCM_COMPAT_DIR}'
  ln -sf \"\$ROCM_LIBDIR/libamdhip64.so.7\" '${ROCM_COMPAT_DIR}/libamdhip64.so.7'
  ln -sf \"\$ROCM_LIBDIR/libamdhip64.so.7\" '${ROCM_COMPAT_DIR}/libamdhip64.so.6'
  ln -sf \"\$ROCM_LIBDIR/libamdhip64.so.7\" '${ROCM_COMPAT_DIR}/libamdhip64.so.5'
"

ROCM_LIBDIR="$ROCM_HOME/lib"
[[ -d "$ROCM_LIBDIR" ]] || ROCM_LIBDIR="$ROCM_HOME/lib64"

TORCH_LIBDIR="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"
export LD_LIBRARY_PATH="$ROCM_COMPAT_DIR:$TORCH_LIBDIR:$CONDA_PREFIX/lib:$ROCM_LIBDIR:/opt/cray/libfabric/2.3.1/lib64:${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - <<'PY'
import torch, torchvision, torch.distributed as dist
print("PyTorch:", torch.__version__)
print("Location:", torch.__file__)
print("torchvision:", torchvision.__version__)
print("HIP available:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("MPI available:", dist.is_mpi_available())
print("has nms:", hasattr(torch.ops.torchvision, "nms"))
PY
