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

unset MPICH_GPU_SUPPORT_ENABLED

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

# point to the right zfp-install library that has hip support instead of cuda one
export ZFP_HOME="$HOME/ddp-allreduce-eval-framework/zfp-install-frontier"
export LD_LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$ZFP_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ZFP_HOME:${CMAKE_PREFIX_PATH:-}"

# ---------------------------------------------------------------------
# MIOpen cache settings (Option B) -- FIXED for multi-node
# ---------------------------------------------------------------------

# 1. Persistent baseline MIOpen cache directory (Lustre)
export MIOPEN_BASE=/lustre/orion/gen243/proj-shared/matilderestelli/miopen_cache_baseline

# 2. Node-local cache directory (NVMe, per job)
: "${SLURM_JOB_ID:?This script should be sourced/run inside a Slurm job allocation}"
: "${SLURM_NNODES:?SLURM_NNODES must be set}"

NVME_BASE="/mnt/bb/${USER}"
export MIOPEN_LOCAL="${NVME_BASE}/miopen_cache_${SLURM_JOB_ID}"

# 4. Explicit paths mapping directly to the local copy
export MIOPEN_USER_DB_PATH="${MIOPEN_LOCAL}/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_LOCAL}/kernel_cache"

# IMPORTANT: Create directories on EVERY NODE (not just the first node)
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  mkdir -p '$NVME_BASE' \
           '$MIOPEN_LOCAL' \
           '$MIOPEN_USER_DB_PATH' \
           '$MIOPEN_CUSTOM_CACHE_DIR'
"

# 3. Copy baseline from Lustre to local NVMe (do it per node)
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  if [[ -d '$MIOPEN_BASE' ]]; then
    echo '=== Copying golden cache from Lustre to local NVMe on ' \$(hostname) ' ==='
    cp -a '$MIOPEN_BASE/.' '$MIOPEN_LOCAL/' 2>/dev/null || true
  fi
"

# 5. Fool MIOpen's shared FS detection (set HOME to local path)
export ORIGINAL_HOME="${HOME}"
export HOME="${MIOPEN_LOCAL}"

# Ensure standard expected dirs exist on EVERY NODE as well
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  mkdir -p '$HOME/.cache/miopen' '$HOME/.config/miopen'
"

# ---------------------------------------------------------------------
# MIOpen cache settings
# MIOpen provides a set of Find modes which are used to accelerate the Find calls. The different modes are set by using the environment variable MIOPEN_FIND_MODE, and setting it to one of the values:
# NORMAL, or 1: Normal Find: This is the full Find mode call, which will benchmark all the solvers and return a list
# FAST, or 2: Fast Find: Checks the Find-Db for an entry. If there is a Find-Db hit, use that entry. If there is a miss, utilize the Immediate mode fallback. If Start-up times are expected to be faster, but worse GPU performance.
# HYBRID, or 3, or unset MIOPEN_FIND_MODE: Hybrid Find: Checks the Find-Db for an entry. If there is a Find-Db hit, use that entry. If there is a miss, use the existing Find machinery. Slower start-up times than Fast Find, but no GPU performance drop.
# 4: This value is reserved and should not be used.
# DYNAMIC_HYBRID, or 5: Dynamic Hybrid Find: Checks the Find-Db for an entry. If there is a Find-Db hit, uses that entry. If there is a miss, uses the existing Find machinery with skipping non-dynamic kernels. Faster start-up times than Hybrid Find, but GPU performance may be a bit worse.
# ---------------------------------------------------------------------

# Post-tuning: leave these unset now that MIOPEN_USER_DB_PATH is fully tuned
# (per AMD's tuning-database docs — see link below)
unset MIOPEN_FIND_MODE
unset MIOPEN_FIND_ENFORCE

# ---------------------------------------------------------------------
# sbcast + unpack conda env to NVMe (/mnt/bb) ... (rest of your script)
# ---------------------------------------------------------------------

: "${SLURM_JOB_ID:?This script should be sourced/run inside a Slurm job allocation}"

ENV_TARBALL="${ENV_TARBALL:-/lustre/orion/gen243/proj-shared/matilderestelli/pytorch/conda_env_torch_vision_20260701_hipfix.tar.gz}"

ENV_DIR="/mnt/bb/${USER}/torch_env"
TARBALL_DST="/mnt/bb/${USER}/torch_env.tar.gz"

echo "Using ENV_TARBALL=$ENV_TARBALL"
echo "Using ENV_DIR=$ENV_DIR"

# Ensure base exists (per node) and broadcast tarball
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 mkdir -p "$NVME_BASE"

sbcast -pf "$ENV_TARBALL" "$TARBALL_DST"

# Make env dir + extract (one task per node)
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 rm -rf "$ENV_DIR"
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 mkdir -p "$ENV_DIR"
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 -c "${SLURM_CPUS_PER_TASK}" \
  tar --use-compress-program=pigz -xf "$TARBALL_DST" -C "$ENV_DIR"

# Activate and conda-unpack ON EACH NODE
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  source \$(conda info --base)/etc/profile.d/conda.sh
  conda activate '$ENV_DIR'
  conda-unpack
"

# Activate in current shell too
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# ---------------------------------------------------------------------
# ROCm HIP SONAME compatibility (must exist on EVERY NODE; /mnt/bb is local)
# ---------------------------------------------------------------------
export ROCM_HOME="${ROCM_HOME:-/opt/rocm-7.1.1}"
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

# Ensure the dynamic loader finds it (must be set before importing torch)
ROCM_LIBDIR="$ROCM_HOME/lib"
[[ -d "$ROCM_LIBDIR" ]] || ROCM_LIBDIR="$ROCM_HOME/lib64"

export LD_LIBRARY_PATH="$ROCM_COMPAT_DIR:$CONDA_PREFIX/lib:$ROCM_LIBDIR:${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - <<'PY'
import os
import torch, torch.distributed as dist
print("PyTorch:", torch.__version__)
print("Location:", torch.__file__)
print("HIP available:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("MPI available:", dist.is_mpi_available())
print("MPICH_GPU_SUPPORT_ENABLED:", os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "<unset>"))
PY
