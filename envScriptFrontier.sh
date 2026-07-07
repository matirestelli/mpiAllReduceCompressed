#!/bin/bash
set -euo pipefail
set -x
trap 'echo "FAILED at line $LINENO"; exit 1' ERR


unset LD_PRELOAD
#env script using sbcast to distribute a conda env tarball to each node's local NVMe storage, then activates it.\

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

# If something (modules/startup scripts) left conda active, this won't error now
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

# --- ROCm HIP SONAME compatibility (torch expects libamdhip64.so.5/.6) ---
export ROCM_HOME=${ROCM_HOME:-/opt/rocm-7.1.1}

export ROCM_COMPAT_DIR="${ROCM_COMPAT_DIR:-/tmp/${USER}/rocm_compat_${SLURM_JOB_ID:-login}}"
mkdir -p "$ROCM_COMPAT_DIR"

# Alias ROCm7 libamdhip64.so.7 to older SONAMEs expected by some torch builds
ln -sf "$ROCM_HOME/lib/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.6"
ln -sf "$ROCM_HOME/lib/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.5"

# Ensure the compat dir is searched first
export LD_LIBRARY_PATH="$ROCM_COMPAT_DIR:${LD_LIBRARY_PATH:-}"

# Threading
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1

#proxy
export http_proxy=http://proxy.ccs.ornl.gov:3128
export https_proxy=http://proxy.ccs.ornl.gov:3128
export HTTP_PROXY=http://proxy.ccs.ornl.gov:3128
export HTTPS_PROXY=http://proxy.ccs.ornl.gov:3128

#point to the right zfp-install library that has hip support instead of cuda one
export ZFP_HOME="$HOME/ddp-allreduce-eval-framework/zfp-install-frontier"
export LD_LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$ZFP_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ZFP_HOME:${CMAKE_PREFIX_PATH:-}"

# persistent baseline MIOpen cache directory (Lustre)

#option A: first time using this model and dataset so create the cache with search of miopen kernels (longer time) only using the job file warmup_cache_miopen.sh,
  # 1. Persistent baseline MIOpen cache directory (Lustre)
  #export MIOPEN_BASE=/lustre/orion/gen243/proj-shared/matilderestelli/miopen_cache_baseline
  #mkdir -p "${MIOPEN_BASE}"

  # 2. Node-local cache directory (NVMe, per job)
  #NVME_BASE="/mnt/bb/${USER}"
  #export MIOPEN_LOCAL="${NVME_BASE}/miopen_cache_${SLURM_JOB_ID}"
  #mkdir -p "${MIOPEN_LOCAL}"

  # 3. Copy baseline to local if it exists
  #if [[ -d "${MIOPEN_BASE}" ]]; then
  #  cp -a "${MIOPEN_BASE}/." "${MIOPEN_LOCAL}/" 2>/dev/null || true
  #fi

  # 4. Set up explicit paths (MIOpen expects directories)
  #export MIOPEN_USER_DB_PATH="${MIOPEN_LOCAL}/miopen_db"
  #export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_LOCAL}/kernel_cache"
  #mkdir -p "${MIOPEN_USER_DB_PATH}"
  #mkdir -p "${MIOPEN_CUSTOM_CACHE_DIR}"

  # 5. CRITICAL: Fool MIOpen's shared FS detection
  # Point HOME to our local NVMe space so MIOpen doesn't fallback to /tmp
  #export ORIGINAL_HOME="${HOME}"
  #export HOME="${MIOPEN_LOCAL}"

  # Re-create standard structure expected by frameworks within our fake HOME
  #mkdir -p "${HOME}/.cache/miopen"
  #mkdir -p "${HOME}/.config/miopen"

#------

#option B: if the cache is already created and you want to use it for training, just set the paths to the persistent baseline MIOpen cache directory (Lustre), copy on temp on the node and read only local cache copy because fast needed
# 1. Persistent baseline MIOpen cache directory (Lustre)
export MIOPEN_BASE=/lustre/orion/gen243/proj-shared/matilderestelli/miopen_cache_baseline

# 2. Node-local cache directory (NVMe, per job)
NVME_BASE="/mnt/bb/${USER}"
export MIOPEN_LOCAL="${NVME_BASE}/miopen_cache_${SLURM_JOB_ID}"
mkdir -p "${MIOPEN_LOCAL}"

# 3. Copy baseline from Lustre to local NVMe (Reading into job memory space)
if [[ -d "${MIOPEN_BASE}" ]]; then
  echo "=== Copying golden cache from Lustre to local NVMe ==="
  cp -a "${MIOPEN_BASE}/." "${MIOPEN_LOCAL}/" 2>/dev/null || true
fi

# 4. Set up explicit paths mapping directly to the local copy
export MIOPEN_USER_DB_PATH="${MIOPEN_LOCAL}/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_LOCAL}/kernel_cache"
mkdir -p "${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_CUSTOM_CACHE_DIR}"

# 5. Fool MIOpen's shared FS detection
export ORIGINAL_HOME="${HOME}"
export HOME="${MIOPEN_LOCAL}"
mkdir -p "${HOME}/.cache/miopen"
mkdir -p "${HOME}/.config/miopen"

# 2 = Fast Find mode: Only read from your populated cache databases. 
export MIOPEN_FIND_MODE=2     

# 1 = Force fallback behavior on cache misses. Strictly blocks any on-the-fly 
# tuning threads or exhaustive compilation loops from waking up on your GPUs.
export MIOPEN_FIND_ENFORCE=1
#--------

#if sourcing script in a login node for downloads and need pytohn use this instead of the whole tarball sbcast/unpack:
# conda activate /lustre/orion/gen243/proj-shared/matilderestelli/pytorch/conda

# ---- sbcast + unpack to NVMe (/mnt/bb) ----
: "${SLURM_JOB_ID:?This script should be sourced/run inside a Slurm job allocation}"

# ENV_TARBALL="${ENV_TARBALL:-/lustre/orion/gen243/proj-shared/matilderestelli/pytorch/conda_env.tar.gz}" old tarball without torch vision for datasets
ENV_TARBALL="${ENV_TARBALL:-/lustre/orion/gen243/proj-shared/matilderestelli/pytorch/conda_env_torch_vision_20260701_hipfix.tar.gz}"

NVME_BASE="/mnt/bb/${USER}"
ENV_DIR="/mnt/bb/${USER}/torch_env"
TARBALL_DST="/mnt/bb/${USER}/torch_env.tar.gz"

echo "Using ENV_TARBALL=$ENV_TARBALL"
echo "Using ENV_DIR=$ENV_DIR"

# Ensure base exists (per node) and broadcast tarball
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 mkdir -p "$NVME_BASE"

sbcast -pf "$ENV_TARBALL" "$TARBALL_DST"
if [[ "$?" != "0" ]]; then
  echo "SBCAST failed; aborting to avoid partial env on nodes."
  exit 1
fi

# Make env dir + extract (one task per node)
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 rm -rf "$ENV_DIR"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 mkdir -p "$ENV_DIR"
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 -c "${SLURM_CPUS_PER_TASK}" \
  tar --use-compress-program=pigz -xf "$TARBALL_DST" -C "$ENV_DIR"


# Activate and conda-unpack ON EACH NODE.
# IMPORTANT: conda-unpack must run on the node-local env path, per node.
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  source \$(conda info --base)/etc/profile.d/conda.sh
  conda activate '$ENV_DIR'
  conda-unpack
"
# If you want your *current shell* to use that env too:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

# Runtime loader fix (needed for torch import on Frontier)
export ROCM_HOME=/opt/rocm-7.1.1
export LIBFABRIC_LIBDIR=/opt/cray/libfabric/2.3.1/lib64
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"


# Node-local compat dir (NVMe). /tmp would also work.
export ROCM_COMPAT_DIR="/mnt/bb/${USER}/rocm_compat"
mkdir -p "$ROCM_COMPAT_DIR"
ln -sf "$ROCM_HOME/lib/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.6"

# Prepend compat + ROCm + libfabric (+ keep Cray + existing)
export LD_LIBRARY_PATH="$ROCM_COMPAT_DIR:$ROCM_HOME/lib:$ROCM_HOME/lib/roctracer:$ROCM_HOME/lib/rocprofiler:$LIBFABRIC_LIBDIR:${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# end loader fix 

python - <<'PY'
import torch, torch.distributed as dist
print("PyTorch:", torch.__version__)
print("Location:", torch.__file__)
print("HIP available:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("MPI available:", dist.is_mpi_available())
PY

#to use this then in a bash script:
# #SBATCH -C nvme
# source ./env_frontier_pytorch.sh
#srun -N 8 -n 64 ... python your_script.py