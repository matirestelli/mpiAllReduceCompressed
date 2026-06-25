#!/bin/bash
set -euo pipefail

unset LD_PRELOAD
#env script using sbcast to distribute a conda env tarball to each node's local NVMe storage, then activates it.\

module purge
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load cray-mpich
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
module load miniforge3   # or whatever provides conda/conda-unpack

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

#if sourcing script in a login node for downloads and need pytohn use this instead of the whole tarball sbcast/unpack:
# conda activate /lustre/orion/gen243/proj-shared/matilderestelli/pytorch/conda

# ---- sbcast + unpack to NVMe (/mnt/bb) ----
: "${SLURM_JOB_ID:?This script should be sourced/run inside a Slurm job allocation}"

ENV_TARBALL="${ENV_TARBALL:-/lustre/orion/gen243/proj-shared/matilderestelli/pytorch/conda_env.tar.gz}"
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
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 -c56 \
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

# MIOpen cache on local storage
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
rm -rf "${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

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