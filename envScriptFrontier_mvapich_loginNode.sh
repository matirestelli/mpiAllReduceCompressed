#!/bin/bash
# Source this file:  source ./env_frontier_login_mvapich.sh
# Unpacks the packed conda env tarball on Lustre and activates it.

set -u

unset LD_PRELOAD 2>/dev/null || true

# Important: reset module system cleanly first
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh

module purge

# UMS prereqs needed for mvapich-plus on Frontier
module load ums/default
module load ums038/basic

# Match your usual login-node stack as closely as possible
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load rocm/7.1.1
module load gcc/12
module load libfabric
module load craype-accel-amd-gfx90a
module load miniforge3

# Load MVAPICH after UMS/basic compiler stack
module load mvapich-plus/4.0-gnu

# Keep Cray/default loader paths
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- Config ---
ENVROOT=/lustre/orion/gen243/proj-shared/matilderestelli/pytorch_mvapich
ENV_TARBALL="${ENV_TARBALL:-$ENVROOT/conda_env.tar.gz}"

LOGIN_ENV_PREFIX_DEFAULT="$ENVROOT/conda_from_tar"
LOGIN_ENV_PREFIX="${LOGIN_ENV_PREFIX:-$LOGIN_ENV_PREFIX_DEFAULT}"

REPLACE_LUSTRE_ENV="${REPLACE_LUSTRE_ENV:-0}"
OLD_LUSTRE_ENV="$ENVROOT/conda"
# --------------

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "$REPLACE_LUSTRE_ENV" == "1" ]]; then
  if [[ -d "$OLD_LUSTRE_ENV" ]]; then
    echo "[env_frontier_login_mvapich] REPLACE_LUSTRE_ENV=1: removing old env: $OLD_LUSTRE_ENV" >&2
    rm -rf "$OLD_LUSTRE_ENV"
  fi
  LOGIN_ENV_PREFIX="$OLD_LUSTRE_ENV"
fi

FORCE_UNPACK="${FORCE_UNPACK:-0}"

if [[ ! -f "$ENV_TARBALL" ]]; then
  echo "[env_frontier_login_mvapich] ERROR: env tarball not found: $ENV_TARBALL" >&2
  return 2
fi

if [[ "$FORCE_UNPACK" == "1" || ! -x "$LOGIN_ENV_PREFIX/bin/python" ]]; then
  echo "[env_frontier_login_mvapich] Extracting $ENV_TARBALL -> $LOGIN_ENV_PREFIX" >&2
  rm -rf "$LOGIN_ENV_PREFIX"
  mkdir -p "$LOGIN_ENV_PREFIX"
  tar -xzf "$ENV_TARBALL" -C "$LOGIN_ENV_PREFIX"
fi

conda activate "$LOGIN_ENV_PREFIX"
hash -r
if command -v conda-unpack >/dev/null 2>&1; then
  conda-unpack >/dev/null 2>&1 || true
fi

# Runtime loader fix
export ROCM_HOME=/opt/rocm-7.1.1
export LIBFABRIC_LIBDIR=/opt/cray/libfabric/2.3.1/lib64
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export ROCM_COMPAT_DIR="/tmp/${USER}/rocm_compat_${SLURM_JOB_ID:-login_mvapich}"
mkdir -p "$ROCM_COMPAT_DIR"
ln -sf "$ROCM_HOME/lib/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.6"
ln -sf "$ROCM_HOME/lib/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.5"

export LD_LIBRARY_PATH="$ROCM_COMPAT_DIR:$ROCM_HOME/lib:$ROCM_HOME/lib/rocprofiler:$ROCM_HOME/lib/roctracer:$LIBFABRIC_LIBDIR:${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# login-node safety
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export ROCR_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Make torch shared libs visible
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Quick sanity check
echo "[env_frontier_login_mvapich] python: $(command -v python)"
echo "[env_frontier_login_mvapich] mpicc : $(command -v mpicc)"
echo "[env_frontier_login_mvapich] mpirun: $(command -v mpirun || true)"
module list 2>&1

python - <<'PY'
import torch
import torch.distributed as dist
print("torch:", getattr(torch, "__version__", "<no __version__>"))
print("torch file:", getattr(torch, "__file__", "<no __file__>"))
print("torch.version.hip:", getattr(getattr(torch, "version", None), "hip", None))
print("mpi available:", dist.is_mpi_available())
PY
