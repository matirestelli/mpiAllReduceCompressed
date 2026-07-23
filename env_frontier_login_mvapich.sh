#!/bin/bash
# Source this file:
#   source ./env_frontier_login_mvapich.sh
#
# Purpose:
# - load the MVAPICH-Plus/ROCm module stack
# - unpack the new packed env on the login node if needed
# - activate it
# - set ROCm compat symlinks + loader paths
# - allow safe import/runtime checks from the login node

set -u

unset LD_PRELOAD 2>/dev/null || true
unset PYTHONPATH 2>/dev/null || true
export PYTHONNOUSERSITE=1

module purge
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
conda deactivate >/dev/null 2>&1 || true

# Keep Cray/default loader paths
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ENVROOT=/lustre/orion/gen243/proj-shared/matilderestelli/pytorch_mvapich
ENV_TARBALL="${ENV_TARBALL:-$ENVROOT/conda_env.tar.gz}"

# Default unpack location on login node
LOGIN_ENV_PREFIX_DEFAULT="$ENVROOT/conda_from_tar"
LOGIN_ENV_PREFIX="${LOGIN_ENV_PREFIX:-$LOGIN_ENV_PREFIX_DEFAULT}"

# If you really want to replace the existing Lustre env in-place:
#   export REPLACE_LUSTRE_ENV=1
# before sourcing
REPLACE_LUSTRE_ENV="${REPLACE_LUSTRE_ENV:-0}"
OLD_LUSTRE_ENV="$ENVROOT/conda"

# Force re-extract:
#   export FORCE_UNPACK=1
FORCE_UNPACK="${FORCE_UNPACK:-0}"

# ZFP install used by your extension
export ZFP_HOME="${ZFP_HOME:-$HOME/ddp-allreduce-eval-framework/zfp-install-frontier}"

# -----------------------------------------------------------------------------
# Decide where to unpack
# -----------------------------------------------------------------------------
if [[ "$REPLACE_LUSTRE_ENV" == "1" ]]; then
  if [[ -d "$OLD_LUSTRE_ENV" ]]; then
    echo "[env_frontier_login_mvapich] REPLACE_LUSTRE_ENV=1: removing old env: $OLD_LUSTRE_ENV" >&2
    rm -rf "$OLD_LUSTRE_ENV"
  fi
  LOGIN_ENV_PREFIX="$OLD_LUSTRE_ENV"
fi

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

# -----------------------------------------------------------------------------
# Activate env
# -----------------------------------------------------------------------------
conda activate "$LOGIN_ENV_PREFIX"
hash -r

if command -v conda-unpack >/dev/null 2>&1; then
  conda-unpack >/dev/null 2>&1 || true
fi

# -----------------------------------------------------------------------------
# Runtime paths
# -----------------------------------------------------------------------------
export ROCM_HOME=/opt/rocm-7.1.1
export LIBFABRIC_LIBDIR=/opt/cray/libfabric/2.3.1/lib64

# MVAPICH path info
if command -v mpicc >/dev/null 2>&1; then
  export MPI_HOME="$(dirname "$(dirname "$(command -v mpicc)")")"
else
  echo "[env_frontier_login_mvapich] WARNING: mpicc not found after module load" >&2
  export MPI_HOME=""
fi

# ROCm compat dir for torch HIP soname expectations
export ROCM_COMPAT_DIR="/tmp/${USER}/rocm_compat_login_mvapich"
mkdir -p "$ROCM_COMPAT_DIR"

ROCM_LIBDIR="$ROCM_HOME/lib"
[[ -d "$ROCM_LIBDIR" ]] || ROCM_LIBDIR="$ROCM_HOME/lib64"

ln -sf "$ROCM_LIBDIR/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.7"
ln -sf "$ROCM_LIBDIR/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.6"
ln -sf "$ROCM_LIBDIR/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.5"

TORCH_LIBDIR="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib"

# ZFP
export LD_LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$ZFP_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ZFP_HOME:${CMAKE_PREFIX_PATH:-}"

# Final loader path
export LD_LIBRARY_PATH="$ROCM_COMPAT_DIR:$TORCH_LIBDIR:$CONDA_PREFIX/lib:${MPI_HOME:+$MPI_HOME/lib:}$ROCM_LIBDIR:$ROCM_HOME/lib/rocprofiler:$ROCM_HOME/lib/roctracer:$LIBFABRIC_LIBDIR:${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# -----------------------------------------------------------------------------
# Login-node safety
# -----------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1

unset MPICH_GPU_SUPPORT_ENABLED
unset MIOPEN_FIND_MODE
unset MIOPEN_FIND_ENFORCE

# -----------------------------------------------------------------------------
# Helpful info
# -----------------------------------------------------------------------------
echo "[env_frontier_login_mvapich] CONDA_PREFIX=$CONDA_PREFIX"
echo "[env_frontier_login_mvapich] ENV_TARBALL=$ENV_TARBALL"
echo "[env_frontier_login_mvapich] MPI_HOME=${MPI_HOME:-<unset>}"
echo "[env_frontier_login_mvapich] TORCH_LIBDIR=$TORCH_LIBDIR"

# -----------------------------------------------------------------------------
# Quick sanity check (safe on login node)
# -----------------------------------------------------------------------------
python - <<'PY'
import torch, torchvision, torch.distributed as dist
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("torch.version.hip:", getattr(torch.version, "hip", None))
print("torchvision:", torchvision.__version__)
print("mpi available:", dist.is_mpi_available())
print("has nms:", hasattr(torch.ops.torchvision, "nms"))
PY
