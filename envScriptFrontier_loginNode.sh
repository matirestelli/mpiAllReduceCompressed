#!/bin/bash
# Source this file:  source ./env_frontier_login.sh
# Unpacks the packed conda env tarball on Lustre (or /tmp) and activates it.

set -u

unset LD_PRELOAD 2>/dev/null || true

module purge
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load cray-mpich
module load rocm/7.1.1
module load gcc/12 
module load libfabric
module load craype-accel-amd-gfx90a
module load miniforge3

# Keep Cray's default loader paths
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- Config (edit if you want different locations) ---
ENVROOT=/lustre/orion/gen243/proj-shared/matilderestelli/pytorch
# ENV_TARBALL="${ENV_TARBALL:-$ENVROOT/conda_env.tar.gz}" old one without torch vision for datasets
ENV_TARBALL="${ENV_TARBALL:-$ENVROOT/conda_env_torch_vision_20260721_hipfix.tar.gz}"

# Where to unpack on login node:
# - Lustre location is persistent and matches your goal.
# - Use a separate prefix so you don't clobber your "build" env unless you explicitly want to.
LOGIN_ENV_PREFIX_DEFAULT="$ENVROOT/conda_from_tar"
LOGIN_ENV_PREFIX="${LOGIN_ENV_PREFIX:-$LOGIN_ENV_PREFIX_DEFAULT}"

# If you REALLY want to replace the old Lustre env in-place, set:
#   export REPLACE_LUSTRE_ENV=1
# before sourcing this script.
REPLACE_LUSTRE_ENV="${REPLACE_LUSTRE_ENV:-0}"
OLD_LUSTRE_ENV="$ENVROOT/conda"
# -----------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"

# Decide where to unpack
if [[ "$REPLACE_LUSTRE_ENV" == "1" ]]; then
  # Replace the old env directory with the tarball contents (destructive).
  if [[ -d "$OLD_LUSTRE_ENV" ]]; then
    echo "[env_frontier_login] REPLACE_LUSTRE_ENV=1: removing old env: $OLD_LUSTRE_ENV" >&2
    rm -rf "$OLD_LUSTRE_ENV"
  fi
  LOGIN_ENV_PREFIX="$OLD_LUSTRE_ENV"
fi

# Unpack tarball if needed (or if forced)
# Set FORCE_UNPACK=1 to re-extract even if directory exists.
FORCE_UNPACK="${FORCE_UNPACK:-0}"

if [[ ! -f "$ENV_TARBALL" ]]; then
  echo "[env_frontier_login] ERROR: env tarball not found: $ENV_TARBALL" >&2
  return 2
fi

if [[ "$FORCE_UNPACK" == "1" || ! -x "$LOGIN_ENV_PREFIX/bin/python" ]]; then
  echo "[env_frontier_login] Extracting $ENV_TARBALL -> $LOGIN_ENV_PREFIX" >&2
  rm -rf "$LOGIN_ENV_PREFIX"
  mkdir -p "$LOGIN_ENV_PREFIX"
  tar -xzf "$ENV_TARBALL" -C "$LOGIN_ENV_PREFIX"
fi

# Activate and fix prefixes inside the unpacked env
conda activate "$LOGIN_ENV_PREFIX"
hash -r
if command -v conda-unpack >/dev/null 2>&1; then
  conda-unpack >/dev/null 2>&1 || true
fi

# Runtime loader fix (needed for torch import on Frontier)
export ROCM_HOME=/opt/rocm-7.1.1
export LIBFABRIC_LIBDIR=/opt/cray/libfabric/2.3.1/lib64
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"


export ROCM_COMPAT_DIR="/tmp/${USER}/rocm_compat_${SLURM_JOB_ID:-login}"
mkdir -p "$ROCM_COMPAT_DIR"
# Torch in this env expects older HIP SONAMEs (libamdhip64.so.5/.6).
# Frontier ROCm provides libamdhip64.so.7, so we alias it.
ln -sf "$ROCM_HOME/lib/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.6"
ln -sf "$ROCM_HOME/lib/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.5"

export LD_LIBRARY_PATH="$ROCM_COMPAT_DIR:$ROCM_HOME/lib:$ROCM_HOME/lib/rocprofiler:$ROCM_HOME/lib/roctracer:$LIBFABRIC_LIBDIR:${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# login-node safety
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# --- Make sure torch shared libs are visible to the dynamic loader ---
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Optional quick sanity check (safe on login: no GPUs expected)
python - <<'PY'
import torch
print("torch:", getattr(torch, "__version__", "<no __version__>"))
print("torch file:", getattr(torch, "__file__", "<no __file__>"))
print("torch.version.hip:", getattr(getattr(torch, "version", None), "hip", None))
PY
