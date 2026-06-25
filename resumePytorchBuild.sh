#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -J Resumetorch-build-and-pack
#SBATCH -p extended
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 04:30:00
#SBATCH -o Resumetorch2-build-and-pack.%j.out
#SBATCH -e Resumetorch2-build-and-pack.%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@300


#what is this script trying to do:
#Allocates a compute node.
#Loads modules, sets proxies.
#Creates/activates a prefix conda env at .../pytorch/conda.
#Updates pip tooling.
#Runs conda install conda-pack (this is where you’re currently sitting).
#Updates git submodules.
#pip install -r requirements.txt (also network unless everything is cached).
#Builds PyTorch from the repo (pip install --no-build-isolation -v .) into that env.
#If build succeeded, conda-pack the env to a tarball and tests unpacking.
#set -euo pipefail

# Auto-requeue handler
requeue_self() {
  echo "[$(date)] Caught USR1; requeuing job $SLURM_JOB_ID to continue..."
  scontrol requeue "$SLURM_JOB_ID" || {
    echo "[$(date)] Requeue failed (permission/site policy)."
  }
}

# 300s before walltime, Slurm sends USR1 (per #SBATCH --signal=...)
trap 'requeue_self; exit 0' USR1

#Paths
ENVROOT=/lustre/orion/gen243/proj-shared/matilderestelli/pytorch
REPO_DIR=$ENVROOT
ENV_PREFIX=$ENVROOT/conda
TARBALL=$ENVROOT/conda_env.tar.gz
CCACHE_DIR=$ENVROOT/ccache

# Build completion flag (prevents repacking/testing if build never finished yet)
BUILD_OK_FLAG="$ENVROOT/.torch_build_ok"

LOCAL_BASE=/tmp/$USER
LOCAL_ENV=$LOCAL_BASE/conda

# Modules / toolchain 
module purge
module load cpe/26.03 PrgEnv-gnu cray-mpich rocm craype-accel-amd-gfx90a gcc-native/13.2 cray-libsci
module load miniforge3
module load ccache 2>/dev/null || true

# Initialize conda for non-interactive shell
source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh

# Environment variables
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# silence Cray warning; safe default
export CRAY_CPU_TARGET=x86-64

# proxy 
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

# PyTorch ROCm build knobs
export USE_ROCM=1
export USE_CUDA=0
export USE_XPU=0
export PYTORCH_ROCM_ARCH=gfx90a
export BUILD_TEST=0
export USE_MKLDNN=0

# Parallelism
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"

# Force Cray compiler wrappers everywhere
export CC=cc
export CXX=CC
export FC=ftn
export CMAKE_C_COMPILER=cc
export CMAKE_CXX_COMPILER=CC

# Keep build artifacts around; do not let pip "clean"
export PIP_NO_CLEAN=1

# Optional speed-up: ccache across jobs
if command -v ccache >/dev/null 2>&1; then
  mkdir -p "$CCACHE_DIR"
  export CCACHE_DIR="$CCACHE_DIR"
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
  ccache -M 50G || true
fi

echo "===== MODULE LIST ====="
module list || true

echo "===== TOOLCHAIN (before conda activate) ====="
which gcc   || true; gcc --version   || true
which cc    || true; cc  --version   || true
which CC    || true; CC  --version   || true
which cmake || true; cmake --version || true

# Create / activate conda env
mkdir -p "$ENVROOT"

# Ensure conda uses proxy + conda-forge
export CONDARC="$ENVROOT/condarc"
cat > "$CONDARC" <<'YAML'
proxy_servers:
  http: http://proxy.ccs.ornl.gov:3128
  https: http://proxy.ccs.ornl.gov:3128
channels:
  - conda-forge
channel_priority: strict
YAML

# Create env only if missing
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.10 pip ninja pkg-config
fi

conda activate "$ENV_PREFIX" || true

# If conda env is broken/incomplete, recreate it
python -c "import site; print('ok')" >/dev/null 2>&1 || {
  echo "Conda env seems broken; recreating..."
  rm -rf "$ENV_PREFIX"
  conda create -y -p "$ENV_PREFIX" python=3.10 pip ninja pkg-config
  conda activate "$ENV_PREFIX"
}

# Prefer system cmake; avoid env cmake wheels
hash -r
export PATH=/usr/bin:$PATH

echo "===== PYTHON ====="
which python
python -V
python -m pip -V

echo "===== CMAKE USED (after conda activate) ====="
which cmake || true
cmake --version || true

python -m pip uninstall -y cmake 2>/dev/null || true
python -m pip install -U pip setuptools wheel

# conda-pack (install once; on later runs should be no-op)
conda install -y -c conda-forge conda-pack || conda install -y -c conda-forge conda-pack

# Get source ready
cd "$REPO_DIR"
git submodule sync --recursive
git submodule update --init --recursive

python -m pip install -r requirements.txt

# Build/install PyTorch into this env
echo "===== BUILD SETTINGS ====="
echo "CC=$CC"
echo "CXX=$CXX"
echo "FC=$FC"
echo "CMAKE_C_COMPILER=$CMAKE_C_COMPILER"
echo "CMAKE_CXX_COMPILER=$CMAKE_CXX_COMPILER"
echo "MAX_JOBS=$MAX_JOBS"
echo "USE_ROCM=$USE_ROCM PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"

if [[ ! -f "$BUILD_OK_FLAG" ]]; then
  echo "===== BUILDING PYTORCH (or resuming) ====="
  python -m pip install --no-build-isolation -v .
  touch "$BUILD_OK_FLAG"
else
  echo "===== BUILD ALREADY COMPLETED (flag exists: $BUILD_OK_FLAG) ====="
fi

#Sanity check + pack only after build completed 
if [[ -f "$BUILD_OK_FLAG" ]]; then
  echo "===== SANITY CHECK ====="
  python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch file:", torch.__file__)
print("torch.version.hip:", torch.version.hip)
print("device_count (ROCm uses cuda namespace):", torch.cuda.device_count())
print("is_available:", torch.cuda.is_available())
PY

  echo "===== PACK ENV TO TARBALL ====="
  rm -f "$TARBALL"
  conda-pack -p "$ENV_PREFIX" -o "$TARBALL"
  ls -lh "$TARBALL"

  echo "===== OPTIONAL: TEST SBCAST/UNPACK ON SAME NODE ====="
  mkdir -p "$LOCAL_BASE"
  sbcast -f "$TARBALL" "$LOCAL_BASE/conda_env.tar.gz"

  rm -rf "$LOCAL_ENV"
  mkdir -p "$LOCAL_ENV"
  tar -xzf "$LOCAL_BASE/conda_env.tar.gz" -C "$LOCAL_ENV"

  source "$LOCAL_ENV/bin/activate"
  conda-unpack

  python - <<'PY'
import torch
import torch.distributed as dist
print("RUNTIME torch:", torch.__version__)
print("RUNTIME torch.version.hip:", torch.version.hip)
print("RUNTIME device_count:", torch.cuda.device_count())
print("RUNTIME is_available:", torch.cuda.is_available())
print("MPI available:", dist.is_mpi_available())
PY
else
  echo "Build not completed yet; skipping pack/test this run."
fi

# show cache stats (optional)
if command -v ccache >/dev/null 2>&1; then
  echo "===== CCACHE STATS ====="
  ccache -s || true
fi
