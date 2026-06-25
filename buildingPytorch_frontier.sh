#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -J torch-build-and-pack
#SBATCH -p extended
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -o torch-build-and-pack.%j.out
#SBATCH -e torch-build-and-pack.%j.err

: <<'PREREQS'
PREREQUISITES (run these on a LOGIN node before sbatch):

1) Choose/install your working directory on Lustre (example used by this script):
   ENVROOT=/lustre/orion/gen243/proj-shared/$USER/pytorch

2) PyTorch source checkout must exist at $ENVROOT and be a valid git repo:
   mkdir -p "$ENVROOT"
   git clone --recursive https://github.com/pytorch/pytorch.git "$ENVROOT"
   # or if already cloned:
   # git -C "$ENVROOT" submodule sync --recursive
   # git -C "$ENVROOT" submodule update --init --recursive

3) TorchVision source checkout must exist at $ENVROOT/vision (separate repo):
   git clone https://github.com/pytorch/vision.git "$ENVROOT/vision"
   # Recommended: checkout a torchvision tag compatible with your torch version/commit.

4) A conda environment MUST already exist at $ENVROOT/conda (install target).
   Create it on the login node, then install build/runtime deps (example):
     module load miniforge3/23.11.0-0
     source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
     conda create -y -p "$ENVROOT/conda" python=3.10 pip cmake ninja pkg-config \
       numpy pyyaml typing_extensions packaging
     conda activate "$ENVROOT/conda"
     conda install -y -c conda-forge conda-pack

   Notes:
   - The job will uninstall any existing torch/torchvision in this env and replace them
     with the wheels it builds.
   - The script assumes it can write to $ENVROOT (wheels/, conda_env.tar.gz, etc.).

5) Optional (but used by the script):
   If you want to stage an AOTriton tarball cache, place it at:
     /lustre/orion/gen243/proj-shared/$USER/aotriton_cache/<aotriton tarball name>
   If not present, the script still proceeds because USE_AOTRITON=0.

6) Run the build on a compute node:
   sbatch torch-build-and-pack.sh

Outputs:
- Torch and torchvision wheels saved to: $ENVROOT/wheels/<jobid>/ and $ENVROOT/wheels/latest/
- Packed conda env tarball: $ENVROOT/conda_env.tar.gz
PREREQS


set -euo pipefail
trap 'echo "[ERROR] Failed at line ${LINENO} at $(date)" >&2' ERR

# Paths 
ENVROOT=/lustre/orion/gen243/proj-shared/matilderestelli/pytorch
REPO_DIR="$ENVROOT"                 # pytorch git checkout lives here
ENV_PREFIX="$ENVROOT/conda"         # conda env on Lustre (install target)
TARBALL="$ENVROOT/conda_env.tar.gz" # packed env output on Lustre

LOCAL_BASE=/tmp/$USER
LOCAL_ENV=$LOCAL_BASE/conda

# Node-local build sandbox: build in /tmp, install into ENV_PREFIX on Lustre
JOBTAG="${SLURM_JOB_ID:-nojobid}"
NODE_SCRATCH="/tmp/${USER}/pytorch_build_${JOBTAG}"

# Modules 
module purge
module load cpe/26.03
module load PrgEnv-gnu
module load gcc-native/13.2
module load cray-mpich
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module load miniforge3/23.11.0-0
module unload darshan-runtime || true

# Initialize conda for non-interactive shell
source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh

# Environment variables
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Keep job offline, because probems before where getting stuck in some installations from git on compute nodes 
unset all_proxy ftp_proxy http_proxy https_proxy no_proxy || true

# PyTorch ROCm build knobs
export USE_ROCM=1
export USE_CUDA=0
export USE_XPU=0
export PYTORCH_ROCM_ARCH=gfx90a
export HCC_AMDGPU_TARGET=gfx90a
export BUILD_TEST=0
export USE_MKLDNN=0

# HARD DISABLE KINETO/PROFILER STACK (otherwise stuck in tryint to install and use that)
export USE_KINETO=0
export USE_ITT=0
export USE_ROCTRACER=0
export USE_ROCPROFILER=0

export USE_NNPACK=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0

# Feature disables (avoid Triton/AOTriton/flash-attn -> because otherwise stuck in trying to download/install them ) 
export USE_TRITON=0
export USE_AOTRITON=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0

# Parallelism
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"

# Force Cray compiler wrappers
export CC=cc
export CXX=CC
export FC=ftn
export CMAKE_C_COMPILER=cc
export CMAKE_CXX_COMPILER=CC

export CFLAGS="${CFLAGS:-} -Wno-error -Wno-pedantic"
export CXXFLAGS="${CXXFLAGS:-} -Wno-error -Wno-pedantic"

# ROCm path hints
export ROCM_SOURCE_DIR="${ROCM_HOME:-/opt/rocm}"

echo "MODULE LIST"
module list || true

echo "TOOLCHAIN (before conda activate)"
which cc    || true; cc  --version   || true
which CC    || true; CC  --version   || true
which cmake || true; cmake --version || true

# Activate existing env (install target) 
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "ERROR: Env not found at $ENV_PREFIX. Create/populate it on the login node first."
  exit 2
fi

conda activate "$ENV_PREFIX"

# Force conda tools to win over module/system ones
hash -r
export PATH="$ENV_PREFIX/bin:$PATH"
export CMAKE_COMMAND="$ENV_PREFIX/bin/cmake"
export CMAKE_PREFIX_PATH="$ENV_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

echo " PYTHON/TOOLS IN ENV "
which python
python -V
python -m pip -V
which ninja || true
ninja --version || true
which -a cmake || true
cmake --version || true

# Hard fail early if key deps missing
python - <<'PY'
import numpy, yaml, packaging, typing_extensions
print("deps ok:", "numpy", numpy.__version__)
PY

# Get source ready (Lustre tree) 
cd "$REPO_DIR"
git submodule sync --recursive
git submodule update --init --recursive

# AOTriton persistent cache (on Lustre) 
# Because even if disabling Triton/AOTriton, but keep the staging logic so before dowbload aotriton from a login node and then get that 
AOTRITON_CACHE_DIR=/lustre/orion/gen243/proj-shared/matilderestelli/aotriton_cache
AOTRITON_TARBALL_NAME="aotriton-0.12b-manylinux_2_28_x86_64-rocm7.0-shared.tar.gz"
AOTRITON_TARBALL_CACHE="$AOTRITON_CACHE_DIR/$AOTRITON_TARBALL_NAME"

# Node-local build sandbox (tried on Sophia of ALCF and worked so decided to keep the same structure for Frontier) 
echo " NODE-LOCAL BUILD DIR "
echo "NODE_SCRATCH=$NODE_SCRATCH"
rm -rf "$NODE_SCRATCH"
mkdir -p "$NODE_SCRATCH"
df -h /tmp || true

echo " COPYING SOURCE: Lustre -> /tmp (excluding .git/build/dist) "
tar -C "$REPO_DIR" \
    --exclude='.git' \
    --exclude='build' \
    --exclude='dist' \
    --exclude='**/__pycache__' \
    -cf - . \
  | tar -C "$NODE_SCRATCH" -xf -

cd "$NODE_SCRATCH"
rm -rf build dist

export TMPDIR="$NODE_SCRATCH/tmp"
mkdir -p "$TMPDIR"

# Stage AOTriton tarball into the *local* expected legacy location (under /tmp tree)
AOTRITON_RUNTIME_TGZ_REL="build/aotriton_runtime-prefix/src/$AOTRITON_TARBALL_NAME"
AOTRITON_RUNTIME_TGZ="$NODE_SCRATCH/$AOTRITON_RUNTIME_TGZ_REL"
mkdir -p "$(dirname "$AOTRITON_RUNTIME_TGZ")"

if [[ -f "$AOTRITON_TARBALL_CACHE" ]]; then
  cp -f "$AOTRITON_TARBALL_CACHE" "$AOTRITON_RUNTIME_TGZ"
else
  echo "NOTE: AOTriton tarball cache not found at $AOTRITON_TARBALL_CACHE"
  echo "      Since USE_AOTRITON=0, build should still proceed."
fi

# Optional/informational: verify the kineto source file (should be irrelevant when disabled)
echo " VERIFY KINETO SOURCE FILE (informational)"
grep -n "fmt/format.h\|fmt/core.h" -n torch/csrc/profiler/kineto/profiler_kineto.cpp || true

# ROCm/AMD prep step (run in the local tree)
python tools/amd_build/build_amd.py

# CMake args
export CMAKE_ARGS="\
-DUSE_TRITON=OFF \
-DUSE_AOTRITON=OFF \
-DUSE_FLASH_ATTENTION=OFF \
-DUSE_MEM_EFF_ATTENTION=OFF \
-DUSE_KINETO=OFF \
-DUSE_ITT=OFF \
-DUSE_ROCTRACER=OFF \
-DUSE_ROCPROFILER=OFF \
-DUSE_NNPACK=OFF \
-DUSE_QNNPACK=OFF \
-DUSE_PYTORCH_QNNPACK=OFF \
"

echo " BUILD SETTINGS "
echo "ROCM_SOURCE_DIR=$ROCM_SOURCE_DIR"
echo "CC=$CC CXX=$CXX FC=$FC"
echo "MAX_JOBS=$MAX_JOBS"
echo "PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
echo "CMAKE_COMMAND=$CMAKE_COMMAND"
echo "CMAKE_ARGS=$CMAKE_ARGS"
echo "USE_KINETO=$USE_KINETO USE_ITT=$USE_ITT USE_ROCTRACER=$USE_ROCTRACER USE_ROCPROFILER=$USE_ROCPROFILER"
echo "USE_TRITON=$USE_TRITON USE_AOTRITON=$USE_AOTRITON"
echo "USE_NNPACK=$USE_NNPACK USE_QNNPACK=$USE_QNNPACK USE_PYTORCH_QNNPACK=$USE_PYTORCH_QNNPACK"

# Ensure don't accidentally keep an old torch in the env (casued problems)
python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

# PHASE 1: configure-only + early Kineto OFF check
export PYTHONUNBUFFERED=1
export NINJA_STATUS="[%f/%t %e] "

echo " PHASE 1: CONFIGURE ONLY (generate build/CMakeCache.txt) "
export ONLY_RUN_CMAKE=1

set +e
MAX_JOBS="$MAX_JOBS" \
USE_ROCM=1 USE_CUDA=OFF \
USE_TRITON=0 USE_AOTRITON=0 \
USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 \
USE_KINETO=0 USE_ITT=0 USE_ROCTRACER=0 USE_ROCPROFILER=0 \
ROCM_SOURCE_DIR="$ROCM_SOURCE_DIR" \
CMAKE_COMMAND="$CMAKE_COMMAND" \
CMAKE_ARGS="$CMAKE_ARGS" \
python -u setup.py build -v 2>&1 | tee configure.log
cfg_rc=${PIPESTATUS[0]}
set -e

unset ONLY_RUN_CMAKE

echo " configure.log tail (debug) "
tail -n 120 configure.log || true

if [[ $cfg_rc -ne 0 ]]; then
  echo "ERROR: Configure step failed (rc=$cfg_rc). Not continuing to build."
  exit $cfg_rc
fi

if [[ ! -f build/CMakeCache.txt ]]; then
  echo "ERROR: build/CMakeCache.txt not found after configure step."
  exit 90
fi

echo " EARLY CHECK: Kineto/profiler flags must be OFF in build/CMakeCache.txt "
grep -E "^(USE_KINETO|USE_ITT|USE_ROCTRACER|USE_ROCPROFILER|USE_NNPACK|USE_QNNPACK|USE_PYTORCH_QNNPACK):" -n build/CMakeCache.txt || true

if grep -qE '^USE_KINETO:BOOL=ON$' build/CMakeCache.txt || \
   grep -qE '^USE_ITT:BOOL=ON$' build/CMakeCache.txt || \
   grep -qE '^USE_ROCTRACER:BOOL=ON$' build/CMakeCache.txt || \
   grep -qE '^USE_ROCPROFILER:BOOL=ON$' build/CMakeCache.txt; then
  echo "ERROR: Kineto/profiler stack is ENABLED in CMakeCache (expected OFF)."
  exit 99
fi

echo "OK: Kineto/profiler stack is OFF. Proceeding to wheel build."

#  PHASE 2: build wheel + install into Lustre env 
echo " PHASE 2: BUILD WHEEL "
MAX_JOBS="$MAX_JOBS" \
USE_ROCM=1 USE_CUDA=OFF \
USE_TRITON=0 USE_AOTRITON=0 \
USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 \
USE_KINETO=0 USE_ITT=0 USE_ROCTRACER=0 USE_ROCPROFILER=0 \
ROCM_SOURCE_DIR="$ROCM_SOURCE_DIR" \
CMAKE_COMMAND="$CMAKE_COMMAND" \
CMAKE_ARGS="$CMAKE_ARGS" \
python -u setup.py bdist_wheel -v 2>&1 | tee build.log

# Install wheel into ENV_PREFIX (on Lustre)
python -m pip install -U dist/*.whl

# Sanity check (build node)
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch file:", torch.__file__)
print("torch.version.hip:", torch.version.hip)
print("device_count:", torch.cuda.device_count())
print("is_available:", torch.cuda.is_available())
PY

# Pack env to tarball on Lustre
if ! command -v conda-pack >/dev/null 2>&1; then
  echo "ERROR: conda-pack not in env. Install it on the login node: conda install -c conda-forge conda-pack"
  exit 3
fi

rm -f "$TARBALL"
conda-pack -p "$ENV_PREFIX" -o "$TARBALL"
ls -lh "$TARBALL"

# bcast/unpack test on same node 
mkdir -p "$LOCAL_BASE"
sbcast -f "$TARBALL" "$LOCAL_BASE/conda_env.tar.gz"

rm -rf "$LOCAL_ENV"
mkdir -p "$LOCAL_ENV"
tar -xzf "$LOCAL_BASE/conda_env.tar.gz" -C "$LOCAL_ENV"

# Activate packed env and fix absolute paths
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

# Cleanup node-local build dir 
rm -rf "$NODE_SCRATCH"
