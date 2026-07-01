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
REPO_DIR="$ENVROOT"
ENV_PREFIX="$ENVROOT/conda"
TARBALL="$ENVROOT/conda_env.tar.gz"

LOCAL_BASE=/tmp/$USER
LOCAL_ENV=$LOCAL_BASE/conda

JOBTAG="${SLURM_JOB_ID:-nojobid}"
NODE_SCRATCH="/tmp/${USER}/pytorch_build_${JOBTAG}"

# Modules
module purge
module load cpe/26.03
module load PrgEnv-gnu
module load gcc-native/13.2
module load cray-mpich
module load rocm/7.1.1
module load libfabric
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module load miniforge3/23.11.0-0
module unload darshan-runtime || true

source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh

# Offline
unset all_proxy ftp_proxy http_proxy https_proxy no_proxy || true

# Environment variables
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export USE_ROCM=1
export USE_CUDA=0
export USE_XPU=0
export USE_DISTRIBUTED=1
export USE_MPI=1

export PYTORCH_ROCM_ARCH=gfx90a
export HCC_AMDGPU_TARGET=gfx90a

export BUILD_TEST=0
export USE_MKLDNN=0

# Disable profiler/kineto stack
export USE_KINETO=0
export USE_ITT=0
export USE_ROCTRACER=0
export USE_ROCPROFILER=0

# Disable mobile/inference extras
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0

# Avoid Triton/aotriton/flash-attn paths
export USE_TRITON=0
export USE_AOTRITON=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0

export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"

# Force Cray compiler wrappers
export CC=cc
export CXX=CC
export FC=ftn
export CMAKE_C_COMPILER=cc
export CMAKE_CXX_COMPILER=CC
export CMAKE_Fortran_COMPILER=ftn

# Force MPI wrappers for FindMPI
export MPI_C_COMPILER=cc
export MPI_CXX_COMPILER=CC
export MPI_Fortran_COMPILER=ftn

export CFLAGS="${CFLAGS:-} -Wno-error -Wno-pedantic"
export CXXFLAGS="${CXXFLAGS:-} -Wno-error -Wno-pedantic"

# Pin ROCm explicitly
export ROCM_HOME=/opt/rocm-7.1.1
export ROCM_SOURCE_DIR=/opt/rocm-7.1.1

echo "MODULE LIST"
module list || true

echo "TOOLCHAIN (before conda activate)"
which cc    || true; cc  --version   || true
which CC    || true; CC  --version   || true
which ftn   || true; ftn --version   || true
which hipcc || true; hipcc --version | head -n 5 || true
which cmake || true; cmake --version || true
echo "MPI tools (before conda activate)"
which mpicc || true
which mpicxx || true
which mpirun || true

# Activate env
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "ERROR: Env not found at $ENV_PREFIX. Create/populate it on the login node first."
  exit 2
fi

conda activate "$ENV_PREFIX"
hash -r

# Prefer conda tools
export PATH="$ENV_PREFIX/bin:$PATH"
export CMAKE_COMMAND="$ENV_PREFIX/bin/cmake"
export CMAKE_PREFIX_PATH="$ENV_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

# ROCm toolchain hints
export ROCM_PATH="$ROCM_HOME"
export HIP_PATH="$ROCM_HOME"
export HSA_PATH="$ROCM_HOME"

# Prefer ROCm tools first
export PATH="$ROCM_HOME/bin:$PATH"

# Library search paths
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/opt/cray/libfabric/2.3.1/lib64:$ROCM_HOME/lib:$ROCM_HOME/lib/roctracer:$ROCM_HOME/lib/rocprofiler:$LD_LIBRARY_PATH"

echo "PYTHON/TOOLS IN ENV"
which python
python -V
python -m pip -V
which ninja || true
ninja --version || true
"$CMAKE_COMMAND" --version || true
echo "MPI tools (after conda activate)"
which mpicc || true
which mpicxx || true
which mpirun || true

python - <<'PY'
import numpy, yaml, packaging, typing_extensions
print("deps ok:", numpy.__version__)
PY

# Source prep
cd "$REPO_DIR"
git submodule sync --recursive
git submodule update --init --recursive

# Local build sandbox
echo "NODE_SCRATCH=$NODE_SCRATCH"
rm -rf "$NODE_SCRATCH"
mkdir -p "$NODE_SCRATCH"
df -h /tmp || true

echo "COPYING SOURCE: Lustre -> /tmp (excluding .git/build/dist)"
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

python tools/amd_build/build_amd.py

# CMake args
export CMAKE_ARGS="\
-DUSE_ROCM=ON \
-DROCM_PATH=$ROCM_HOME \
-DHIP_ROOT_DIR=$ROCM_HOME \
-DUSE_MPI=ON \
-DMPI_C_COMPILER=cc \
-DMPI_CXX_COMPILER=CC \
-DMPI_Fortran_COMPILER=ftn \
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

echo "BUILD SETTINGS"
echo "ROCM_HOME=$ROCM_HOME"
echo "ROCM_SOURCE_DIR=$ROCM_SOURCE_DIR"
echo "CC=$CC CXX=$CXX FC=$FC"
echo "MPI_C_COMPILER=$MPI_C_COMPILER MPI_CXX_COMPILER=$MPI_CXX_COMPILER"
echo "MAX_JOBS=$MAX_JOBS"
echo "PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
echo "CMAKE_COMMAND=$CMAKE_COMMAND"
echo "CMAKE_ARGS=$CMAKE_ARGS"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

# ROCm compat: provide libamdhip64.so.6 at build time
export ROCM_COMPAT_DIR="/tmp/${USER}/rocm_compat_build_${SLURM_JOB_ID:-nojob}"
mkdir -p "$ROCM_COMPAT_DIR"
ln -sf "$ROCM_HOME/lib/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.6"
ls -l "$ROCM_COMPAT_DIR/libamdhip64.so.6" || true
export LD_LIBRARY_PATH="$ROCM_COMPAT_DIR:$LD_LIBRARY_PATH"

export PYTHONUNBUFFERED=1
export NINJA_STATUS="[%f/%t %e] "

echo "PHASE 1: CONFIGURE ONLY"
export ONLY_RUN_CMAKE=1

set +e
MAX_JOBS="$MAX_JOBS" \
USE_ROCM=1 USE_CUDA=OFF \
USE_DISTRIBUTED=1 USE_MPI=1 \
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

tail -n 120 configure.log || true

if [[ $cfg_rc -ne 0 ]]; then
  echo "ERROR: Configure step failed (rc=$cfg_rc)."
  exit $cfg_rc
fi

if [[ ! -f build/CMakeCache.txt ]]; then
  echo "ERROR: build/CMakeCache.txt not found after configure."
  exit 90
fi

echo "EARLY CHECK: key flags from build/CMakeCache.txt"
egrep '^(USE_ROCM|USE_CUDA|USE_DISTRIBUTED|USE_MPI|MPI_FOUND|MPI_C_FOUND|MPI_CXX_FOUND|MPI_C_INCLUDE_PATH|MPI_C_LIBRARIES|MPI_C_COMPILER|MPI_CXX_COMPILER|USE_KINETO|USE_ITT|USE_ROCTRACER|USE_ROCPROFILER):' \
  build/CMakeCache.txt || true

if grep -qE '^USE_ROCM:BOOL=OFF$' build/CMakeCache.txt; then
  echo "ERROR: USE_ROCM is OFF after configure; aborting."
  exit 98
fi

if grep -qE '^USE_MPI:BOOL=OFF$' build/CMakeCache.txt || \
   grep -qE '^MPI_C_FOUND:BOOL=FALSE$' build/CMakeCache.txt || \
   grep -qE '^MPI_CXX_FOUND:BOOL=FALSE$' build/CMakeCache.txt; then
  echo "ERROR: MPI not properly detected in CMakeCache; aborting before wheel build."
  exit 97
fi

if grep -qE '^USE_KINETO:BOOL=ON$' build/CMakeCache.txt || \
   grep -qE '^USE_ITT:BOOL=ON$' build/CMakeCache.txt || \
   grep -qE '^USE_ROCTRACER:BOOL=ON$' build/CMakeCache.txt || \
   grep -qE '^USE_ROCPROFILER:BOOL=ON$' build/CMakeCache.txt; then
  echo "ERROR: profiler stack ENABLED in CMakeCache (expected OFF)."
  exit 99
fi

echo "OK: ROCm is ON, MPI is ON, profiler stack is OFF. Proceeding."

echo "PHASE 2: BUILD WHEEL"
MAX_JOBS="$MAX_JOBS" \
USE_ROCM=1 USE_CUDA=OFF \
USE_DISTRIBUTED=1 USE_MPI=1 \
USE_TRITON=0 USE_AOTRITON=0 \
USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 \
USE_KINETO=0 USE_ITT=0 USE_ROCTRACER=0 USE_ROCPROFILER=0 \
ROCM_SOURCE_DIR="$ROCM_SOURCE_DIR" \
CMAKE_COMMAND="$CMAKE_COMMAND" \
CMAKE_ARGS="$CMAKE_ARGS" \
python -u setup.py bdist_wheel -v 2>&1 | tee build.log

echo "WHEELS:"
ls -lh dist/*.whl

python -m pip install -U dist/*.whl

echo "INSTALLED TORCH:"
python -m pip show torch || true
python -m pip freeze | egrep '^torch(==| @ )' || true

python - <<'PY'
import torch
import torch.distributed as dist
print("torch version:", torch.__version__)
print("torch.file:", torch.__file__)
print("torch.version.hip:", torch.version.hip)
print("device_count:", torch.cuda.device_count())
print("is_available:", torch.cuda.is_available())
print("MPI available:", dist.is_mpi_available())
PY

echo "LDD CHECK torch/_C*.so (must NOT show 'not found')"
ldd "$ENV_PREFIX"/lib/python3.10/site-packages/torch/_C*.so | egrep "not found|libfabric|amdhip|rocm_smi" || true

if ! command -v conda-pack >/dev/null 2>&1; then
  echo "ERROR: conda-pack not in env. Install it on login node: conda install -c conda-forge conda-pack"
  exit 3
fi

rm -f "$TARBALL"
conda-pack -p "$ENV_PREFIX" -o "$TARBALL"
ls -lh "$TARBALL"

mkdir -p "$LOCAL_BASE"
sbcast -f "$TARBALL" "$LOCAL_BASE/conda_env.tar.gz"

rm -rf "$LOCAL_ENV"
mkdir -p "$LOCAL_ENV"
tar -xzf "$LOCAL_BASE/conda_env.tar.gz" -C "$LOCAL_ENV"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LOCAL_ENV"
conda-unpack

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/opt/cray/libfabric/2.3.1/lib64:$ROCM_HOME/lib:$ROCM_HOME/lib/roctracer:$ROCM_HOME/lib/rocprofiler:$LD_LIBRARY_PATH"

echo "MPI toolchain sanity (runtime test env)"
which mpicc || true
which mpicxx || true
which mpirun || true
cc --craype-verbose 2>/dev/null | head -n 50 || true

python - <<'PY'
import torch
import torch.distributed as dist
print("RUNTIME torch:", torch.__version__)
print("RUNTIME torch.version.hip:", torch.version.hip)
print("RUNTIME device_count:", torch.cuda.device_count())
print("RUNTIME is_available:", torch.cuda.is_available())
print("MPI available:", dist.is_mpi_available())
PY

rm -rf "$NODE_SCRATCH"