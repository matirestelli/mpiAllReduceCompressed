#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -J torch-build-and-pack
#SBATCH -p extended
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -o torch-build-and-pack.%j.out
#SBATCH -e torch-build-and-pack.%j.err

set -euo pipefail
trap 'echo "[ERROR] Failed at line ${LINENO} at $(date)" >&2' ERR

: <<'PREREQS'
PREREQUISITES (run on a LOGIN node before sbatch):

This script builds a fully independent PyTorch + torchvision + ZFP-extension stack
against MVAPICH-Plus, patches PyTorch source automatically, installs everything
into a dedicated conda env, then conda-packs that env.

Expected existing inputs:
1) NEW isolated root:
     ENVROOT=/lustre/orion/gen243/proj-shared/$USER/pytorch_mvapich

2) Existing PyTorch source checkout:
     SRC_ROOT=/lustre/orion/gen243/proj-shared/$USER/pytorch

3) Existing torchvision source checkout:
     VISION_SRC=/lustre/orion/gen243/proj-shared/$USER/vision_build

4) Existing standalone ZFP extension source:
     EXT_SRC=/ccs/home/$USER/ddp-allreduce-eval-framework/ext_frontier

5) Existing ZFP install:
     ZFP_PREFIX=/ccs/home/$USER/ddp-allreduce-eval-framework/zfp-install-frontier

6) Existing conda env at:
     $ENVROOT/conda

This script will:
- patch PyTorch source for MVAPICH-Plus GPU support detection
- build/install torch wheel
- build/install torchvision
- build/install framework_allreduce_zfp_hip
- conda-pack the env
- optionally unpack locally and verify runtime imports

Outputs:
- wheels:      $ENVROOT/wheels/<jobid>/
- env tarball: $ENVROOT/conda_env.tar.gz
PREREQS

# =============================================================================
# PATHS
# =============================================================================
ENVROOT=/lustre/orion/gen243/proj-shared/matilderestelli/pytorch_mvapich
SRC_ROOT=/lustre/orion/gen243/proj-shared/matilderestelli/pytorch
REPO_DIR="$SRC_ROOT"
VISION_SRC=/lustre/orion/gen243/proj-shared/matilderestelli/vision_build

EXT_SRC=/ccs/home/matilderestelli/ddp-allreduce-eval-framework/ext_frontier
ZFP_PREFIX=/ccs/home/matilderestelli/ddp-allreduce-eval-framework/zfp-install-frontier

ENV_PREFIX="$ENVROOT/conda"
TARBALL="$ENVROOT/conda_env.tar.gz"
COMPAT_DIR="$ENVROOT/compat"

JOBTAG="${SLURM_JOB_ID:-nojobid}"
WHEELS_DIR="$ENVROOT/wheels/${JOBTAG}"

LOCAL_BASE=/tmp/$USER/mvapich
LOCAL_ENV=$LOCAL_BASE/conda
NODE_SCRATCH="/tmp/${USER}/pytorch_mvapich_build_${JOBTAG}"
VIS_SCRATCH="/tmp/${USER}/vision_mvapich_build_${JOBTAG}"
EXT_SCRATCH="/tmp/${USER}/zfpext_mvapich_build_${JOBTAG}"

mkdir -p "$WHEELS_DIR" "$COMPAT_DIR"

stage_tree () {
  local src="$1"
  local dst="$2"
  local label="$3"

  echo "STAGING: $label"
  echo "  SRC=$src"
  echo "  DST=$dst"
  date
  du -sh "$src" || true
  mkdir -p "$dst"

  time rsync -a \
    --delete \
    --info=progress2 \
    --exclude='.git/' \
    --exclude='build/' \
    --exclude='dist/' \
    --exclude='*.egg-info/' \
    --exclude='.eggs/' \
    --exclude='.venv/' \
    --exclude='.conda/' \
    --exclude='.cache/' \
    --exclude='.mypy_cache/' \
    --exclude='.pytest_cache/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.o' \
    --exclude='*.so' \
    --exclude='*.a' \
    "$src/" "$dst/"

  date
  echo "STAGING DONE: $label"
  du -sh "$dst" || true
}

# =============================================================================
# MODULES
# =============================================================================
echo "DEBUG: inherited env before module setup"
env | egrep '^(CONDA|MODULE|LMOD)=' | sort || true

unset CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PREFIX_2 CONDA_PREFIX_3 || true
unset CONDA_SHLVL CONDA_DEFAULT_ENV _CE_CONDA _CE_M CONDA_PROMPT_MODIFIER || true

module --force purge || true
source /opt/cray/pe/cpe/26.03/restore_lmod_system_defaults.sh || true

module load cpe/26.03
module load PrgEnv-gnu
module load gcc-native/14.2
module unload cray-mpich || true
module use /sw/frontier/ums/modulefiles
module load ums038/basic
module load mvapich-plus/4.0-gnu
module load rocm/7.1.1
module load libfabric
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module load miniforge3/23.11.0-0
module unload darshan-runtime || true

source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
unset all_proxy ftp_proxy http_proxy https_proxy no_proxy || true

# =============================================================================
# BUILD ENV
# =============================================================================
export USE_ROCM=1
export USE_CUDA=0
export USE_XPU=0
export USE_DISTRIBUTED=1
export USE_MPI=1
export USE_CUDA_MPI=1
export PYTORCH_ROCM_ARCH=gfx90a
export HCC_AMDGPU_TARGET=gfx90a
export BUILD_TEST=0
export USE_MKLDNN=0
export USE_KINETO=0
export USE_ITT=0
export USE_ROCTRACER=0
export USE_ROCPROFILER=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export USE_TRITON=0
export USE_AOTRITON=0
export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export FC="$(command -v gfortran)"
export CMAKE_C_COMPILER="$CC"
export CMAKE_CXX_COMPILER="$CXX"
export CMAKE_Fortran_COMPILER="$FC"

if ! command -v mpicc >/dev/null 2>&1; then
  echo "ERROR: mpicc not found after loading mvapich-plus/4.0-gnu"
  exit 10
fi

export MPI_HOME="$(dirname "$(dirname "$(command -v mpicc)")")"
export MPI_C_COMPILER="$MPI_HOME/bin/mpicc"
export MPI_CXX_COMPILER="$MPI_HOME/bin/mpicxx"

export CFLAGS="${CFLAGS:-} -Wno-error -Wno-pedantic"
export CXXFLAGS="${CXXFLAGS:-} -Wno-error -Wno-pedantic"

export ROCM_HOME=/opt/rocm-7.1.1
export ROCM_SOURCE_DIR=/opt/rocm-7.1.1
export ROCM_PATH="$ROCM_HOME"
export HIP_PATH="$ROCM_HOME"
export HSA_PATH="$ROCM_HOME"

# Runtime vars for MVAPICH-Plus ROCm
unset MPICH_GPU_SUPPORT_ENABLED || true
unset MV2_USE_CUDA || true
export MV2_USE_ROCM=1
export MV2_SHOW_ENV_INFO=1

echo "MODULE LIST"
module list || true
echo "TOOLCHAIN"
which gcc; "$CC" --version | head -n1 || true
which g++; "$CXX" --version | head -n1 || true
echo "MPI_HOME=$MPI_HOME"
which mpicc; mpicc -show || true
which mpicxx; mpicxx -show || true
which hipcc || true
hipcc --version | head -n 5 || true

# =============================================================================
# ACTIVATE ENV
# =============================================================================
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "ERROR: Env not found at $ENV_PREFIX. Create it on the login node first."
  exit 2
fi

conda activate "$ENV_PREFIX"
hash -r
command -v rsync >/dev/null 2>&1 || { echo "ERROR: rsync not found"; exit 16; }
command -v conda-pack >/dev/null 2>&1 || { echo "ERROR: conda-pack missing in env"; exit 3; }

export PATH="$ENV_PREFIX/bin:$ROCM_HOME/bin:$PATH"
export CMAKE_COMMAND="$ENV_PREFIX/bin/cmake"
export CMAKE_PREFIX_PATH="$MPI_HOME:$ENV_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

ln -sf "$ROCM_HOME/lib/libamdhip64.so.7" "$COMPAT_DIR/libamdhip64.so.6"

TORCH_LIBDIR="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib"
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$COMPAT_DIR:$TORCH_LIBDIR:$MPI_HOME/lib:/opt/cray/libfabric/2.3.1/lib64:$ROCM_HOME/lib:$ROCM_HOME/lib/roctracer:$ROCM_HOME/lib/rocprofiler:$LD_LIBRARY_PATH"

echo "PYTHON/TOOLS IN ENV"
which python
python -V
python -m pip -V
"$CMAKE_COMMAND" --version || true

python - <<'PY'
import numpy, yaml, packaging, typing_extensions
from PIL import Image
print("deps ok")
PY

# =============================================================================
# STAGE PYTORCH SOURCE
# =============================================================================
if [[ ! -f "$REPO_DIR/third_party/pybind11/CMakeLists.txt" ]]; then
  echo "ERROR: submodules look uninitialized in $REPO_DIR."
  echo "Run once on login node: git -C '$REPO_DIR' submodule update --init --recursive"
  exit 11
fi

echo "NODE_SCRATCH=$NODE_SCRATCH"
rm -rf "$NODE_SCRATCH"
mkdir -p "$NODE_SCRATCH"
df -h /tmp || true

echo "SOURCE TREE CHECK"
du -sh "$REPO_DIR" || true
ls -lah "$REPO_DIR" | head -n 50 || true

stage_tree "$REPO_DIR" "$NODE_SCRATCH" "PyTorch source"

cd "$NODE_SCRATCH"
rm -rf build dist
export TMPDIR="$NODE_SCRATCH/tmp"
mkdir -p "$TMPDIR"

# =============================================================================
# PATCH PYTORCH SOURCE FOR MVAPICH-PLUS GPU SUPPORT DETECTION
# =============================================================================
PGMPI_CPP="$NODE_SCRATCH/torch/csrc/distributed/c10d/ProcessGroupMPI.cpp"
if [[ ! -f "$PGMPI_CPP" ]]; then
  echo "ERROR: expected file not found: $PGMPI_CPP"
  exit 20
fi

cp -f "$PGMPI_CPP" "${PGMPI_CPP}.bak"

python - <<PY
from pathlib import Path

p = Path(r"$PGMPI_CPP")
s = p.read_text()

old = '''#elif defined(MPIX_GPU_SUPPORT_CUDA)
  const char* cray_gpu_support = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
  if (cray_gpu_support != nullptr && std::string(cray_gpu_support) == "1") {
    return true;
  } else {
    return false;
  }'''

new = '''#elif defined(MPIX_GPU_SUPPORT_CUDA) || defined(MPIX_GPU_SUPPORT_HIP) || defined(MPIX_GPU_SUPPORT_ZE)
  return MPIX_Query_cuda_support();'''

if new in s:
    print("Patch already present; no changes made.")
elif old in s:
    s = s.replace(old, new, 1)
    p.write_text(s)
    print(f"Patched {p}")
else:
    raise SystemExit("Expected old block not found; patch not applied.")
PY

echo "PATCH VERIFY"
grep -n -A3 -B3 'MPIX_Query_cuda_support\|MPIX_GPU_SUPPORT_HIP\|MPIX_GPU_SUPPORT_ZE' "$PGMPI_CPP" || true

python tools/amd_build/build_amd.py

# =============================================================================
# CMAKE ARGS
# =============================================================================
export CMAKE_ARGS="\
-DUSE_ROCM=ON \
-DROCM_PATH=$ROCM_HOME \
-DHIP_ROOT_DIR=$ROCM_HOME \
-DUSE_MPI=ON \
-DMPI_C_COMPILER=$MPI_C_COMPILER \
-DMPI_CXX_COMPILER=$MPI_CXX_COMPILER \
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
echo "ENVROOT=$ENVROOT  (writes)   SRC_ROOT=$SRC_ROOT (read-only)"
echo "CC=$CC CXX=$CXX"
echo "MPI_HOME=$MPI_HOME"
echo "MPI_C_COMPILER=$MPI_C_COMPILER MPI_CXX_COMPILER=$MPI_CXX_COMPILER"
echo "CMAKE_ARGS=$CMAKE_ARGS"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

export PYTHONUNBUFFERED=1
export NINJA_STATUS="[%f/%t %e] "

echo "PHASE 1: CONFIGURE ONLY"
export ONLY_RUN_CMAKE=1
set +e
MAX_JOBS="$MAX_JOBS" \
USE_ROCM=1 USE_CUDA=OFF USE_DISTRIBUTED=1 USE_MPI=1 USE_CUDA_MPI=1 \
USE_TRITON=0 USE_AOTRITON=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 \
USE_KINETO=0 USE_ITT=0 USE_ROCTRACER=0 USE_ROCPROFILER=0 \
ROCM_SOURCE_DIR="$ROCM_SOURCE_DIR" CMAKE_COMMAND="$CMAKE_COMMAND" CMAKE_ARGS="$CMAKE_ARGS" \
python -u setup.py build -v 2>&1 | tee configure.log
cfg_rc=${PIPESTATUS[0]}
set -e
unset ONLY_RUN_CMAKE
tail -n 120 configure.log || true

[[ $cfg_rc -eq 0 ]] || exit $cfg_rc
[[ -f build/CMakeCache.txt ]] || { echo "ERROR: no CMakeCache.txt"; exit 90; }

echo "PHASE 2: BUILD TORCH WHEEL"
MAX_JOBS="$MAX_JOBS" \
USE_ROCM=1 USE_CUDA=OFF USE_DISTRIBUTED=1 USE_MPI=1 USE_CUDA_MPI=1 \
USE_TRITON=0 USE_AOTRITON=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 \
USE_KINETO=0 USE_ITT=0 USE_ROCTRACER=0 USE_ROCPROFILER=0 \
ROCM_SOURCE_DIR="$ROCM_SOURCE_DIR" CMAKE_COMMAND="$CMAKE_COMMAND" CMAKE_ARGS="$CMAKE_ARGS" \
python -u setup.py bdist_wheel -v 2>&1 | tee build.log

ls -lh dist/*.whl
cp -av dist/*.whl "$WHEELS_DIR"/
python -m pip install --no-deps -U dist/*.whl

cd "$ENVROOT"
python - <<'PY'
import torch, torch.distributed as dist
print("torch:", torch.__version__, "| file:", torch.__file__)
print("hip:", torch.version.hip, "| mpi:", dist.is_mpi_available())
PY

# =============================================================================
# BUILD TORCHVISION
# =============================================================================
echo "PHASE 3: TORCHVISION from $VISION_SRC"
[[ -d "$VISION_SRC" ]] || { echo "ERROR: VISION_SRC not found: $VISION_SRC"; exit 12; }

rm -rf "$VIS_SCRATCH"
mkdir -p "$VIS_SCRATCH"
stage_tree "$VISION_SRC" "$VIS_SCRATCH" "torchvision source"
cd "$VIS_SCRATCH"
rm -rf build dist *.egg-info .eggs
find torchvision/csrc -maxdepth 1 -name 'vision_hip.cpp' -delete || true

export BUILD_VERSION=0.24.0
export FORCE_CUDA=0
export USE_CUDA=0
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx90a
export TORCHVISION_USE_VIDEO_CODEC=0

python -m pip uninstall -y torchvision >/dev/null 2>&1 || true
python -m pip install -v --no-build-isolation --no-deps -U . 2>&1 | tee "$VIS_SCRATCH/vision_build.log"
python -m pip wheel -v --no-build-isolation --no-deps -w "$WHEELS_DIR" . || true

cd "$ENVROOT"
hash -r

python - <<'PY'
import torch, torchvision, torchvision.extension
print("torchvision:", torchvision.__version__)
print("torchvision file:", torchvision.__file__)
print("HAS_OPS:", torchvision.extension._HAS_OPS)
import torchvision.ops
print("torchvision.ops import ok | hip:", torch.version.hip)
print("has nms:", hasattr(torch.ops.torchvision, "nms"))
PY

# =============================================================================
# BUILD ZFP EXTENSION
# =============================================================================
echo "PHASE 4: build framework_allreduce_zfp_hip from $EXT_SRC"
[[ -d "$EXT_SRC" ]] || { echo "ERROR: EXT_SRC not found: $EXT_SRC"; exit 13; }
[[ -d "$ZFP_PREFIX" ]] || { echo "ERROR: ZFP_PREFIX not found: $ZFP_PREFIX"; exit 14; }

if [[ -d "$ZFP_PREFIX/lib64" ]]; then
  ZFP_LIBDIR="$ZFP_PREFIX/lib64"
else
  ZFP_LIBDIR="$ZFP_PREFIX/lib"
fi

ls -l "$ZFP_LIBDIR"/libzfp* || { echo "ERROR: no libzfp in $ZFP_LIBDIR"; exit 15; }

rm -rf "$EXT_SCRATCH"
mkdir -p "$EXT_SCRATCH"
stage_tree "$EXT_SRC" "$EXT_SCRATCH" "ZFP extension source"
cd "$EXT_SCRATCH"
rm -rf build dist *.egg-info .eggs

export ZFP_PREFIX
export ROCM_HOME
python -m pip uninstall -y framework_allreduce_zfp_hip >/dev/null 2>&1 || true
python -m pip install -v --no-build-isolation --no-deps -U . 2>&1 | tee "$EXT_SCRATCH/zfpext_build.log"
python -m pip wheel -v --no-build-isolation --no-deps -w "$WHEELS_DIR" . || true

cd "$ENVROOT"
hash -r

ZFP_EXT_SO="$(python -c 'import framework_allreduce_zfp_hip as m; print(m.__file__)')"
echo "zfp ext .so: $ZFP_EXT_SO"
ldd "$ZFP_EXT_SO" | egrep -i 'zfp|amdhip|mpi|not found' || true
python - <<'PY'
import framework_allreduce_zfp_hip as m
print("zfp ext import ok | file:", m.__file__)
PY

# =============================================================================
# PACK ENV
# =============================================================================
rm -f "$TARBALL"
conda-pack -p "$ENV_PREFIX" -o "$TARBALL"
ls -lh "$TARBALL"

# =============================================================================
# LOCAL UNPACK + RUNTIME VERIFY
# =============================================================================
mkdir -p "$LOCAL_BASE"
rm -rf "$LOCAL_ENV"
mkdir -p "$LOCAL_ENV"

cp "$TARBALL" "$LOCAL_BASE/conda_env.tar.gz"
tar -xzf "$LOCAL_BASE/conda_env.tar.gz" -C "$LOCAL_ENV"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LOCAL_ENV"
conda-unpack

LOCAL_TORCH_LIBDIR="$LOCAL_ENV/lib/python3.10/site-packages/torch/lib"
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$COMPAT_DIR:$LOCAL_TORCH_LIBDIR:$ZFP_LIBDIR:$MPI_HOME/lib:/opt/cray/libfabric/2.3.1/lib64:$ROCM_HOME/lib:$ROCM_HOME/lib/roctracer:$ROCM_HOME/lib/rocprofiler:$LD_LIBRARY_PATH"

cd /tmp
python - <<'PY'
import torch, torchvision, torch.distributed as dist
import framework_allreduce_zfp_hip as zfp_ext
print("RUNTIME torch:", torch.__version__, "| hip:", torch.version.hip)
print("RUNTIME torchvision:", torchvision.__version__)
print("RUNTIME MPI available:", dist.is_mpi_available())
print("RUNTIME has nms:", hasattr(torch.ops.torchvision, "nms"))
print("RUNTIME zfp ext file:", zfp_ext.__file__)
PY

rm -rf "$NODE_SCRATCH" "$VIS_SCRATCH" "$EXT_SCRATCH"
echo "DONE"
