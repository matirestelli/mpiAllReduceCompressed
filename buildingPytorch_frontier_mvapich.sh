#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -J torch-build-and-pack-mvapich
#SBATCH -p extended
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 06:00:00
#SBATCH -o torch-build-and-pack-mvapich.%j.out
#SBATCH -e torch-build-and-pack-mvapich.%j.err

: <<'PREREQS'
PREREQUISITES (run on a LOGIN node before sbatch):

This script builds a SECOND, fully independent PyTorch+torchvision stack against
MVAPICH-Plus. It NEVER touches your existing working Cray-MPICH env.

1) NEW isolated root (everything the job WRITES goes here):
     ENVROOT=/lustre/orion/gen243/proj-shared/$USER/pytorch_mvapich
   Create it:  mkdir -p "$ENVROOT"

2) It REUSES (read-only) your existing PyTorch source checkout:
     SRC_ROOT=/lustre/orion/gen243/proj-shared/$USER/pytorch
   The job copies from here into /tmp and does NOT run git in it, so your other
   build is untouched. Submodules must already be populated (they are, since your
   Cray build worked). If ever needed once:
     git -C "$SRC_ROOT" submodule update --init --recursive

3) It REUSES (read-only) your existing, ALREADY-FIXED torchvision checkout
   (the one that carries your setup.py vision_hip.cpp exclusion + any zfp edits):
     VISION_SRC=/lustre/orion/gen243/proj-shared/$USER/vision_build   # <-- CONFIRM THIS PATH

4) A NEW conda env MUST already exist at $ENVROOT/conda. Create it on the login node:
     module load miniforge3/23.11.0-0
     source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
     conda create -y -p "$ENVROOT/conda" python=3.10 pip cmake ninja pkg-config \
       numpy pyyaml typing_extensions packaging
     conda activate "$ENVROOT/conda"
     conda install -y -c conda-forge conda-pack

5) The job replaces torch/torchvision in THIS env only, then conda-packs it.

6) It also REBUILDS your standalone zfp C++ extension (framework_allreduce_zfp_hip)
   against the new torch, reusing your existing stream-aware ZFP install as-is:
     EXT_SRC    = your ext_frontier folder (setup.py + csrc/)   <-- CONFIRM path below
     ZFP_PREFIX = your zfp-install-frontier dir                 <-- CONFIRM path below
   ZFP itself is MPI-agnostic and is NOT rebuilt. The extension bakes an rpath to
   $ZFP_PREFIX/lib(64) (an absolute path that stays valid on compute nodes), so
   nothing is copied into the env.

Outputs:
- torch + torchvision + zfp-ext wheels: $ENVROOT/wheels/<jobid>/
- packed env tarball:                   $ENVROOT/conda_env.tar.gz
PREREQS


set -euo pipefail
trap 'echo "[ERROR] Failed at line ${LINENO} at $(date)" >&2' ERR

# =============================================================================
# PATHS  (all WRITE paths point at the NEW mvapich root; SOURCE paths are reused)
# =============================================================================
ENVROOT=/lustre/orion/gen243/proj-shared/matilderestelli/pytorch_mvapich

# Reuse the EXISTING PyTorch source checkout (read-only from this job's POV)
SRC_ROOT=/lustre/orion/gen243/proj-shared/matilderestelli/pytorch
REPO_DIR="$SRC_ROOT"

# Reuse the EXISTING, already-fixed torchvision checkout  (<-- confirm path)
VISION_SRC=/lustre/orion/gen243/proj-shared/matilderestelli/vision_build

# --- custom ZFP (HIP) extension -------------------------------------------------
# The zfp C++ extension source (your ext_frontier folder, contains setup.py + csrc/)
EXT_SRC=/ccs/home/matilderestelli/ddp-allreduce-eval-framework/ext_frontier   
# The already-built, stream-aware ZFP install (headers + libzfp in lib/ or lib64/).
# MPI-agnostic — reused as-is, NOT rebuilt.
ZFP_PREFIX=/ccs/home/matilderestelli/ddp-allreduce-eval-framework/zfp-install-frontier 
# --------------------------------------------------------------------------------

ENV_PREFIX="$ENVROOT/conda"
TARBALL="$ENVROOT/conda_env.tar.gz"

JOBTAG="${SLURM_JOB_ID:-nojobid}"
WHEELS_DIR="$ENVROOT/wheels/${JOBTAG}"

# node-local scratch, namespaced so it can't collide with a concurrent Cray build
LOCAL_BASE=/tmp/$USER/mvapich
LOCAL_ENV=$LOCAL_BASE/conda
NODE_SCRATCH="/tmp/${USER}/pytorch_mvapich_build_${JOBTAG}"
VIS_SCRATCH="/tmp/${USER}/vision_mvapich_build_${JOBTAG}"

mkdir -p "$WHEELS_DIR"

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
    --exclude='.mypy_cache/' \
    --exclude='.pytest_cache/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    "$src/" "$dst/"

  date
  echo "STAGING DONE: $label"
  du -sh "$dst" || true
}

# =============================================================================
# MODULES  (MVAPICH-Plus instead of cray-mpich)
# =============================================================================
module purge
module load cpe/26.03
module load PrgEnv-gnu
module load gcc-native/14.2

# --- swap cray-mpich -> MVAPICH-Plus ------------------------------------------
module unload cray-mpich || true
module use /sw/frontier/ums/modulefiles
module load ums038/basic
module load mvapich-plus/4.0-gnu
# ------------------------------------------------------------------------------

module load rocm/7.1.1
module load libfabric
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module load miniforge3/23.11.0-0
module unload darshan-runtime || true

source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh

# Offline
unset all_proxy ftp_proxy http_proxy https_proxy no_proxy || true

# =============================================================================
# BUILD-FEATURE ENV
# =============================================================================
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export USE_ROCM=1
export USE_CUDA=0
export USE_XPU=0
export USE_DISTRIBUTED=1
export USE_MPI=1

# GPU-aware MPI: on MVAPICH-Plus this is a RUNTIME setting (CVARs at srun), NOT a
# build knob. So we do NOT set MPICH_GPU_SUPPORT_ENABLED here (that's Cray-only).
# USE_CUDA_MPI=1 only has an effect on the OSU-Nowlab/pytorch fork; it's a harmless
# no-op on stock PyTorch, kept here to document intent.
export USE_CUDA_MPI=1

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

# =============================================================================
# COMPILER + MPI WIRING  (the key change vs the Cray script)
#   base host compiler = gcc-native  (NOT cc/CC, which would relink cray-mpich)
#   MPI wrappers       = MVAPICH-Plus mpicc/mpicxx
# =============================================================================
export CC="$(which gcc)"
export CXX="$(which g++)"
export FC="$(which gfortran)"
export CMAKE_C_COMPILER="$CC"
export CMAKE_CXX_COMPILER="$CXX"
export CMAKE_Fortran_COMPILER="$FC"

# Resolve MVAPICH-Plus prefix from its mpicc on PATH
if ! command -v mpicc >/dev/null 2>&1; then
  echo "ERROR: mpicc not found after loading mvapich-plus/4.0"; exit 10
fi
export MPI_HOME="$(dirname "$(dirname "$(command -v mpicc)")")"
export MPI_C_COMPILER="$MPI_HOME/bin/mpicc"
export MPI_CXX_COMPILER="$MPI_HOME/bin/mpicxx"

export CFLAGS="${CFLAGS:-} -Wno-error -Wno-pedantic"
export CXXFLAGS="${CXXFLAGS:-} -Wno-error -Wno-pedantic"

# Pin ROCm explicitly
export ROCM_HOME=/opt/rocm-7.1.1
export ROCM_SOURCE_DIR=/opt/rocm-7.1.1

echo "MODULE LIST"; module list || true
echo "TOOLCHAIN"
which gcc; "$CC" --version | head -n1 || true
which g++; "$CXX" --version | head -n1 || true
echo "MPI_HOME=$MPI_HOME"
which mpicc; mpicc -show || true
which mpicxx; mpicxx -show || true
which hipcc || true; hipcc --version | head -n 5 || true

# =============================================================================
# ACTIVATE THE NEW ENV
# =============================================================================
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "ERROR: Env not found at $ENV_PREFIX. Create it on the login node first."
  exit 2
fi
conda activate "$ENV_PREFIX"
hash -r

export PATH="$ENV_PREFIX/bin:$PATH"
export CMAKE_COMMAND="$ENV_PREFIX/bin/cmake"
# MPI prefix first so FindMPI/find_package resolve MVAPICH-Plus, then the env
export CMAKE_PREFIX_PATH="$MPI_HOME:$ENV_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

export ROCM_PATH="$ROCM_HOME"
export HIP_PATH="$ROCM_HOME"
export HSA_PATH="$ROCM_HOME"
export PATH="$ROCM_HOME/bin:$PATH"

# Library search paths: MVAPICH-Plus + Slingshot libfabric + ROCm
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$MPI_HOME/lib:/opt/cray/libfabric/2.3.1/lib64:$ROCM_HOME/lib:$ROCM_HOME/lib/roctracer:$ROCM_HOME/lib/rocprofiler:$LD_LIBRARY_PATH"

echo "PYTHON/TOOLS IN ENV"
which python; python -V; python -m pip -V
"$CMAKE_COMMAND" --version || true

python - <<'PY'
import numpy, yaml, packaging, typing_extensions
print("deps ok:", numpy.__version__)
PY

# =============================================================================
# STAGE SOURCE  (copy from the shared clone; do NOT run git in it)
# =============================================================================
# sanity: submodules must be present in the shared checkout
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


python tools/amd_build/build_amd.py

# =============================================================================
# CMAKE ARGS  (MVAPICH-Plus wrappers instead of cc/CC)
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

# ROCm compat: provide libamdhip64.so.6 at build time (ROCm 7 ships .so.7)
export ROCM_COMPAT_DIR="/tmp/${USER}/rocm_compat_build_${JOBTAG}"
mkdir -p "$ROCM_COMPAT_DIR"
ln -sf "$ROCM_HOME/lib/libamdhip64.so.7" "$ROCM_COMPAT_DIR/libamdhip64.so.6"
export LD_LIBRARY_PATH="$ROCM_COMPAT_DIR:$LD_LIBRARY_PATH"

export PYTHONUNBUFFERED=1
export NINJA_STATUS="[%f/%t %e] "

# ---------------------------------------------------------------------------
# PHASE 1: CONFIGURE ONLY
# ---------------------------------------------------------------------------
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

[[ $cfg_rc -eq 0 ]] || { echo "ERROR: Configure failed (rc=$cfg_rc)."; exit $cfg_rc; }
[[ -f build/CMakeCache.txt ]] || { echo "ERROR: no CMakeCache.txt"; exit 90; }

echo "EARLY CHECK: key flags"
egrep '^(USE_ROCM|USE_CUDA|USE_DISTRIBUTED|USE_MPI|MPI_FOUND|MPI_C_FOUND|MPI_CXX_FOUND|MPI_C_INCLUDE_PATH|MPI_C_LIBRARIES|MPI_C_COMPILER|MPI_CXX_COMPILER|USE_KINETO|USE_ITT|USE_ROCTRACER|USE_ROCPROFILER):' \
  build/CMakeCache.txt || true

grep -qE '^USE_ROCM:BOOL=OFF$' build/CMakeCache.txt && { echo "ERROR: USE_ROCM OFF"; exit 98; } || true
if grep -qE '^USE_MPI:BOOL=OFF$' build/CMakeCache.txt || \
   grep -qE '^MPI_C_FOUND:BOOL=FALSE$' build/CMakeCache.txt || \
   grep -qE '^MPI_CXX_FOUND:BOOL=FALSE$' build/CMakeCache.txt; then
  echo "ERROR: MPI not detected in CMakeCache; aborting."; exit 97
fi
if grep -qE '^USE_(KINETO|ITT|ROCTRACER|ROCPROFILER):BOOL=ON$' build/CMakeCache.txt; then
  echo "ERROR: profiler stack ENABLED (expected OFF)."; exit 99
fi
echo "OK: ROCm ON, MPI ON (MVAPICH-Plus), profiler OFF. Proceeding."

# ---------------------------------------------------------------------------
# PHASE 2: BUILD WHEEL
# ---------------------------------------------------------------------------
echo "PHASE 2: BUILD TORCH WHEEL"
MAX_JOBS="$MAX_JOBS" \
USE_ROCM=1 USE_CUDA=OFF USE_DISTRIBUTED=1 USE_MPI=1 USE_CUDA_MPI=1 \
USE_TRITON=0 USE_AOTRITON=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 \
USE_NNPACK=0 USE_QNNPACK=0 USE_PYTORCH_QNNPACK=0 \
USE_KINETO=0 USE_ITT=0 USE_ROCTRACER=0 USE_ROCPROFILER=0 \
ROCM_SOURCE_DIR="$ROCM_SOURCE_DIR" CMAKE_COMMAND="$CMAKE_COMMAND" CMAKE_ARGS="$CMAKE_ARGS" \
python -u setup.py bdist_wheel -v 2>&1 | tee build.log

echo "TORCH WHEELS:"; ls -lh dist/*.whl
cp -av dist/*.whl "$WHEELS_DIR"/
python -m pip install --no-deps -U dist/*.whl

python - <<'PY'
import torch, torch.distributed as dist
print("torch:", torch.__version__, "| file:", torch.__file__)
print("hip:", torch.version.hip, "| device_count:", torch.cuda.device_count(),
      "| is_available:", torch.cuda.is_available())
print("MPI available:", dist.is_mpi_available())
PY

# ===========================================================================
# PHASE 3: TORCHVISION  (built against THIS torch, from your already-fixed tree)
# ===========================================================================
echo "PHASE 3: TORCHVISION from $VISION_SRC"
[[ -d "$VISION_SRC" ]] || { echo "ERROR: VISION_SRC not found: $VISION_SRC"; exit 12; }

rm -rf "$VIS_SCRATCH"
mkdir -p "$VIS_SCRATCH"
stage_tree "$VISION_SRC" "$VIS_SCRATCH" "torchvision source"
cd "$VIS_SCRATCH"
rm -rf build dist *.egg-info .eggs

# Safety net for the duplicate-symbol issue (vision::cuda_version() defined in both
# vision.cpp and the stray generated vision_hip.cpp). Your copied tree should already
# carry the setup.py exclusion; deleting the stray root-level file is harmless and
# guarantees it can't be double-compiled. (hipify only targets ops/cuda, so it is
# not regenerated at the csrc root.)
find torchvision/csrc -maxdepth 1 -name 'vision_hip.cpp' -delete || true

export FORCE_CUDA=0
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx90a

# Build against the CUSTOM torch already in this env (no isolation, no dep resolution)
python -m pip install -v --no-build-isolation --no-deps -U . 2>&1 | tee "$VIS_SCRATCH/vision_build.log"

# archive the torchvision wheel too (build a wheel copy for the record)
python -m pip wheel -v --no-build-isolation --no-deps -w "$WHEELS_DIR" . || true

python - <<'PY'
import torch, torchvision, torchvision.ops
print("torchvision:", torchvision.__version__, "| ops ok | hip:", torch.version.hip)
# NOTE: do NOT `import torchvision._C` — it's a torch op library, not a PyInit module.
# A successful `import torchvision.ops` above already proves _C loaded correctly.
PY

# ===========================================================================
# PHASE 4: CUSTOM ZFP (HIP) EXTENSION  (framework_allreduce_zfp_hip)
# ---------------------------------------------------------------------------
# Your zfp is a STANDALONE torch C++ extension, NOT part of torchvision.
# The stream-aware ZFP library (zfp-install-frontier) is MPI-agnostic and reused
# as-is. Only the extension is rebuilt, because a torch C++ extension must compile
# against the ABI of the torch it will run under (your notes, section 4).
# The extension bakes an rpath to $ZFP_PREFIX/lib(64), an absolute Lustre/home path
# that stays valid on compute nodes after conda-pack, so nothing needs copying in.
# ---------------------------------------------------------------------------
echo "PHASE 4: build framework_allreduce_zfp_hip from $EXT_SRC"
[[ -d "$EXT_SRC" ]]    || { echo "ERROR: EXT_SRC not found: $EXT_SRC"; exit 13; }
[[ -d "$ZFP_PREFIX" ]] || { echo "ERROR: ZFP_PREFIX not found: $ZFP_PREFIX"; exit 14; }

# pick lib64 vs lib (your setup.py does the same)
if [[ -d "$ZFP_PREFIX/lib64" ]]; then ZFP_LIBDIR="$ZFP_PREFIX/lib64"; else ZFP_LIBDIR="$ZFP_PREFIX/lib"; fi
echo "ZFP_PREFIX=$ZFP_PREFIX  ZFP_LIBDIR=$ZFP_LIBDIR"
ls -l "$ZFP_LIBDIR"/libzfp* || { echo "ERROR: no libzfp in $ZFP_LIBDIR"; exit 15; }

EXT_SCRATCH="/tmp/${USER}/zfpext_mvapich_build_${JOBTAG}"
rm -rf "$EXT_SCRATCH"
mkdir -p "$EXT_SCRATCH"
stage_tree "$EXT_SRC" "$EXT_SCRATCH" "ZFP extension source"
cd "$EXT_SCRATCH"
rm -rf build dist *.egg-info .eggs

# Drive your setup.py: absolute ZFP_PREFIX overrides its relative default, and
# ROCM_HOME points at the Frontier ROCm (not /opt/rocm) so includes/rpath are right.
export ZFP_PREFIX
export ROCM_HOME
python -m pip install -v --no-build-isolation --no-deps -U . 2>&1 | tee "$EXT_SCRATCH/zfpext_build.log"
python -m pip wheel  -v --no-build-isolation --no-deps -w "$WHEELS_DIR" . || true

# verify import + that the extension resolves libzfp
ZFP_EXT_SO="$(python -c 'import framework_allreduce_zfp_hip as m; print(m.__file__)')"
echo "zfp ext .so: $ZFP_EXT_SO"
ldd "$ZFP_EXT_SO" | egrep -i 'zfp|amdhip|not found' || true
python -c "import framework_allreduce_zfp_hip; print('zfp ext import ok')"
# ===========================================================================

# ---------------------------------------------------------------------------
# LDD sanity (must NOT show 'not found')
# ---------------------------------------------------------------------------
echo "LDD CHECK torch/_C*.so"
ldd "$ENV_PREFIX"/lib/python3.10/site-packages/torch/_C*.so | egrep "not found|libfabric|amdhip|mpi" || true
echo "LDD CHECK torchvision/_C*.so"
ldd "$ENV_PREFIX"/lib/python3.10/site-packages/torchvision/_C*.so | egrep "not found|amdhip" || true
echo "LDD CHECK framework_allreduce_zfp_hip (must resolve libzfp via rpath)"
ldd "$ENV_PREFIX"/lib/python3.10/site-packages/framework_allreduce_zfp_hip*.so | egrep -i "not found|zfp|amdhip" || true

# ===========================================================================
# PACK + STAGE + RUNTIME VERIFY
# ===========================================================================
command -v conda-pack >/dev/null 2>&1 || { echo "ERROR: conda-pack missing in env"; exit 3; }

rm -f "$TARBALL"
conda-pack -p "$ENV_PREFIX" -o "$TARBALL"
ls -lh "$TARBALL"

mkdir -p "$LOCAL_BASE"
sbcast -f "$TARBALL" "$LOCAL_BASE/conda_env.tar.gz"
rm -rf "$LOCAL_ENV"; mkdir -p "$LOCAL_ENV"
tar -xzf "$LOCAL_BASE/conda_env.tar.gz" -C "$LOCAL_ENV"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LOCAL_ENV"
conda-unpack

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# zfp lib dir added as belt-and-suspenders (rpath already covers it)
export LD_LIBRARY_PATH="$ZFP_LIBDIR:$MPI_HOME/lib:/opt/cray/libfabric/2.3.1/lib64:$ROCM_HOME/lib:$ROCM_HOME/lib/roctracer:$ROCM_HOME/lib/rocprofiler:$LD_LIBRARY_PATH"

python - <<'PY'
import torch, torchvision, torch.distributed as dist
print("RUNTIME torch:", torch.__version__, "| hip:", torch.version.hip)
print("RUNTIME torchvision:", torchvision.__version__)
print("RUNTIME device_count:", torch.cuda.device_count(), "| is_available:", torch.cuda.is_available())
print("RUNTIME MPI available:", dist.is_mpi_available())
import framework_allreduce_zfp_hip
print("RUNTIME zfp ext import ok")
PY

rm -rf "$NODE_SCRATCH" "$VIS_SCRATCH" "$EXT_SCRATCH"