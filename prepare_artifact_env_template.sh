# Reference environment rebuild script for artifact evaluation.
#
# This template documents the full software-stack rebuild workflow used for
# our experiments:
#   1. load system modules
#   2. activate a clean conda environment
#   3. build/install ZFP into a system-specific prefix
#   4. build/install PyTorch
#   5. build/install torchvision
#   6. build/install the custom ZFP PyTorch extension
#   7. optionally package the environment with conda-pack
#   8. optionally verify the packaged environment
#
# Platform-specific ZFP source trees:
#   - NVIDIA/CUDA systems: build from repo_root/zfp
#   - AMD/HIP systems:     build from repo_root/zfp-staging
#
# ZFP must be built out-of-source in a fresh directory:
#   build-<system>
# and installed into a fresh system-specific prefix:
#   zfp-install-<system>
#
# This script is a template and requires site-specific edits for:
#   - module commands
#   - compiler/MPI settings
#   - source paths
#   - output paths


#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERROR] Failed at line ${LINENO}" >&2' ERR

# Template: rebuild artifact software stack on a target system.
# Mirrors Frontier-style workflow, but supports NVIDIA/CUDA too.
#
# Order:
#   1. load modules
#   2. activate conda env
#   3. build/install ZFP
#   4. build/install PyTorch
#   5. build/install torchvision
#   6. build/install ZFP extension
#   7. optionally pack env
#   8. optionally verify packed env
#
# IMPORTANT:
#   CUDA systems build ZFP from:       repo_root/zfp
#   AMD/HIP systems build ZFP from:    repo_root/zfp-staging
#
#   Use a fresh out-of-source build dir:
#     build-<system>
#   and a fresh install prefix:
#     zfp-install-<system>

# =============================================================================
# USER CONFIGURATION
# =============================================================================

PLATFORM="${PLATFORM:-amd}"              # amd | nvidia
MPI_IMPL="${MPI_IMPL:-cray-mpich}"      # cray-mpich | mvapich-plus | other
SYSTEM_NAME="${SYSTEM_NAME:-newSystem}"

PATCH_PYTORCH_MPI="${PATCH_PYTORCH_MPI:-0}"
PACKAGE_ENV="${PACKAGE_ENV:-1}"
VERIFY_PACKED_ENV="${VERIFY_PACKED_ENV:-1}"
STAGE_TO_SCRATCH="${STAGE_TO_SCRATCH:-0}"

REPO_ROOT="${REPO_ROOT:-</path/to/ddp-allreduce-eval-framework>}"

PYTORCH_SRC="${PYTORCH_SRC:-</path/to/pytorch/source>}"
VISION_SRC="${VISION_SRC:-</path/to/vision/source>}"

if [[ "$PLATFORM" == "amd" ]]; then
  ZFP_SRC="${ZFP_SRC:-$REPO_ROOT/zfp-staging}"
  EXT_SRC="${EXT_SRC:-$REPO_ROOT/ext_frontier}"
else
  ZFP_SRC="${ZFP_SRC:-$REPO_ROOT/zfp}"
  EXT_SRC="${EXT_SRC:-$REPO_ROOT/ext_polaris}"
fi

ENVROOT="${ENVROOT:-</path/to/output/root>}"
ENV_PREFIX="${ENV_PREFIX:-$ENVROOT/conda}"

ZFP_BUILD_DIR="${ZFP_BUILD_DIR:-$ZFP_SRC/build-${SYSTEM_NAME}}"
ZFP_PREFIX="${ZFP_PREFIX:-$REPO_ROOT/zfp-install-${SYSTEM_NAME}}"

TARBALL="${TARBALL:-$ENVROOT/conda_env.tar.gz}"
COMPAT_DIR="${COMPAT_DIR:-$ENVROOT/compat}"

EXT_MODULE_NAME_AMD="${EXT_MODULE_NAME_AMD:-framework_allreduce_zfp_hip}"
EXT_MODULE_NAME_NVIDIA="${EXT_MODULE_NAME_NVIDIA:-framework_allreduce_zfp_cuda}"

JOBTAG="${SLURM_JOB_ID:-nojobid}"
WHEELS_DIR="${WHEELS_DIR:-$ENVROOT/wheels/${JOBTAG}}"

BASE_SCRATCH="/tmp/${USER}/artifact_build_${JOBTAG}"
PYTORCH_BUILD_DIR="${PYTORCH_BUILD_DIR:-$BASE_SCRATCH/pytorch}"
VISION_BUILD_DIR="${VISION_BUILD_DIR:-$BASE_SCRATCH/vision}"
EXT_BUILD_DIR="${EXT_BUILD_DIR:-$BASE_SCRATCH/ext}"

LOCAL_BASE="${LOCAL_BASE:-/tmp/${USER}/artifact_env_local}"
LOCAL_ENV="${LOCAL_ENV:-$LOCAL_BASE/conda}"

mkdir -p "$ENVROOT" "$WHEELS_DIR" "$COMPAT_DIR"

# =============================================================================
# HELPERS
# =============================================================================

require_dir () { [[ -d "$1" ]] || { echo "ERROR: directory not found: $1"; exit 1; }; }
require_file () { [[ -f "$1" ]] || { echo "ERROR: file not found: $1"; exit 1; }; }

stage_tree () {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  rsync -a --delete \
    --exclude='.git/' \
    --exclude='build/' \
    --exclude='dist/' \
    --exclude='*.egg-info/' \
    --exclude='.eggs/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.o' \
    --exclude='*.so' \
    "$src/" "$dst/"
}

choose_build_tree () {
  local src="$1"
  local dst="$2"
  if [[ "$STAGE_TO_SCRATCH" == "1" ]]; then
    rm -rf "$dst"
    stage_tree "$src" "$dst"
    echo "$dst"
  else
    echo "$src"
  fi
}

# =============================================================================
# MODULES
# =============================================================================

setup_modules () {
  echo "=== MODULE SETUP ==="
  module purge || true

  if [[ "$PLATFORM" == "amd" ]]; then
    cat <<'EOF'
[INFO] AMD/HIP selected.
Replace this block with your real Frontier-style module commands.
EOF
  else
    cat <<'EOF'
[INFO] NVIDIA/CUDA selected.
Replace this block with your real Polaris-style module commands.
EOF
  fi

  :
  # Example:
  # module load ...
  # source </path/to/conda.sh>

  module list || true
}

# =============================================================================
# BUILD ENV
# =============================================================================

setup_build_env () {
  echo "=== BUILD ENV ==="

  export MAX_JOBS="${SLURM_CPUS_PER_TASK:-16}"
  export BUILD_TEST=0
  export USE_DISTRIBUTED=1
  export USE_MPI=1

  export USE_KINETO=0
  export USE_ITT=0
  export USE_NNPACK=0
  export USE_QNNPACK=0
  export USE_PYTORCH_QNNPACK=0
  export USE_TRITON=0
  export USE_AOTRITON=0
  export USE_FLASH_ATTENTION=0
  export USE_MEM_EFF_ATTENTION=0

  export CFLAGS="${CFLAGS:-} -Wno-error -Wno-pedantic"
  export CXXFLAGS="${CXXFLAGS:-} -Wno-error -Wno-pedantic"

  if [[ "$PLATFORM" == "amd" ]]; then
    export USE_ROCM=1
    export USE_CUDA=0
    export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
    export ROCM_PATH="$ROCM_HOME"
    export HIP_PATH="$ROCM_HOME"
    export HSA_PATH="$ROCM_HOME"
    export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx90a}"
    export HCC_AMDGPU_TARGET="${HCC_AMDGPU_TARGET:-gfx90a}"

    export CC="${CC:-$(command -v gcc)}"
    export CXX="${CXX:-$(command -v g++)}"
    export FC="${FC:-$(command -v gfortran)}"

    if [[ "$MPI_IMPL" == "mvapich-plus" ]]; then
      unset MPICH_GPU_SUPPORT_ENABLED || true
      unset MV2_USE_CUDA || true
      export MV2_USE_ROCM=1
    fi
  else
    export USE_ROCM=0
    export USE_CUDA=1
    export USE_NINJA=1
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
  fi

  if command -v mpicc >/dev/null 2>&1; then
    export MPI_HOME="${MPI_HOME:-$(dirname "$(dirname "$(command -v mpicc)")")}"
    export MPI_C_COMPILER="${MPI_C_COMPILER:-$MPI_HOME/bin/mpicc}"
    export MPI_CXX_COMPILER="${MPI_CXX_COMPILER:-$MPI_HOME/bin/mpicxx}"
  fi
}

# =============================================================================
# ACTIVATE ENV
# =============================================================================

activate_env () {
  echo "=== ACTIVATE ENV ==="

  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    cat <<EOF
ERROR: conda env not found at:
  $ENV_PREFIX

Create it first, e.g.:
  conda create -y -p "$ENV_PREFIX" python=3.10 pip cmake ninja pkg-config \
    numpy pyyaml typing_extensions packaging pillow
  conda activate "$ENV_PREFIX"
  conda install -y -c conda-forge conda-pack
EOF
    exit 2
  fi

  conda activate "$ENV_PREFIX"
  hash -r

  export PATH="$ENV_PREFIX/bin:${ROCM_HOME:-}/bin:$PATH"
  export CMAKE_COMMAND="${CMAKE_COMMAND:-$ENV_PREFIX/bin/cmake}"

  if [[ -n "${MPI_HOME:-}" ]]; then
    export CMAKE_PREFIX_PATH="$MPI_HOME:$ENV_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  else
    export CMAKE_PREFIX_PATH="$ENV_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  fi

  local torch_libdir="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib"
  export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$COMPAT_DIR:$torch_libdir${MPI_HOME:+:$MPI_HOME/lib}${ROCM_HOME:+:$ROCM_HOME/lib}:$LD_LIBRARY_PATH"
}

# =============================================================================
# BUILD ZFP
# =============================================================================

build_zfp () {
  echo "=== BUILD ZFP ==="
  require_dir "$ZFP_SRC"

  mkdir -p "$ZFP_BUILD_DIR"
  cd "$ZFP_BUILD_DIR"

  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$ZFP_PREFIX" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

  cmake --build . -j "${MAX_JOBS}"
  cmake --install .

  echo "Installed ZFP into: $ZFP_PREFIX"
}

# =============================================================================
# OPTIONAL PYTORCH PATCH
# =============================================================================

patch_pytorch_if_needed () {
  [[ "$PATCH_PYTORCH_MPI" == "1" ]] || return 0

  local pgmpi="$1/torch/csrc/distributed/c10d/ProcessGroupMPI.cpp"
  require_file "$pgmpi"
  cp -f "$pgmpi" "${pgmpi}.bak"

  python - <<PY
from pathlib import Path
p = Path(r"$pgmpi")
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
    print("Patch already present")
elif old in s:
    p.write_text(s.replace(old, new, 1))
    print("Applied MPI patch")
else:
    print("WARNING: expected block not found")
PY
}

# =============================================================================
# BUILD PYTORCH
# =============================================================================

build_pytorch () {
  echo "=== BUILD PYTORCH ==="
  require_dir "$PYTORCH_SRC"
  require_file "$PYTORCH_SRC/pyproject.toml"

  local torch_tree
  torch_tree="$(choose_build_tree "$PYTORCH_SRC" "$PYTORCH_BUILD_DIR")"

  cd "$torch_tree"
  rm -rf build dist
  export TMPDIR="$torch_tree/tmp"
  mkdir -p "$TMPDIR"

  patch_pytorch_if_needed "$torch_tree"

  if [[ "$PLATFORM" == "amd" ]]; then
    python tools/amd_build/build_amd.py
    export CMAKE_ARGS="\
-DUSE_ROCM=ON \
-DROCM_PATH=$ROCM_HOME \
-DHIP_ROOT_DIR=$ROCM_HOME \
-DUSE_MPI=ON \
${MPI_C_COMPILER:+-DMPI_C_COMPILER=$MPI_C_COMPILER} \
${MPI_CXX_COMPILER:+-DMPI_CXX_COMPILER=$MPI_CXX_COMPILER}"
  else
    export CMAKE_ARGS="\
-DUSE_MPI=ON \
${MPI_C_COMPILER:+-DMPI_C_COMPILER=$MPI_C_COMPILER} \
${MPI_CXX_COMPILER:+-DMPI_CXX_COMPILER=$MPI_CXX_COMPILER}"
  fi

  python -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

  export ONLY_RUN_CMAKE=1
  python -u setup.py build -v | tee configure.log
  unset ONLY_RUN_CMAKE

  python -u setup.py bdist_wheel -v | tee build.log
  cp -av dist/*.whl "$WHEELS_DIR"/
  python -m pip install --no-deps -U dist/*.whl
}

# =============================================================================
# BUILD TORCHVISION
# =============================================================================

build_torchvision () {
  echo "=== BUILD TORCHVISION ==="
  require_dir "$VISION_SRC"

  local vision_tree
  vision_tree="$(choose_build_tree "$VISION_SRC" "$VISION_BUILD_DIR")"

  cd "$vision_tree"
  rm -rf build dist *.egg-info .eggs

  if [[ "$PLATFORM" == "amd" ]]; then
    find torchvision/csrc -maxdepth 1 -name 'vision_hip.cpp' -delete || true
    export BUILD_VERSION="${BUILD_VERSION:-0.24.0}"
    export FORCE_CUDA=0
    export USE_CUDA=0
    export USE_ROCM=1
    export TORCHVISION_USE_VIDEO_CODEC=0
  else
    export FORCE_CUDA=1
    export USE_CUDA=1
    export USE_ROCM=0
  fi

  python -m pip uninstall -y torchvision >/dev/null 2>&1 || true
  python -m pip install -v --no-build-isolation --no-deps -U . | tee vision_build.log
  python -m pip wheel -v --no-build-isolation --no-deps -w "$WHEELS_DIR" . || true
}

# =============================================================================
# BUILD ZFP EXTENSION
# =============================================================================

build_zfp_extension () {
  echo "=== BUILD ZFP EXTENSION ==="
  require_dir "$EXT_SRC"
  require_dir "$ZFP_PREFIX"

  local ext_tree
  ext_tree="$(choose_build_tree "$EXT_SRC" "$EXT_BUILD_DIR")"

  cd "$ext_tree"
  rm -rf build dist *.egg-info .eggs

  export ZFP_PREFIX

  if [[ "$PLATFORM" == "amd" ]]; then
    export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
    python -m pip uninstall -y "$EXT_MODULE_NAME_AMD" >/dev/null 2>&1 || true
  else
    python -m pip uninstall -y "$EXT_MODULE_NAME_NVIDIA" >/dev/null 2>&1 || true
  fi

  python -m pip install -v --no-build-isolation --no-deps -U . | tee zfpext_build.log
  python -m pip wheel -v --no-build-isolation --no-deps -w "$WHEELS_DIR" . || true

  if [[ "$PLATFORM" == "amd" ]]; then
    python - <<PY
__import__("$EXT_MODULE_NAME_AMD")
print("Imported $EXT_MODULE_NAME_AMD OK")
PY
  else
    python - <<PY
__import__("$EXT_MODULE_NAME_NVIDIA")
print("Imported $EXT_MODULE_NAME_NVIDIA OK")
PY
  fi
}

# =============================================================================
# PACKAGE ENV
# =============================================================================

package_env () {
  [[ "$PACKAGE_ENV" == "1" ]] || return 0
  echo "=== PACKAGE ENV ==="

  command -v conda-pack >/dev/null 2>&1 || { echo "ERROR: conda-pack not found"; exit 4; }

  rm -f "$TARBALL"
  conda-pack -p "$ENV_PREFIX" -o "$TARBALL"
  ls -lh "$TARBALL"
}

# =============================================================================
# VERIFY PACKED ENV
# =============================================================================

verify_packed_env () {
  [[ "$VERIFY_PACKED_ENV" == "1" ]] || return 0
  [[ "$PACKAGE_ENV" == "1" ]] || return 0

  echo "=== VERIFY PACKED ENV ==="

  mkdir -p "$LOCAL_BASE"
  rm -rf "$LOCAL_ENV"
  mkdir -p "$LOCAL_ENV"

  cp "$TARBALL" "$LOCAL_BASE/conda_env.tar.gz"
  tar -xzf "$LOCAL_BASE/conda_env.tar.gz" -C "$LOCAL_ENV"

  conda activate "$LOCAL_ENV"
  conda-unpack

  local zfp_libdir
  if [[ -d "$ZFP_PREFIX/lib64" ]]; then
    zfp_libdir="$ZFP_PREFIX/lib64"
  else
    zfp_libdir="$ZFP_PREFIX/lib"
  fi

  local local_torch_libdir="$LOCAL_ENV/lib/python3.10/site-packages/torch/lib"
  export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$COMPAT_DIR:$local_torch_libdir:$zfp_libdir${MPI_HOME:+:$MPI_HOME/lib}${ROCM_HOME:+:$ROCM_HOME/lib}:$LD_LIBRARY_PATH"

  if [[ "$PLATFORM" == "amd" ]]; then
    python - <<PY
import torch, torchvision
__import__("$EXT_MODULE_NAME_AMD")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("ext: ok")
PY
  else
    python - <<PY
import torch, torchvision
__import__("$EXT_MODULE_NAME_NVIDIA")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("ext: ok")
PY
  fi
}

# =============================================================================
# MAIN
# =============================================================================

main () {
  echo "Started: $(date)"
  echo "PLATFORM=$PLATFORM"
  echo "SYSTEM_NAME=$SYSTEM_NAME"

  require_dir "$REPO_ROOT"
  require_dir "$PYTORCH_SRC"
  require_dir "$VISION_SRC"
  require_dir "$ZFP_SRC"
  require_dir "$EXT_SRC"

  setup_modules
  setup_build_env
  activate_env
  build_zfp
  build_pytorch
  build_torchvision
  build_zfp_extension
  package_env
  verify_packed_env

  rm -rf "$PYTORCH_BUILD_DIR" "$VISION_BUILD_DIR" "$EXT_BUILD_DIR"
  echo "DONE: $(date)"
}

main "$@"
