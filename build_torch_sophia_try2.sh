#!/usr/bin/env bash
# =============================================================================
# build_torch_sophia.sh
# One-command PyTorch (source) build/install on Sophia with MPI+UCX+CUDA.
#
# Usage:
#   bash build_torch_sophia.sh /path/to/pytorch/src [/path/to/venv]
#
# Examples:
#   bash build_torch_sophia.sh \
#     /lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build_sophia/src \
#     /lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build_sophia/mpi_enabled_overlay/venvs/pt-mpi
#
# Notes:
# - This script enforces correct CMAKE_ARGS usage (ENV var), and refuses the
#   wrong "-DCMAKE_ARGS=..." misuse.
# - It does NOT run PyTorch tests (BUILD_TEST=0).
# - It can optionally disable NVTX (default: 1 = disable) because your build
#   keeps failing on CUDA::nvToolsExt.
# =============================================================================

set -euo pipefail

# ------------------------- user-tunable defaults ------------------------------
: "${CUDA_HOME:=/usr/local/cuda-12.9}"
: "${UCX_ROOT:=/soft/libraries/ucx/1.17.0}"
: "${OPENMPI_MODULE:=compilers/openmpi/5.0.10}"

# Default behavior: disable NVTX to avoid CUDA::nvToolsExt failure.
: "${USE_NVTX:=0}"
: "${BUILD_TEST:=0}"
: "${PYTHONNOUSERSITE:=1}"

# Optional: if you want to force a specific CMake module (recommended if you have cmake/4.x issues)
# Set e.g.: export CMAKE_MODULE=cmake/3.28.6
: "${CMAKE_MODULE:=}"

# ------------------------------ helpers --------------------------------------
die() { echo "[build_torch_sophia][ERROR] $*" >&2; exit 1; }

banner() {
  echo "============================================================"
  echo "$*"
  echo "============================================================"
}

# ------------------------------ args -----------------------------------------
TORCH_SRC="${1:-}"
VENV_PATH="${2:-}"

[[ -n "$TORCH_SRC" ]] || die "Missing PyTorch source dir. Usage: $0 /path/to/pytorch/src [/path/to/venv]"
[[ -d "$TORCH_SRC" ]] || die "PyTorch source dir not found: $TORCH_SRC"

# ------------------------------ env setup ------------------------------------
banner "Loading modules"
if command -v module >/dev/null 2>&1; then
  module purge
  module use /soft/modulefiles
  module load "$OPENMPI_MODULE"
  if [[ -n "$CMAKE_MODULE" ]]; then
    module load "$CMAKE_MODULE"
  fi
else
  echo "[build_torch_sophia][WARN] module command not found; skipping module load." >&2
fi

banner "Proxy (ALCF)"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"

banner "Temp directories"
export TMPDIR="${TMPDIR:-/tmp/${USER}}"
mkdir -p "$TMPDIR" 2>/dev/null || true
chmod 700 "$TMPDIR" 2>/dev/null || true
export OMPI_MCA_prte_tmpdir_base="$TMPDIR"
export PRTE_MCA_prte_tmpdir_base="$TMPDIR"
export OMPI_MCA_prte_silence_shared_fs="${OMPI_MCA_prte_silence_shared_fs:-1}"

banner "Clear stale/foreign settings"
unset LD_PRELOAD 2>/dev/null || true
unset MPICH_GPU_SUPPORT_ENABLED MPICH_GPU_SUPPORT_LEVEL MPICH_MAX_THREAD_SAFETY CRAY_ACCEL_TARGET 2>/dev/null || true
unset FI_CXI_DEFAULT_CQ_SIZE FI_CXI_RX_MATCH_MODE 2>/dev/null || true

export PYTHONNOUSERSITE="$PYTHONNOUSERSITE"

banner "CUDA setup"
# Ensure the correct CUDA is first
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CUDAToolkit_ROOT="$CUDA_HOME"
export CUB_HOME="${CUB_HOME:-$CUDA_HOME}"

# UCX
export PATH="${UCX_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${UCX_ROOT}/lib:${LD_LIBRARY_PATH}"

# Threading defaults
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# OpenMPI defaults
export OMPI_MCA_pml="${OMPI_MCA_pml:-ucx}"

banner "CMake args enforcement"
# Correct usage: CMAKE_ARGS is an ENV var whose content is appended as cmake flags.
# We set it here and refuse the incorrect "-DCMAKE_ARGS=..."
export CMAKE_ARGS="-DCUDAToolkit_ROOT=${CUDA_HOME}"

if [[ "${CMAKE_ARGS}" == *"-DCMAKE_ARGS="* ]]; then
  die "Misuse detected: CMAKE_ARGS contains '-DCMAKE_ARGS=...'. Fix your environment."
fi

# Guard: refuse users passing bad flags via EXTRA_CMAKE_FLAGS
: "${EXTRA_CMAKE_FLAGS:=}"
if [[ "$EXTRA_CMAKE_FLAGS" == *"-DCMAKE_ARGS="* ]]; then
  die "Do not use -DCMAKE_ARGS=... anywhere. Put flags in CMAKE_ARGS env var."
fi
export CMAKE_ARGS="${CMAKE_ARGS} ${EXTRA_CMAKE_FLAGS}"

banner "NVTX check (nvToolsExt)"
NVTX_HDR="${CUDA_HOME}/include/nvToolsExt.h"
NVTX_LIB1="${CUDA_HOME}/lib64/libnvToolsExt.so"
NVTX_LIB2="${CUDA_HOME}/targets/x86_64-linux/lib/libnvToolsExt.so"

if [[ -r "$NVTX_HDR" && ( -r "$NVTX_LIB1" || -r "$NVTX_LIB2" ) ]]; then
  echo "[build_torch_sophia] NVTX appears present under CUDA_HOME."
else
  echo "[build_torch_sophia][WARN] NVTX not found under CUDA_HOME:"
  echo "  header: $NVTX_HDR"
  echo "  libs  : $NVTX_LIB1 or $NVTX_LIB2"
  echo "  If build fails on CUDA::nvToolsExt, you must use a CUDA install/module that includes NVTX dev files,"
  echo "  or keep USE_NVTX=0 (default in this script)."
fi

# ------------------------------ python/venv ----------------------------------
if [[ -n "$VENV_PATH" ]]; then
  banner "Activating venv: $VENV_PATH"
  [[ -f "$VENV_PATH/bin/activate" ]] || die "Venv activation script not found: $VENV_PATH/bin/activate"
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

command -v python >/dev/null 2>&1 || die "python not found in PATH"
command -v pip >/dev/null 2>&1 || die "pip not found in PATH"
command -v cmake >/dev/null 2>&1 || die "cmake not found in PATH"
command -v ninja >/dev/null 2>&1 || echo "[build_torch_sophia][WARN] ninja not found; PyTorch will try another generator."

banner "Versions"
echo "python: $(python -V 2>&1)"
echo "pip   : $(pip -V 2>&1)"
echo "cmake : $(cmake --version | head -n 1)"
echo "CUDA_HOME=$CUDA_HOME"
echo "CMAKE_ARGS=$CMAKE_ARGS"
echo "USE_NVTX=$USE_NVTX BUILD_TEST=$BUILD_TEST"

# ------------------------------ build/install --------------------------------
banner "Building PyTorch (pip install -v .)"
cd "$TORCH_SRC"

# Important: remove stale build dir to avoid mixed configure attempts.
rm -rf build

export BUILD_TEST="$BUILD_TEST"
export USE_NVTX="$USE_NVTX"

# Hard guard: fail if someone exported a wrong thing elsewhere
if env | grep -qE '(^| )CMAKE_ARGS=.*-DCMAKE_ARGS='; then
  die "Environment contains wrong '-DCMAKE_ARGS=' usage."
fi

# Run build
pip install -v . 2>&1 | tee "build_$(date +%F_%H%M%S).log"

banner "Done"
echo "If you still see 'Manually-specified variables were not used: CMAKE_ARGS',"
echo "then some wrapper is explicitly passing -DCMAKE_ARGS=... to CMake."
echo "Search with: grep -RIn \"DCMAKE_ARGS\" /lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build_sophia"


#chmod +x
#./build_torch_sophia.sh /lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build_sophia/src \
#  /lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build_sophia/mpi_enabled_overlay/venvs/pt-mpi
