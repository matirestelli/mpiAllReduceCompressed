#!/usr/bin/env bash
# =============================================================================
# Sophia PyTorch-from-source build env
# OpenMPI + GCC 13.2 + CUDA + Python 3.11 + NCCL build support
# Safe to source.
# =============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
else
  set -u
fi

_env_fail() {
  echo "[env_build_pytorch_sophia][ERROR] $*" >&2
  return 1 2>/dev/null || exit 1
}

# -----------------------------------------------------------------------------
# 0) Paths for your current test build
# -----------------------------------------------------------------------------
export TEST_ROOT="${TEST_ROOT:-/lus/eagle/projects/UIC-HPC/mrest/pytorch_sophia_try2}"
export PYTORCH_SOURCE="${PYTORCH_SOURCE:-${TEST_ROOT}/src/pytorch}"
export PYENV="${PYENV:-${TEST_ROOT}/conda/pt-mpi-py311}"

# -----------------------------------------------------------------------------
# 1) ALCF proxy
# -----------------------------------------------------------------------------
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="localhost,127.0.0.1,*.local,*.alcf.anl.gov,sophia-*,polaris-*,grand.alcf.anl.gov"

# -----------------------------------------------------------------------------
# 2) Modules: GCC first, then OpenMPI
# -----------------------------------------------------------------------------
if command -v module >/dev/null 2>&1; then
  module purge
  module use /soft/modulefiles

  # gcc/13.2.0 is hidden until this is loaded.
  module load spack-pe-base/0.7.1 || _env_fail "Could not load spack-pe-base/0.7.1"
  module load gcc/13.2.0 || _env_fail "Could not load gcc/13.2.0"
  module load compilers/openmpi/5.0.10 || _env_fail "Could not load compilers/openmpi/5.0.10"
else
  echo "[env_build_pytorch_sophia][WARN] module command not found"
fi

# -----------------------------------------------------------------------------
# 3) Python 3.11 env: force this, avoid /usr/bin/python and ~/.local Python 3.9
# -----------------------------------------------------------------------------
[[ -x "${PYENV}/bin/python" ]] || _env_fail "Python env not found at ${PYENV}/bin/python"

export PYTHONNOUSERSITE=1
export PATH="${PYENV}/bin:${PATH}"
hash -r
export PYTHONPATH="${PYTORCH_SOURCE}:${PYTHONPATH:-}"
# -----------------------------------------------------------------------------
# 4) CUDA: pick a real CUDA install and make CMake/nvcc see it
# -----------------------------------------------------------------------------
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  :
elif [[ -x /soft/compilers/cudatoolkit/cuda-13.2.1/bin/nvcc ]]; then
  export CUDA_HOME="/soft/compilers/cudatoolkit/cuda-13.2.1"
elif [[ -x /usr/local/cuda-12.9/bin/nvcc ]]; then
  export CUDA_HOME="/usr/local/cuda-12.9"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDA_HOME="/usr/local/cuda"
else
  _env_fail "Could not find nvcc. Check module avail cuda or module avail cudatoolkit."
fi

_strip_path_pattern() {
  local var="$1"
  local pattern="$2"
  local val="${!var:-}"
  val="$(printf '%s' "$val" | tr ':' '\n' | grep -v -E "$pattern" | paste -sd: -)"
  printf -v "$var" "%s" "$val"
}

_strip_path_pattern PATH '/usr/local/cuda|/soft/compilers/cudatoolkit/cuda'
_strip_path_pattern LD_LIBRARY_PATH '/usr/local/cuda|/soft/compilers/cudatoolkit/cuda'

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CUDAToolkit_ROOT="${CUDA_HOME}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export CUDA_NVCC_EXECUTABLE="${CUDA_HOME}/bin/nvcc"

# -----------------------------------------------------------------------------
# 5) GCC/libstdc++: important for NCCL GLIBCXX_3.4.30 error
# -----------------------------------------------------------------------------
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="${CXX}"
export CUDA_HOST_COMPILER="${CXX}"
export CMAKE_CUDA_HOST_COMPILER="${CXX}"

export GCC_LIBDIR="$(dirname "$("${CXX}" -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH}"

# -----------------------------------------------------------------------------
# 6) MPI: OpenMPI compiler wrappers for PyTorch MPI backend
# -----------------------------------------------------------------------------
export MPI_HOME="$(dirname "$(dirname "$(command -v mpicc)")")"
export MPI_INCLUDE="${MPI_HOME}/include"
export MPI_LIB="${MPI_HOME}/lib"
export MPI_C_COMPILER="$(command -v mpicc)"
export MPI_CXX_COMPILER="$(command -v mpicxx)"

# -----------------------------------------------------------------------------
# 7) PyTorch build options
# -----------------------------------------------------------------------------
export USE_DISTRIBUTED=1
export USE_MPI=1
export USE_CUDA=1

# Keep NCCL enabled because you need it later.
export USE_NCCL=1
export USE_SYSTEM_NCCL=0

# Disable optional CPU/mobile backends that already caused noise.
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export USE_FBGEMM=0
export BUILD_TEST=0

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

# CUB/CCCL headers from CUDA, useful for newer PyTorch builds.
if [[ -f "${CUDA_HOME}/include/cub/cub.cuh" ]]; then
  export CUB_INCLUDE_DIR="${CUDA_HOME}/include"
elif [[ -f "${CUDA_HOME}/targets/x86_64-linux/include/cub/cub.cuh" ]]; then
  export CUB_INCLUDE_DIR="${CUDA_HOME}/targets/x86_64-linux/include"
fi

export CMAKE_PREFIX_PATH="${PYENV}:${CUDA_HOME}:${CMAKE_PREFIX_PATH:-}"
if [[ -n "${CUB_INCLUDE_DIR:-}" ]]; then
  export CMAKE_PREFIX_PATH="${CUB_INCLUDE_DIR}:${CMAKE_PREFIX_PATH}"
fi

export CMAKE_ARGS="-DCUDAToolkit_ROOT=${CUDA_HOME} -DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=${CXX}"

# -----------------------------------------------------------------------------
# 8) Temp dirs
# -----------------------------------------------------------------------------
export TMPDIR="${TMPDIR:-/tmp/${USER}}"
mkdir -p "${TMPDIR}" 2>/dev/null || true
export PRTE_MCA_prte_tmpdir_base="${TMPDIR}"
export OMPI_MCA_prte_tmpdir_base="${TMPDIR}"
export OMPI_MCA_prte_silence_shared_fs=1

# -----------------------------------------------------------------------------
# 9) Summary
# -----------------------------------------------------------------------------
echo "============================================="
echo " Sophia PyTorch source build env"
echo "============================================="
echo "TEST_ROOT     : ${TEST_ROOT}"
echo "PYTORCH_SOURCE: ${PYTORCH_SOURCE}"
echo "Python        : $(command -v python)"
echo "Python ver    : $(python --version 2>&1)"
echo "pip           : $(python -m pip --version)"
echo "gcc           : $(command -v gcc)"
echo "gcc ver       : $(gcc --version | head -1)"
echo "g++           : $(command -v g++)"
echo "GCC_LIBDIR    : ${GCC_LIBDIR}"
echo "CUDA_HOME     : ${CUDA_HOME}"
echo "nvcc          : $(command -v nvcc)"
echo "nvcc ver      : $(nvcc --version | grep release || true)"
echo "MPI_HOME      : ${MPI_HOME}"
echo "mpicc         : $(command -v mpicc)"
echo "mpirun        : $(command -v mpirun)"
echo "USE_MPI       : ${USE_MPI}"
echo "USE_CUDA      : ${USE_CUDA}"
echo "USE_NCCL      : ${USE_NCCL}"
echo "MAX_JOBS      : ${MAX_JOBS}"
echo "============================================="

strings "$("${CXX}" -print-file-name=libstdc++.so.6)" | grep GLIBCXX_3.4.30 >/dev/null \
  && echo "[OK] libstdc++ has GLIBCXX_3.4.30" \
  || echo "[WARN] libstdc++ does not show GLIBCXX_3.4.30"

ompi_info --parsable --all 2>/dev/null | grep mpi_built_with_cuda_support:value || true