#!/bin/bash
# =============================================================================
# Sophia DDP + Eagle PyTorch (installed in conda env) + system CUDA + OpenMPI
# Safe to SOURCE on login nodes (won't crash your shell if CUDA libs are absent).
# =============================================================================

# If executed as a script: be strict. If sourced: avoid `set -e` (can abort your shell).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
else
  set -u
fi

# helper: fail gracefully whether sourced or executed
_env_fail() {
  echo "[envScriptSophia][ERROR] $*" >&2
  return 1 2>/dev/null || exit 1
}

# --- 1. Clear Polaris-only state ---
unset LD_PRELOAD || true
unset MPICH_GPU_SUPPORT_ENABLED || true
unset MPICH_GPU_SUPPORT_LEVEL || true
unset MPICH_MAX_THREAD_SAFETY || true
unset CRAY_ACCEL_TARGET || true
unset FI_CXI_DEFAULT_CQ_SIZE || true
unset FI_CXI_RX_MATCH_MODE || true

# --- 2. Sophia modules (keep as you had) ---
module purge
module use /soft/modulefiles
module load compilers/openmpi/5.0.10

# --- 3. CUDA ---
# On login nodes, CUDA runtime libs may not exist; that's OK. We still set vars.
if [[ -d /usr/local/cuda-12.9 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.9
else
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Informational preflight only (do NOT error on login nodes)
if [[ ! -e "${CUDA_HOME}/lib64/libcudart.so.12" ]]; then
  echo "[envScriptSophia][INFO] ${CUDA_HOME}/lib64/libcudart.so.12 not found on this node (common on login nodes)."
fi
if [[ ! -e "${CUDA_HOME}/lib64/libcublas.so.12" ]]; then
  echo "[envScriptSophia][INFO] ${CUDA_HOME}/lib64/libcublas.so.12 not found on this node (common on login nodes)."
fi

# --- 4. Eagle Python / PyTorch (USE INSTALLED ENV, NOT SOURCE TREE) ---
export PYTORCH_MPI_BUILD="${PYTORCH_MPI_BUILD:-/lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build_sophia}"

# Keep source location as a variable for reference, but DO NOT add to PYTHONPATH.
export PYTORCH_SOURCE="${PYTORCH_SOURCE:-${PYTORCH_MPI_BUILD}/src/pytorch}"

# Use the env you installed torch into:
export EAGLE_PYTHON_PREFIX="${EAGLE_PYTHON_PREFIX:-${PYTORCH_MPI_BUILD}/conda/pt-sophia-openmpi}"

if [[ -x "${EAGLE_PYTHON_PREFIX}/bin/python" ]]; then
  export PATH="${EAGLE_PYTHON_PREFIX}/bin:${PATH}"
else
  _env_fail "Eagle Python env not found at ${EAGLE_PYTHON_PREFIX}/bin/python"
fi

# Make sure we don't accidentally import user-site or source-tree torch
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# Optional: pip cache (nice to keep)
export PIP_CACHE_DIR="${PYTORCH_MPI_BUILD}/cache/pip"
mkdir -p "${PIP_CACHE_DIR}" >/dev/null 2>&1 || true

# Keep these extra library paths if they exist (your original intent)
# NOTE: If you get CUDA/torch loader conflicts, comment this block first.
for libdir in \
  "/soft/libraries/nccl/nccl_2.28.3-1+cuda12.9_x86_64/lib" \
  "/soft/libraries/cudnn/cudnn-cuda12-linux-x64-v9.5.1.17/lib" \
  "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64"
do
  if [[ -d "${libdir}" ]]; then
    export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH}"
  fi
done

# NOTE: Intentionally not adding "${PYTORCH_SOURCE}/torch/lib" to LD_LIBRARY_PATH.

# --- 5. ZFP / cuZFP (kept; make stable even if you source from a different directory) ---
# Prefer existing ZFP_HOME if set;  otherwise default to your Sophia install prefix.
if [[ -z "${ZFP_HOME:-}" ]]; then
  if [[ -n "${PBS_O_WORKDIR:-}" && -d "${PBS_O_WORKDIR}/zfp-install-sophia" ]]; then
    export ZFP_HOME="${PBS_O_WORKDIR}/zfp-install-sophia"
  elif [[ -d "${HOME}/mpiAllReduceCompressed/zfp-install-sophia" ]]; then
    export ZFP_HOME="${HOME}/mpiAllReduceCompressed/zfp-install-sophia"
  else
    export ZFP_HOME="/home/mrest/mpiAllReduceCompressed/zfp-install-sophia"
  fi
fi

export PATH="${ZFP_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${ZFP_HOME}/lib:${ZFP_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="${ZFP_HOME}/include:${CPATH:-}"
export LIBRARY_PATH="${ZFP_HOME}/lib:${ZFP_HOME}/lib64:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${ZFP_HOME}:${CMAKE_PREFIX_PATH:-}"

# --- 6. Proxy (kept) ---
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="localhost,127.0.0.1,*.local,*.alcf.anl.gov,sophia-*,polaris-*,grand.alcf.anl.gov"

# --- 7. Threading (kept) ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# --- 8. Sanity check (safe on login nodes) ---
echo "============================================="
echo "  Sophia DDP + Installed PyTorch + OpenMPI"
echo "============================================="
echo "Node               : $(hostname)"
echo "Python             : $(command -v python)"
echo "NVCC               : $(command -v nvcc || true)"
echo "MPI launcher       : $(command -v mpirun || command -v mpiexec || true)"
echo "CUDA_HOME          : ${CUDA_HOME}"
echo "PYTORCH_MPI_BUILD  : ${PYTORCH_MPI_BUILD}"
echo "PYTORCH_SOURCE     : ${PYTORCH_SOURCE}"
echo "EAGLE_PYTHON_PREFIX: ${EAGLE_PYTHON_PREFIX}"
echo "ZFP_HOME           : ${ZFP_HOME}"
echo "PYTHONPATH         : ${PYTHONPATH:-<unset>}"
echo "============================================="

# Only try importing torch if CUDA runtime libs exist; otherwise torch CUDA wheels will fail import.
if [[ -e "${CUDA_HOME}/lib64/libcudart.so.12" && ( -e "${CUDA_HOME}/lib64/libcublas.so.12" || -e "${CUDA_HOME}/lib64/libcublas.so.11" ) ]]; then
  python - <<'PY'
import sys
import torch
import torch.distributed as dist

print(f"Python version    : {sys.version.split()[0]}")
print(f"Python executable : {sys.executable}")
print(f"PyTorch version   : {torch.__version__}")
print(f"PyTorch location  : {torch.__file__}")
print(f"Built CUDA        : {torch.version.cuda}")
print(f"CUDA available    : {torch.cuda.is_available()} {torch.cuda.device_count()}")
print(f"MPI available     : {dist.is_mpi_available()}")
print(f"NCCL available    : {dist.is_nccl_available()}")
print(f"Gloo available    : {dist.is_gloo_available()}")

try:
    from mpi4py import MPI
    print("mpi4py            : " + MPI.Get_library_version().splitlines()[0])
except Exception as exc:
    print(f"[WARN] mpi4py check failed: {exc!r}")
PY
else
  echo "[envScriptSophia][INFO] CUDA runtime libs not available on this node."
  echo "[envScriptSophia][INFO] Skipping 'import torch' sanity check to avoid import failure."
  echo "[envScriptSophia][INFO] Run this script inside an interactive GPU job / compute node to test CUDA PyTorch."
fi
