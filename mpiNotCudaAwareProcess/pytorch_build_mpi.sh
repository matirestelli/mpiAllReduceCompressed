#!/bin/bash
#PBS -l select=1:ncpus=32:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A UIC-HPC
#PBS -j oe
#PBS -N pytorch_mpi_build

set -euo pipefail
trap 'echo "[ERROR] Failed at line $LINENO at $(date)" >&2' ERR

# ── Proxy (required on Polaris compute nodes) ─────────────────
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="localhost,127.0.0.1,*.local,*.alcf.anl.gov,polaris-*"

# ── Paths ─────────────────────────────────────────────────────
export BASE=/lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build
export TMPDIR=$BASE/tmp
export PIP_CACHE_DIR=$BASE/cache/pip
BUILD_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$BASE/logs/pytorch_build_$BUILD_TIMESTAMP.log"
HOME_LOGFILE="/home/mrest/mpiAllReduceCompressed/pytorch_build_$BUILD_TIMESTAMP.log"

mkdir -p "$BASE/tmp" "$BASE/cache/pip" "$BASE/logs" "/home/mrest/mpiAllReduceCompressed"
exec 1> >(tee -a "$LOGFILE" "$HOME_LOGFILE") 2>&1

echo "=========================================="
echo "PyTorch Build with CUDA-aware Cray MPICH"
echo "=========================================="
echo "Started at: $(date)"
echo "Node: $(hostname)"
echo "Log: $LOGFILE"

# ── Modules ───────────────────────────────────────────────────
module purge
module use /soft/modulefiles
module load PrgEnv-gnu
module load gcc-native/13
module load cuda/12.9
module load cray-mpich
module load craype-accel-nvidia80  # CRITICAL: activates GPU-aware MPI path

echo "[INFO] Modules loaded"
gcc --version | head -1
nvcc --version | head -1

# ── Cray MPICH ────────────────────────────────────────────────
export MPI_HOME=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3
if [ ! -f "$MPI_HOME/lib/libmpi.so" ]; then
    echo "[ERROR] Cray MPICH not found at $MPI_HOME — run 'module show cray-mpich' to get the right path"
    exit 1
fi

export LD_LIBRARY_PATH=$MPI_HOME/lib:${LD_LIBRARY_PATH:-}
export CPATH=$MPI_HOME/include:${CPATH:-}
export CMAKE_PREFIX_PATH=$MPI_HOME:${CMAKE_PREFIX_PATH:-}
export PKG_CONFIG_PATH=$MPI_HOME/lib/pkgconfig:${PKG_CONFIG_PATH:-}

GTL_LIB="$MPI_HOME/lib/libmpi_gtl_cuda.so"
if [ -f "$GTL_LIB" ]; then
    export LD_PRELOAD="$GTL_LIB"
    echo "[INFO] GTL CUDA library preloaded: $GTL_LIB"
else
    echo "[WARN] GTL CUDA library not found at $GTL_LIB"
fi

# ── GPU-aware MPI (per Polaris docs) ──────────────────────────
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_SUPPORT_LEVEL=1
export CRAY_ACCEL_TARGET=nvidia80

# ── Compilers: Cray wrappers ONLY, NOT mpicc/mpicxx ──────────
# Per Polaris docs: mpicc/mpicxx WILL NOT WORK for GPU applications
export CC=cc
export CXX=CC
export CUDAHOSTCXX=cc
export CUDA_HOST_COMPILER=cc

# ── CUDA-aware MPI define for PyTorch DDP whitelist check ─────
export CFLAGS="-DMPIX_GPU_SUPPORT_CUDA=1"
export CXXFLAGS="-DMPIX_GPU_SUPPORT_CUDA=1"

# ── Conda ─────────────────────────────────────────────────────
source /soft/applications/conda/2025-09-28/mconda3/bin/activate
conda activate base
unset CONDA_PREFIX_MPI  # prevent conda MPI from shadowing Cray MPICH

echo "[INFO] Environment ready:"
echo "  CC=$CC  CXX=$CXX"
echo "  MPI_HOME=$MPI_HOME"
echo "  MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED"
echo "  CFLAGS=$CFLAGS"
echo "  LD_PRELOAD=${LD_PRELOAD:-<none>}"

# ── PyTorch source ────────────────────────────────────────────
cd $BASE/src/pytorch
if [ ! -f "pyproject.toml" ]; then
    echo "[ERROR] PyTorch source not found in $BASE/src/pytorch"
    exit 1
fi
echo "[INFO] Source: $(pwd)  $(git log --oneline -1 2>/dev/null || true)"

# ── Patch ProcessGroupMPI.cpp ─────────────────────────────────
PGMPI="torch/csrc/distributed/c10d/ProcessGroupMPI.cpp"
if [ ! -f "$PGMPI" ]; then
    echo "[ERROR] $PGMPI not found"
    exit 1
fi
cp "$PGMPI" "${PGMPI}.backup.$(date +%s)"
if grep -q "Polaris Cray MPICH" "$PGMPI"; then
    echo "[INFO] ProcessGroupMPI.cpp already patched — skipping"
else
    sed -i '/^#else.*MPIX_CUDA_AWARE_SUPPORT.*MPIX_GPU_SUPPORT_CUDA/,/^#endif/ {
        s/^  return false;$/  \/\/ Polaris Cray MPICH is CUDA-aware via MPICH_GPU_SUPPORT_ENABLED\n  return true;/
    }' "$PGMPI"
    if grep -q "Polaris Cray MPICH" "$PGMPI"; then
        echo "[INFO] ProcessGroupMPI.cpp patched successfully"
    else
        echo "[WARN] Patch may not have applied — verify $PGMPI manually"
    fi
fi

# ── Clean previous build ──────────────────────────────────────
echo "[INFO] Cleaning build/ directory..."
rm -rf build/

# ── System NCCL (Polaris provides it at /soft/libraries/nccl) ─
# Use system NCCL instead of building from source — building from source
# with nvcc+GCC13 system headers crashes (previous build failure).
NCCL_DIR=$(ls -d /soft/libraries/nccl/nccl_*cuda12.9* 2>/dev/null | sort | tail -1)
if [ -z "$NCCL_DIR" ]; then
    echo "[WARN] System NCCL not found at /soft/libraries/nccl — building without NCCL"
    export USE_NCCL=0
    export USE_SYSTEM_NCCL=0
else
    echo "[INFO] Using system NCCL at: $NCCL_DIR"
    export USE_NCCL=1
    export USE_SYSTEM_NCCL=1
    export NCCL_ROOT=$NCCL_DIR
    export NCCL_INCLUDE_DIR=$NCCL_DIR/include
    export NCCL_LIB_DIR=$NCCL_DIR/lib
    export LD_LIBRARY_PATH=$NCCL_DIR/lib:${LD_LIBRARY_PATH:-}
    export CMAKE_PREFIX_PATH=$NCCL_DIR:${CMAKE_PREFIX_PATH:-}
fi

# ── cuDNN ─────────────────────────────────────────────────────
CUDNN_DIR=$(ls -d /soft/libraries/cudnn/cudnn-cuda12-* 2>/dev/null | sort | tail -1)
if [ -z "$CUDNN_DIR" ]; then
    echo "[WARN] System cuDNN not found at /soft/libraries/cudnn — building WITHOUT cuDNN (convolutions will be ~8x slower)"
    export USE_CUDNN=0
else
    echo "[INFO] Using system cuDNN at: $CUDNN_DIR"
    export USE_CUDNN=1
    export CUDNN_ROOT=$CUDNN_DIR
    export CUDNN_INCLUDE_DIR=$CUDNN_DIR/include
    export CUDNN_LIB_DIR=$CUDNN_DIR/lib
    export LD_LIBRARY_PATH=$CUDNN_DIR/lib:${LD_LIBRARY_PATH:-}
    export CMAKE_PREFIX_PATH=$CUDNN_DIR:${CMAKE_PREFIX_PATH:-}
fi

# ── Build flags ───────────────────────────────────────────────
export USE_MPI=1
export USE_CUDA=1
export USE_NINJA=1
export MAX_JOBS=32
export TORCH_CUDA_ARCH_LIST="8.0"  # Polaris A100

export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export USE_FBGEMM=0
export USE_NNPACK=0
export USE_CUFILE=0
export BUILD_TEST=0

echo ""
echo "=========================================="
echo "Starting build at $(date)"
echo "=========================================="

python -m pip install -v --no-build-isolation -e .
BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
    echo "[ERROR] Build FAILED (status $BUILD_STATUS) at $(date)"
    exit $BUILD_STATUS
fi
echo "[INFO] Build COMPLETED at $(date)"

# ── Verification ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
python3 - << 'VERIFY'
import torch
import torch.distributed as dist

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"CUDA version    : {torch.version.cuda}")
print(f"MPI available   : {dist.is_mpi_available()}")
print(f"NCCL available  : {dist.is_nccl_available()}")
print(f"cuDNN available : {torch.backends.cudnn.is_available()}")
print(f"cuDNN version   : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")

ok = True
if not dist.is_mpi_available():
    print("\n✗ FAILED: MPI not compiled in")
    ok = False
if not dist.is_nccl_available():
    print("\n✗ FAILED: NCCL not compiled in")
    ok = False
if not torch.backends.cudnn.is_available():
    print("\n✗ FAILED: cuDNN not available — convolutions will be ~8x slower")
    ok = False
if ok:
    print("\n✓ MPI, NCCL, and cuDNN all compiled in!")
    print("  MPI test : mpirun -np 4 --ppn 4 -env MPICH_GPU_SUPPORT_ENABLED=1 python test.py")
    print("  NCCL test: torchrun --nproc_per_node=4 test.py")
else:
    raise SystemExit(1)
VERIFY

echo "[INFO] Logs saved to:"
echo "  Eagle : $LOGFILE"
echo "  Home  : $HOME_LOGFILE"
echo "[INFO] Done at: $(date)"