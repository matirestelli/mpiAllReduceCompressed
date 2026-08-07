#!/bin/bash
set -euo pipefail
set -x
trap 'echo "FAILED at line $LINENO"; exit 1' ERR

# -----------------------------------------------------------------------------
# Clean base environment
# -----------------------------------------------------------------------------
unset LD_PRELOAD
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# -----------------------------------------------------------------------------
# Module environment
# -----------------------------------------------------------------------------
module purge || true
module load cpe/26.03
module load PrgEnv-gnu
module load gcc-native/14.2
module use /sw/frontier/ums/modulefiles
module load ums038/basic
module load mvapich-plus/4.0-gnu
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module load miniforge3/23.11.0-0

# IMPORTANT:
# mvapich-plus/4.0-gnu already loads the ROCm version it was built against.
# Do NOT override it with a different ROCm module.

source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
conda deactivate || true

# Prefer Cray library path additions first
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# -----------------------------------------------------------------------------
# GPU-aware MVAPICH-Plus settings
# -----------------------------------------------------------------------------
# Runtime gate used by patched PyTorch HIP-aware MPI detection
export MPICH_GPU_SUPPORT_ENABLED=1

# Explicit MVAPICH-Plus GPU-aware settings
export MVP_ENABLE_GPU=1
export MVP_CH4_OFI_ENABLE_HMEM=1
export MVP_CH4_OFI_ENABLE_MR_HMEM=1


# Optional diagnostics
export MV2_SHOW_ENV_INFO=1

# -----------------------------------------------------------------------------
# OFI / Slingshot / RCCL tuning
# -----------------------------------------------------------------------------
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=2048
export FI_CXI_RX_MATCH_MODE=hybrid

export NCCL_NET_GDR_LEVEL=3
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

# -----------------------------------------------------------------------------
# Threading
# -----------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# -----------------------------------------------------------------------------
# Proxy, if needed
# -----------------------------------------------------------------------------
export http_proxy=http://proxy.ccs.ornl.gov:3128
export https_proxy=http://proxy.ccs.ornl.gov:3128
export HTTP_PROXY=http://proxy.ccs.ornl.gov:3128
export HTTPS_PROXY=http://proxy.ccs.ornl.gov:3128

# -----------------------------------------------------------------------------
# ZFP
# -----------------------------------------------------------------------------
export ZFP_HOME="$HOME/ddp-allreduce-eval-framework/zfp-install-frontier"
export LD_LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$ZFP_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$ZFP_HOME/lib:$ZFP_HOME/lib64:${LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$ZFP_HOME:${CMAKE_PREFIX_PATH:-}"

# -----------------------------------------------------------------------------
# Must run inside a Slurm allocation
# -----------------------------------------------------------------------------
: "${SLURM_JOB_ID:?This script should be sourced/run inside a Slurm job allocation}"
: "${SLURM_NNODES:?SLURM_NNODES must be set}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

# -----------------------------------------------------------------------------
# MIOpen cache staging to node-local storage
# -----------------------------------------------------------------------------
export MIOPEN_BASE=/lustre/orion/gen243/proj-shared/matilderestelli/miopen_cache_baseline
NVME_BASE="/mnt/bb/${USER}"
export MIOPEN_LOCAL="${NVME_BASE}/miopen_cache_${SLURM_JOB_ID}"
export MIOPEN_USER_DB_PATH="${MIOPEN_LOCAL}/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_LOCAL}/kernel_cache"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  mkdir -p '$NVME_BASE' '$MIOPEN_LOCAL' '$MIOPEN_USER_DB_PATH' '$MIOPEN_CUSTOM_CACHE_DIR'
"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  if [[ -d '$MIOPEN_BASE' ]]; then
    cp -a '$MIOPEN_BASE/.' '$MIOPEN_LOCAL/' 2>/dev/null || true
  fi
"

export ORIGINAL_HOME="${HOME}"
export HOME="${MIOPEN_LOCAL}"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  mkdir -p '$HOME/.cache/miopen' '$HOME/.config/miopen'
"

unset MIOPEN_FIND_MODE
unset MIOPEN_FIND_ENFORCE

# -----------------------------------------------------------------------------
# Packed conda environment staged to node-local storage
# -----------------------------------------------------------------------------
ENV_TARBALL="${ENV_TARBALL:-/lustre/orion/gen243/proj-shared/matilderestelli/pytorch_mvapich/conda_env.tar.gz}"
ENV_DIR="/mnt/bb/${USER}/torch_env"
TARBALL_DST="/mnt/bb/${USER}/torch_env.tar.gz"

echo "Using ENV_TARBALL=$ENV_TARBALL"
echo "Using ENV_DIR=$ENV_DIR"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 mkdir -p "$NVME_BASE"

sbcast -pf "$ENV_TARBALL" "$TARBALL_DST"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 rm -rf "$ENV_DIR"
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 mkdir -p "$ENV_DIR"
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 -c "${SLURM_CPUS_PER_TASK}" \
  tar --use-compress-program=pigz -xf "$TARBALL_DST" -C "$ENV_DIR"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
  conda activate '$ENV_DIR'
  conda-unpack
"

source /autofs/nccs-svm1_sw/frontier/miniforge3/23.11.0-0/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

# -----------------------------------------------------------------------------
# ROCm discovery from loaded environment
# -----------------------------------------------------------------------------
if command -v hipcc >/dev/null 2>&1; then
  export ROCM_HOME="$(dirname "$(dirname "$(command -v hipcc)")")"
elif [[ -n "${ROCM_PATH:-}" ]]; then
  export ROCM_HOME="${ROCM_PATH}"
else
  export ROCM_HOME="/opt/rocm"
fi

ROCM_LIBDIR="$ROCM_HOME/lib"
[[ -d "$ROCM_LIBDIR" ]] || ROCM_LIBDIR="$ROCM_HOME/lib64"

# -----------------------------------------------------------------------------
# Optional HIP soname compatibility shim for PyTorch envs expecting another soname
# -----------------------------------------------------------------------------
export ROCM_COMPAT_DIR="/mnt/bb/${USER}/rocm_compat_${SLURM_JOB_ID}"

srun -N "${SLURM_NNODES}" --ntasks-per-node=1 bash -lc "
  set -euo pipefail
  mkdir -p '${ROCM_COMPAT_DIR}'
  ROCM_LIBDIR='${ROCM_LIBDIR}'
  if [[ -f \"\$ROCM_LIBDIR/libamdhip64.so.7\" ]]; then
    ln -sf \"\$ROCM_LIBDIR/libamdhip64.so.7\" '${ROCM_COMPAT_DIR}/libamdhip64.so.7'
    ln -sf \"\$ROCM_LIBDIR/libamdhip64.so.7\" '${ROCM_COMPAT_DIR}/libamdhip64.so.6'
    ln -sf \"\$ROCM_LIBDIR/libamdhip64.so.7\" '${ROCM_COMPAT_DIR}/libamdhip64.so.5'
  elif [[ -f \"\$ROCM_LIBDIR/libamdhip64.so.6\" ]]; then
    ln -sf \"\$ROCM_LIBDIR/libamdhip64.so.6\" '${ROCM_COMPAT_DIR}/libamdhip64.so.6'
    ln -sf \"\$ROCM_LIBDIR/libamdhip64.so.6\" '${ROCM_COMPAT_DIR}/libamdhip64.so.5'
  fi
"

# -----------------------------------------------------------------------------
# Library path setup after conda activation
# -----------------------------------------------------------------------------
TORCH_LIBDIR=""
for cand in \
  "$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib" \
  "$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib" \
  "$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib"
do
  if [[ -d "$cand" ]]; then
    TORCH_LIBDIR="$cand"
    break
  fi
done

export LD_LIBRARY_PATH="${ROCM_COMPAT_DIR}${TORCH_LIBDIR:+:$TORCH_LIBDIR}:$CONDA_PREFIX/lib:$ROCM_LIBDIR:/opt/cray/libfabric/2.3.1/lib64:${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# -----------------------------------------------------------------------------
# Debug info
# -----------------------------------------------------------------------------
module list
env | grep -E '^(MVP|MV2|MPICH|FI_|NCCL_|ROCM|MIOPEN_)' || true

which mpicc || true
which mpirun || true
which python || true
command -v hipcc || true

# -----------------------------------------------------------------------------
# Python sanity checks
# -----------------------------------------------------------------------------
python - <<'PY'
import os
import sys

print("Python executable:", sys.executable)
print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("ROCM_HOME:", os.environ.get("ROCM_HOME"))

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("Location:", torch.__file__)
    print("HIP version:", getattr(torch.version, "hip", None))
    print("HIP available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
except Exception as e:
    print("Torch import failed:", repr(e))

try:
    import torchvision
    print("torchvision:", torchvision.__version__)
    print("has nms:", hasattr(torch.ops.torchvision, "nms"))
except Exception as e:
    print("torchvision import failed:", repr(e))

try:
    import torch.distributed as dist
    print("MPI available:", dist.is_mpi_available())
except Exception as e:
    print("torch.distributed check failed:", repr(e))
PY

# -----------------------------------------------------------------------------
# Optional: build a tiny GPU-aware MPI validation test
# Set BUILD_GPU_AWARE_MPI_TEST=1 to compile it
# -----------------------------------------------------------------------------
if [[ "${BUILD_GPU_AWARE_MPI_TEST:-0}" == "1" ]]; then
  TEST_SRC="${PWD}/gpu_aware_mpi_test.cpp"
  TEST_EXE="${PWD}/gpu_aware_mpi_test.exe"

  cat > "$TEST_SRC" <<'EOF'
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define HIP_CHECK(cmd) do {                                      \
  hipError_t e = cmd;                                            \
  if (e != hipSuccess) {                                         \
    fprintf(stderr, "HIP error %s:%d: %s\n",                     \
            __FILE__, __LINE__, hipGetErrorString(e));           \
    std::abort();                                                \
  }                                                              \
} while (0)

__global__ void fill_kernel(float* x, float v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev <= 0) {
    if (rank == 0) fprintf(stderr, "No HIP devices found\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  HIP_CHECK(hipSetDevice(rank % ndev));

  const int N = 16;
  float* d_buf = nullptr;
  float h_buf[N];

  HIP_CHECK(hipMalloc(&d_buf, N * sizeof(float)));
  fill_kernel<<<1, 64>>>(d_buf, float(rank + 1), N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int rc = MPI_Allreduce(MPI_IN_PLACE, d_buf, N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "Rank %d: MPI_Allreduce failed\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  HIP_CHECK(hipMemcpy(h_buf, d_buf, N * sizeof(float), hipMemcpyDeviceToHost));

  const float expected = float(size * (size + 1) / 2);
  int ok = 1;
  for (int i = 0; i < N; ++i) {
    if (std::fabs(h_buf[i] - expected) > 1e-3) ok = 0;
  }

  if (rank == 0) {
    if (ok) {
      printf("GPU-aware MPI Allreduce appears to work. Expected=%f got=%f\n",
             expected, h_buf[0]);
    } else {
      printf("GPU-aware MPI test failed. Expected=%f got=%f\n",
             expected, h_buf[0]);
    }
  }

  hipFree(d_buf);
  MPI_Finalize();
  return ok ? 0 : 3;
}
EOF

  mpicxx -x hip -O2 "$TEST_SRC" -o "$TEST_EXE"
fi

# -----------------------------------------------------------------------------
# Optional: run the tiny GPU-aware MPI validation test
# Set RUN_GPU_AWARE_MPI_TEST=1 to run it
# -----------------------------------------------------------------------------
if [[ "${RUN_GPU_AWARE_MPI_TEST:-0}" == "1" ]]; then
  TEST_EXE="${PWD}/gpu_aware_mpi_test.exe"
  [[ -x "$TEST_EXE" ]] || { echo "Missing $TEST_EXE; set BUILD_GPU_AWARE_MPI_TEST=1 first"; exit 1; }

  # Default one rank per node; override if desired
  TEST_NTASKS="${TEST_NTASKS:-$SLURM_NNODES}"

  srun -N "${SLURM_NNODES}" -n "${TEST_NTASKS}" --ntasks-per-node=1 "$TEST_EXE"
fi
