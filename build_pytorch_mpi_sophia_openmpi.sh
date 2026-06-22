#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N pt-sophia-openmpi
#PBS -q by-gpu
#PBS -A UIC-HPC
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o pt-sophia-openmpi.pbs.out

set -euo pipefail
trap 'echo "[ERROR] Failed at line ${LINENO} at $(date)" >&2' ERR

# ---------- Safe defaults (works with set -u) ----------
: "${SOPHIA_BUILD_ROOT:=/lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build_sophia}"
: "${PYTORCH_GIT_URL:=https://github.com/pytorch/pytorch.git}"
: "${PYTORCH_GIT_COMMIT:=ba56102387e}"

: "${SOPHIA_PYTORCH_SOURCE:=${SOPHIA_BUILD_ROOT}/src/pytorch}"
: "${SOPHIA_PYTHON_PREFIX:=${SOPHIA_BUILD_ROOT}/conda/pt-sophia-openmpi}"
: "${PIP_CACHE_DIR:=${SOPHIA_BUILD_ROOT}/cache/pip}"

MAX_JOBS_DEFAULT="$(nproc 2>/dev/null || echo 16)"
: "${MAX_JOBS:=${PBS_NP:-$MAX_JOBS_DEFAULT}}"

# ---------- Proxies ----------
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export ftp_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="localhost,127.0.0.1,*.local,*.alcf.anl.gov,sophia-*,polaris-*,grand.alcf.anl.gov"

# ---------- Logging ----------
BUILD_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOBTAG="${PBS_JOBID:-nojobid}"
OUTDIR="${PBS_O_WORKDIR:-$PWD}"

mkdir -p "${SOPHIA_BUILD_ROOT}/"{src,conda,cache/pip,logs}
LOGFILE="${SOPHIA_BUILD_ROOT}/logs/pytorch_sophia_openmpi_${JOBTAG}_${BUILD_TIMESTAMP}.log"
RUNOUT="${OUTDIR}/pt-sophia-openmpi.${JOBTAG}.out"

# Redirect all output to files you can see
exec > >(tee -a "${RUNOUT}" "${LOGFILE}") 2>&1

echo "=========================================="
echo "Started   : $(date)"
echo "Node      : $(hostname)"
echo "JobID     : ${JOBTAG}"
echo "Workdir   : ${OUTDIR}"
echo "Buildroot : ${SOPHIA_BUILD_ROOT}"
echo "Prefix    : ${SOPHIA_PYTHON_PREFIX}"
echo "Source    : ${SOPHIA_PYTORCH_SOURCE}"
echo "MAX_JOBS  : ${MAX_JOBS}"
echo "=========================================="

# ---------- Modules / toolchain ----------
module purge
module use /soft/modulefiles
module load compilers/openmpi/5.0.10
module load gcc || true

# CUDA
if [[ -d /usr/local/cuda-12.9 ]]; then
  export CUDA_HOME=/usr/local/cuda-12.9
else
  export CUDA_HOME=/usr/local/cuda
fi
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "== Toolchain =="
command -v mpicc || true
mpicc --showme || true
command -v nvcc || true
nvcc --version | head -n 1 || true

# ---------- Python env (must already exist) ----------
if [[ ! -x "${SOPHIA_PYTHON_PREFIX}/bin/python" ]]; then
  echo "[ERROR] Conda env not found: ${SOPHIA_PYTHON_PREFIX}"
  exit 1
fi
export PATH="${SOPHIA_PYTHON_PREFIX}/bin:${PATH}"
hash -r

python - <<'PY'
import sys
print("Python:", sys.executable)
print("Version:", sys.version)
PY
python -m pip --version
export PIP_CACHE_DIR

# ---------- Get/refresh PyTorch source on Eagle ----------
if [[ ! -f "${SOPHIA_PYTORCH_SOURCE}/pyproject.toml" ]]; then
  echo "== Cloning PyTorch source to Eagle =="
  mkdir -p "$(dirname "${SOPHIA_PYTORCH_SOURCE}")"
  git clone "${PYTORCH_GIT_URL}" "${SOPHIA_PYTORCH_SOURCE}"
  git -C "${SOPHIA_PYTORCH_SOURCE}" checkout "${PYTORCH_GIT_COMMIT}"
  git -C "${SOPHIA_PYTORCH_SOURCE}" submodule sync --recursive
  git -C "${SOPHIA_PYTORCH_SOURCE}" submodule update --init --recursive --jobs 8
else
  echo "== Updating PyTorch submodules (Eagle tree) =="
  git -C "${SOPHIA_PYTORCH_SOURCE}" submodule update --init --recursive --jobs 8
fi

# ---------- Node-local build dir ----------
NODE_SCRATCH="/tmp/${USER}/pytorch_build_${JOBTAG}"
echo "== Using node-local build dir: ${NODE_SCRATCH}"
rm -rf "${NODE_SCRATCH}"
mkdir -p "${NODE_SCRATCH}"

echo "== /tmp free space =="
df -h /tmp || true

echo "== Copying source Eagle -> /tmp (excluding .git) =="

# Clean destination
rm -rf "${NODE_SCRATCH:?}/"*
mkdir -p "${NODE_SCRATCH}"

# Use tar stream copy (fast, preserves perms, no rsync needed)
tar -C "${SOPHIA_PYTORCH_SOURCE}" \
    --exclude='.git' \
    --exclude='build' \
    --exclude='**/__pycache__' \
    -cf - . \
  | tar -C "${NODE_SCRATCH}" -xf -


cd "${NODE_SCRATCH}"
rm -rf build/

export TMPDIR="${NODE_SCRATCH}/tmp"
mkdir -p "${TMPDIR}"

# ---------- Build configuration ----------
export USE_DISTRIBUTED=1
export USE_MPI=1
export USE_CUDA=1
export USE_NINJA=1
export BUILD_TEST=0

export USE_FLASH_ATTENTION=0
export USE_MEM_EFF_ATTENTION=0
export USE_FBGEMM=0
export USE_NNPACK=0
export USE_CUFILE=0

MPI_BIN="$(command -v mpicc)"
export MPI_HOME="$(dirname "$(dirname "${MPI_BIN}")")"
export CMAKE_PREFIX_PATH="${MPI_HOME}:${CMAKE_PREFIX_PATH:-}"

export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export CMAKE_CUDA_HOST_COMPILER=g++

export MAX_JOBS
export Python_EXECUTABLE="$(command -v python)"
export CMAKE_VERBOSE_MAKEFILE=1

echo "== Build env =="
echo "PWD       : $(pwd)"
echo "TMPDIR    : ${TMPDIR}"
echo "CUDA_HOME : ${CUDA_HOME}"
echo "MPI_HOME  : ${MPI_HOME}"
echo "CC/CXX    : ${CC} / ${CXX}"
echo "MAX_JOBS  : ${MAX_JOBS}"

echo "== Building and installing PyTorch (non-editable) =="
python -m pip install -v --no-build-isolation .

echo "== Verification =="
python - <<'PY'
import torch
import torch.distributed as dist
print("torch.__version__      =", torch.__version__)
print("torch.__file__         =", torch.__file__)
print("cuda available/devices =", torch.cuda.is_available(), torch.cuda.device_count())
print("torch.version.cuda     =", torch.version.cuda)
print("mpi available          =", dist.is_mpi_available())
if not dist.is_mpi_available():
    raise SystemExit("ERROR: MPI backend not compiled in")
PY

echo "== Done =="
echo "RUNOUT : ${RUNOUT}"
echo "LOGFILE: ${LOGFILE}"

rm -rf "${NODE_SCRATCH}"
