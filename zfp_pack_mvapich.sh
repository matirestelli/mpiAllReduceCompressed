#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -J zfp_pack_mvapich
#SBATCH -p batch
#SBATCH -q debug 
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -o zfp_pack_mvapich.%j.out
#SBATCH -e zfp_pack_mvapich.%j.err

set -euo pipefail
trap 'echo "[ERROR] Failed at line ${LINENO} at $(date)" >&2' ERR

ENVROOT=/lustre/orion/gen243/proj-shared/matilderestelli/pytorch_mvapich
ENV_PREFIX="$ENVROOT/conda"
COMPAT_DIR="$ENVROOT/compat"
TARBALL="$ENVROOT/conda_env.tar.gz"
LOCAL_BASE="/tmp/${USER}/mvapich"
LOCAL_ENV="$LOCAL_BASE/conda"
ZFP_PREFIX=/ccs/home/matilderestelli/ddp-allreduce-eval-framework/zfp-install-frontier

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

export ROCM_HOME=/opt/rocm-7.1.1
export ROCM_PATH="$ROCM_HOME"
export HIP_PATH="$ROCM_HOME"
export HSA_PATH="$ROCM_HOME"

unset MPICH_GPU_SUPPORT_ENABLED || true
unset MV2_USE_CUDA || true
export MV2_USE_ROCM=1
export MV2_SHOW_ENV_INFO=1

export MPI_HOME="$(dirname "$(dirname "$(command -v mpicc)")")"

mkdir -p "$COMPAT_DIR"
ln -sf "$ROCM_HOME/lib/libamdhip64.so.7" "$COMPAT_DIR/libamdhip64.so.6"

conda activate "$ENV_PREFIX"

TORCH_LIBDIR="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib"
export PATH="$ENV_PREFIX/bin:$ROCM_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="$COMPAT_DIR:$TORCH_LIBDIR:$MPI_HOME/lib:/opt/cray/libfabric/2.3.1/lib64:$ROCM_HOME/lib:$ROCM_HOME/lib/roctracer:$ROCM_HOME/lib/rocprofiler:$LD_LIBRARY_PATH"

if [[ -d "$ZFP_PREFIX/lib64" ]]; then
  ZFP_LIBDIR="$ZFP_PREFIX/lib64"
else
  ZFP_LIBDIR="$ZFP_PREFIX/lib"
fi

python - <<'PY'
import torch, torchvision, torch.distributed as dist
import framework_allreduce_zfp_hip as zfp_ext
print("torch:", torch.__version__, torch.__file__)
print("torchvision:", torchvision.__version__, torchvision.__file__)
print("hip:", torch.version.hip)
print("mpi:", dist.is_mpi_available())
print("zfp ext:", zfp_ext.__file__)
PY

command -v conda-pack >/dev/null 2>&1 || { echo "ERROR: conda-pack missing"; exit 3; }

rm -f "$TARBALL"
conda-pack -p "$ENV_PREFIX" -o "$TARBALL"
ls -lh "$TARBALL"

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

python - <<'PY'
import torch, torchvision, torch.distributed as dist
import framework_allreduce_zfp_hip as zfp_ext
print("RUNTIME torch:", torch.__version__, "| hip:", torch.version.hip)
print("RUNTIME torchvision:", torchvision.__version__)
print("RUNTIME MPI available:", dist.is_mpi_available())
print("RUNTIME zfp ext file:", zfp_ext.__file__)
PY

echo "DONE"
