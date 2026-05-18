#!/bin/bash -l
#PBS -N build_zfp_cuda
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -q debug
#PBS -A UIC-HPC

set -euo pipefail

cd "${PBS_O_WORKDIR}"

unset LD_PRELOAD

module purge
module use /soft/modulefiles
module load PrgEnv-gnu
module load gcc-native/13
module load cuda/12.9
module load conda/2025-09-28

conda activate base

echo "Running on host: $(hostname)"
nvidia-smi

ZFP_SRC="${PBS_O_WORKDIR}/zfp"
ZFP_INSTALL="${PBS_O_WORKDIR}/zfp-install"

if [[ ! -d "${ZFP_SRC}" ]]; then
    echo "[ERROR] ZFP source not found at ${ZFP_SRC}"
    exit 1
fi

rm -rf "${ZFP_SRC}/build-cuda" "${ZFP_INSTALL}"

cmake -S "${ZFP_SRC}" -B "${ZFP_SRC}/build-cuda" \
  -DCMAKE_INSTALL_PREFIX="${ZFP_INSTALL}" \
  -DBUILD_SHARED_LIBS=ON \
  -DZFP_WITH_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUDA_NVCC_FLAGS="-arch=sm_80"

cmake --build "${ZFP_SRC}/build-cuda" -j 8
cmake --install "${ZFP_SRC}/build-cuda"

export ZFP_HOME="${ZFP_INSTALL}"
export LD_LIBRARY_PATH="${ZFP_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "ZFP_HOME=${ZFP_HOME}"
ls -l "${ZFP_HOME}/include/zfp.h"
ls -l "${ZFP_HOME}/lib64/libzfp.so"

echo "ZFP CUDA build completed."