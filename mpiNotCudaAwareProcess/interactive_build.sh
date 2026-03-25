#!/bin/bash
set -e

# Activate conda
source /soft/applications/conda/2025-09-25/mconda3/bin/activate
conda activate /lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build/conda/pt28

# Set up MPI environment
export MPI_HOME=/opt/cray/pe/mpich/9.0.1/ofi/nvidia/23.3
export MPICH_GPU_SUPPORT_ENABLED=1
export USE_MPI=1
export MPI_INCLUDE_DIR=$MPI_HOME/include
export MPI_LIBRARIES=$MPI_HOME/lib/libmpi.so
export CC=cc
export CXX=CC
export CUDAHOSTCXX=cc
export TORCH_CUDA_ARCH_LIST="8.0"
export USE_NINJA=1
export MAX_JOBS=4

# Load Polaris modules
module purge
module load PrgEnv-gnu/8.6.0
module load gcc-native/12.3
module load cuda/12.9
module load cray-mpich
module load craype-accel-nvidia80

# Build PyTorch
cd /lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build/src/pytorch
rm -rf build

echo "=========================================="
echo "MPI Environment Variables:"
echo "  MPI_HOME=$MPI_HOME"
echo "  MPI_INCLUDE_DIR=$MPI_INCLUDE_DIR"
echo "  MPI_LIBRARIES=$MPI_LIBRARIES"
echo "  MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED"
echo "  USE_MPI=$USE_MPI"
echo "=========================================="

python -m pip install -v --no-build-isolation -e . 2>&1 | tee /lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build/logs/interactive_build.log
