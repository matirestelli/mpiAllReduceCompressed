#!/bin/bash
# Polaris MPI PyTorch (Your Original + Fixes)

module purge
module load PrgEnv-nvidia/8.6.0 cuda/12.9 craype-accel-nvidia80  # Key: cuda before accel!

# Proxy for pip
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export no_proxy="localhost,*.alcf.anl.gov,polaris-*"

# Conda + PyTorch
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate baseline_mpi  # Your env w/ mpi4py fix

# GPU-Aware MPI (Critical)
export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_ACCEL_TARGET=nvidia80

# Threading/Scale (Your Original)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 GOTO_NUM_THREADS=1 BLIS_NUM_THREADS=1
export FI_CXI_DEFAULT_CQ_SIZE=131072

echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
python -c "import torch.distributed as dist; print('MPI backend:', 'Yes' if dist.is_mpi_available() else 'No')" 
