#!/bin/bash
# Polaris baseline_mpi environment (your working strategy)

module purge
module load PrgEnv-nvidia/8.6.0 cuda/12.9
module load craype-accel-nvidia80 2>/dev/null || true

# Conda activate (shared HOME = instant!)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate baseline_mpi

# Threading safety (OpenBLAS fix)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export GOTO_NUM_THREADS=1 BLIS_NUM_THREADS=1
export MPICH_GPU_SUPPORT=1
export FI_CXI_DEFAULT_CQ_SIZE=131072

# Verify
echo "========================================="
echo "Baseline MPI environment loaded!"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "Pandas: $(python -c 'import pandas; print(pandas.__version__)' 2>&1)"
echo "Python: $(which python)"
echo "========================================="
