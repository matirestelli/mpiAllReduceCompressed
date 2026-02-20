#!/bin/bash -l
#PBS -l select=1:ngpus=4             
#PBS -l walltime=00:30:00             
#PBS -l filesystems=home:grand        
#PBS -q debug-scaling                  
#PBS -A UIC-HPC               
#PBS -j n              
#PBS -o baseline.%j.out               
#PBS -e baseline.%j.err 
#PBS -N baseline-ddp

cd ${PBS_O_WORKDIR}
echo "Staged files:"; ls -la *.py *.sh

# Load environment (your working pattern!)
source polaris_baseline_env.sh

# Your existing MPI/thread vars (safe redundancy)
export OMP_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1

echo "Starting DDP baseline: 4 GPUs" | tee -a baseline.log

# Kill threading (your original)
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1

mpirun -np 4 python baselineTraining.py \
  --models wideresnet resnext convnext \
  --batch-size 128 \
  --lr 0.001 \
  --epochs 5

echo "Done! Check results_4gpus_5epochs.csv + baseline.*.{out,err,log}"
