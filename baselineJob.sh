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
echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
echo "Staged files:"
ls -la *.py *.sh
source ${PBS_O_WORKDIR}/envScriptV2.sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=1
echo "=== Starting DDP baseline: 4 GPUs ==="
mpiexec -n 4 -ppn 4 \
--env LD_PRELOAD="${LD_PRELOAD}" \
--env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
--env MPICH_GPU_SUPPORT_ENABLED=1 \
--env OMP_NUM_THREADS=1 \
/soft/applications/conda/2025-09-28/mconda3/bin/python baselineTraining.py \
--models resnet50 resnext101 convnext_base \
--batch-size 128 \
--lr 0.001 \
--epochs 10
echo "=== Job finished: $(date) ==="

qsub -I -l select=1:ngpus=4 -l walltime=01:00:00 -l filesystems=home:grand -q debug-scaling -A UIC-HPC

rm -rf ~/.vscode-server/data/User/globalStorage/anthropic.claude-code/