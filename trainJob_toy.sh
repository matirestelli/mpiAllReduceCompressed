#!/bin/bash -l
#PBS -l select=1:ngpus=4
#PBS -l walltime=00:15:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A UIC-HPC
#PBS -j oe
#PBS -o toy_train.%j.out
#PBS -N ddp-toy

cd ${PBS_O_WORKDIR}

# Override with: ALGO=recursive_doubling_zfp qsub trainJob_toy.sh
ALGO=${ALGO:-recursive_doubling_zfp}

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== Algorithm: ${ALGO} ==="

source ${PBS_O_WORKDIR}/envScript3.sh

# Clear pycache so JIT picks up any .py changes
rm -rf ${PBS_O_WORKDIR}/__pycache__

echo "=== Starting toy_train (algorithm: ${ALGO}) ==="
mpirun -np 4 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    -env LD_PRELOAD="${LD_PRELOAD}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    python toy_train.py ${ALGO}

echo "=== Job finished: $(date) ==="
