#!/bin/bash -l
#PBS -l select=8
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -A UIC-HPC
#PBS -j oe
#PBS -o train.%j.out
#PBS -N ddp-train

cd ${PBS_O_WORKDIR}

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source ${PBS_O_WORKDIR}/envScript3.sh

# Edit these lists to run multiple experiments inside one queued job.
# "default" = built-in DDP allreduce wrapped with NVTX timing.
# "none"    = no custom communication hook.
BACKENDS=(
    "mpi"
    # "nccl"
)

COMM_ALGORITHMS=(
    "default"
    # "ring"
    # "recursive_doubling"
    # "ring_zfp_naive"
    # "recursive_doubling_zfp_naive"
    # "ring_zfp_online_coll"
    # "recursive_doubling_zfp_online_coll"
    # "none"
)

for BACKEND in "${BACKENDS[@]}"; do
    for COMM_ALGORITHM in "${COMM_ALGORITHMS[@]}"; do
        echo "=== Starting training: backend=${BACKEND}, hook=${COMM_ALGORITHM} at $(date) ==="

        mpiexec -np 8 --ppn 8 --depth=8 --cpu-bind depth \
            -env MPICH_GPU_SUPPORT_ENABLED=1 \
            -env LD_PRELOAD="${LD_PRELOAD}" \
            -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
            -env PYTHONPATH="${PYTHONPATH}" \
            -env BACKEND="${BACKEND}" \
            -env COMM_ALGORITHM="${COMM_ALGORITHM}" \
            python interface.py

        status=$?
        echo "=== Finished training: backend=${BACKEND}, hook=${COMM_ALGORITHM}, status=${status} at $(date) ==="

        if [[ ${status} -ne 0 ]]; then
            echo "=== Stopping job because training failed ==="
            exit ${status}
        fi
    done
done

echo "=== Job finished: $(date) ==="