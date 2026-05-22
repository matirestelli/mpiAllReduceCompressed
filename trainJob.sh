#!/bin/bash -l
#PBS -l select=4:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A UIC-HPC
#PBS -j oe
#PBS -o ring_hooks_online_10.%j.out
#PBS -N ddp-train

cd ${PBS_O_WORKDIR}

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source ${PBS_O_WORKDIR}/envScript3.sh

#for enabling profiling time for communication hooks:
export DDP_HOOK_TIMING=1
export DDP_HOOK_TIMING_RANK0_ONLY=1

# Edit these lists to run multiple experiments inside one queued job.
# "default" = built-in DDP allreduce wrapped with NVTX timing.
# "none"    = no custom communication hook.
BACKENDS=(
    "mpi"
    # "nccl"
)

EXPERIMENTS=(
    # "None:"
    # "ring:"
    # "ring_zfp_naive:16"
    # "ring_zfp_online_coll:16"
    "ring_zfp_online_coll:10"
    # "recursive_doubling:"
    # "recursive_doubling_zfp_naive:16"
    # "recursive_doubling_zfp_online_coll:16"
    # "recursive_doubling_zfp_online_coll:8"
)

for BACKEND in "${BACKENDS[@]}"; do
    for EXPERIMENT in "${EXPERIMENTS[@]}"; do
        COMM_ALGORITHM="${EXPERIMENT%%:*}"
        ZFP_RATE="${EXPERIMENT#*:}"

        echo "=== Starting training: backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp_rate=${ZFP_RATE:-none} at $(date) ==="

        MPI_ENV_ARGS=(
            -env MPICH_GPU_SUPPORT_ENABLED=1
            -env LD_PRELOAD="${LD_PRELOAD}"
            -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
            -env PYTHONPATH="${PYTHONPATH}"
            -env BACKEND="${BACKEND}"
            -env COMM_ALGORITHM="${COMM_ALGORITHM}"
        )

        if [[ -n "${ZFP_RATE}" ]]; then
            MPI_ENV_ARGS+=(-env ZFP_RATE="${ZFP_RATE}")
        fi

        mpiexec -np 16 --ppn 4 --depth=8 --cpu-bind depth \
            "${MPI_ENV_ARGS[@]}" \
            python interface.py

        status=$?
        echo "=== Finished training: backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp_rate=${ZFP_RATE:-none}, status=${status} at $(date) ==="

        if [[ ${status} -ne 0 ]]; then
            echo "=== Stopping job because training failed ==="
            exit ${status}
        fi
    done
done

echo "=== Job finished: $(date) ==="