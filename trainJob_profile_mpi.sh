#!/bin/bash -l
#PBS -l select=1:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A UIC-HPC
#PBS -j oe
#PBS -o train_profile_mpi.%j.out
#PBS -N ddp-profile-mpi

cd ${PBS_O_WORKDIR}

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source ${PBS_O_WORKDIR}/envScript3.sh

echo "=== Starting MPI profiling run ==="
mpirun -np 4 --ppn 4 --depth=8 --cpu-bind depth \
    -env MPICH_GPU_SUPPORT_ENABLED=1 \
    -env DDP_PROFILE_NVTX=1 \
    -env TMPDIR="/home/${USER}" \
    -env LD_PRELOAD="${LD_PRELOAD}" \
    -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
    -env PYTHONPATH="${PYTHONPATH}" \
    nsys profile \
        -o mpi_profile_%q{PMI_RANK} \
        --stats=true \
        --trace=cuda,nvtx \
        python interface.py
        # NOTE: --trace=mpi omitted — nsys's async MPI collective tracing segfaults
        # (exit 139) on Cray MPICH when tracking MPI_Iallreduce completions on a
        # background thread (used by the default hook's dist.all_reduce(async_op=True)).
        # NOTE: --trace=osrt omitted — OS runtime tracing floods the nsys buffer with
        # ~226K syscall/mutex events per 25s (from MPI background threads), causing
        # collection to stop before backward runs.  cuda,nvtx is sufficient to see
        # forward/backward/allreduce CUDA kernels and NVTX iteration ranges.
        # NOTE: --cuda-event-trace=false removed — PyTorch uses cudaEventRecord as
        # stream fences to coordinate async allreduce; suppressing them breaks kernel
        # timeline attribution and hides backward CUDA kernels in the report.
        # To restore MPI-level visibility when profiling ring: add --trace=mpi back.

echo "=== Job finished: $(date) ==="
