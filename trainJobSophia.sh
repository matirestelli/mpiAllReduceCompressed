#!/bin/bash -l
#PBS -l select=1:ngpus=4
#PBS -l walltime=05:00:00
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -A UIC-HPC
#PBS -o ddp_trainSophia.out
#PBS -e ddp_trainSophia.err
#PBS -N ddp-trainSophia

cd "${PBS_O_WORKDIR}"

# PBS does NOT substitute %j in #PBS -o. Use runtime jobid instead:
LOG="ddp_train.${PBS_JOBID}.out"
exec > >(tee -a "${LOG}") 2>&1

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== PBS_JOBID: ${PBS_JOBID} ==="

# Sophia env (replace Polaris env)
source "${PBS_O_WORKDIR}/envScriptSophia.sh"

# quick sanity
python -c "import torch, torch.distributed as dist; print(torch.__version__, torch.__file__); print('mpi', dist.is_mpi_available())"
python -c "import torch, torch.distributed as dist; print(torch.__version__, torch.__file__); print('CUDA built:', torch.version.cuda); print('CUDA visible:', torch.cuda.is_available(), torch.cuda.device_count()); print('MPI:', dist.is_mpi_available()); print('NCCL:', dist.is_nccl_available())"
python -c "from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])"

export NVIDIA_TF32_OVERRIDE=1

# Prevent CPU oversubscription across 4 MPI ranks (+ dataloader workers)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# For enabling profiling time for communication hooks.
export DDP_HOOK_TIMING=1
export DDP_HOOK_TIMING_RANK0_ONLY=1

# Benchmarking toggles
export DDP_PERF_DEBUG=1
export DDP_PERF_GPU_UTIL=0
export DDP_PERF_MEASURE_EVERY=50


export DATA_DIR="/eagle/UIC-HPC/mrest/mpiAllReduceCompressed/data/cifar"

NUM_PROCS=4
PPN=4

MODELS=(
    "wide_resnet50_2"
    # "resnext101_32x8d"
)

DATASETS=(
    "cifar10"
    # "imagenet"
)

NUM_CLASSES_LIST=(
    "10"       # CIFAR-10
    # "1000"  # ImageNet
)

IMAGE_SIZES=(
    "32"       # CIFAR
    # "224"   # ImageNet / ImageNet-like
)

NUM_EPOCHS_LIST=(
    "20"
)

BATCH_SIZES=(
    "16"
    "32"       # strong global 128 on 4 GPUs
)

LEARNING_RATES=(
    "0.001"    # paper-style CIFAR reproduction
)

SCHEDULERS=(
    "cosine"
)

WARMUP_EPOCHS_LIST=(
    "0"
)

GRAD_CLIPS=(
    "none"
)

PRETRAINED_VALUES=(
    "true"   # fine-tuning
)

CIFAR_STEM_VALUES=(
    "false"  # ImageNet / ImageNet-like
)

DROP_LAST_VALUES=(
    "false"
)

BACKENDS=(
    "mpi"
)

# "default" = built-in DDP allreduce wrapped with NVTX timing.
# "none"    = no custom communication hook.
EXPERIMENTS=(
    "none:"
    "ring:"
    "ring_zfp_naive:16"
    "ring_zfp_online_coll:16"
    "ring_zfp_online_coll:10"
    "recursive_doubling:"
    "recursive_doubling_zfp_naive:16"
    "recursive_doubling_zfp_online_coll:16"
    "recursive_doubling_zfp_online_coll:8"
)

for MODEL_NAME in "${MODELS[@]}"; do
for DATASET in "${DATASETS[@]}"; do
for NUM_CLASSES in "${NUM_CLASSES_LIST[@]}"; do
for IMAGE_SIZE in "${IMAGE_SIZES[@]}"; do
for NUM_EPOCHS in "${NUM_EPOCHS_LIST[@]}"; do
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
for SCHEDULER in "${SCHEDULERS[@]}"; do
for WARMUP_EPOCHS in "${WARMUP_EPOCHS_LIST[@]}"; do
for GRAD_CLIP in "${GRAD_CLIPS[@]}"; do
for PRETRAINED in "${PRETRAINED_VALUES[@]}"; do
for CIFAR_STEM in "${CIFAR_STEM_VALUES[@]}"; do
for DROP_LAST in "${DROP_LAST_VALUES[@]}"; do
for BACKEND in "${BACKENDS[@]}"; do
for EXPERIMENT in "${EXPERIMENTS[@]}"; do

    COMM_ALGORITHM="${EXPERIMENT%%:*}"
    ZFP_RATE="${EXPERIMENT#*:}"

    echo "=== Starting training: model=${MODEL_NAME}, dataset=${DATASET}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}, scheduler=${SCHEDULER}, backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp_rate=${ZFP_RATE:-none} at $(date) ==="

    # Sophia/OpenMPI: use -x instead of MPICH-style -env
    MPI_ENV_ARGS=(

        -x PATH
        # -x LD_PRELOAD -> empty
        -x LD_LIBRARY_PATH
        -x PYTHONNOUSERSITE
        -x PYTHONPATH 
        -x CUDA_HOME
        -x UCX_ROOT
        -x TMPDIR
        -x OMPI_MCA_prte_tmpdir_base
        -x PRTE_MCA_prte_tmpdir_base

        -x OMP_NUM_THREADS
        -x MKL_NUM_THREADS
        -x OPENBLAS_NUM_THREADS
        -x NUMEXPR_NUM_THREADS

        -x DDP_HOOK_TIMING
        -x DDP_HOOK_TIMING_RANK0_ONLY
        -x DDP_PERF_DEBUG
        -x DDP_PERF_GPU_UTIL

        -x MODEL_NAME="${MODEL_NAME}"
        -x DATASET="${DATASET}"
        -x NUM_CLASSES="${NUM_CLASSES}"
        -x IMAGE_SIZE="${IMAGE_SIZE}"
        -x NUM_EPOCHS="${NUM_EPOCHS}"
        -x BATCH_SIZE="${BATCH_SIZE}"
        -x LEARNING_RATE="${LEARNING_RATE}"
        -x MOMENTUM="0.9"
        -x WEIGHT_DECAY="5e-4"
        -x GRAD_CLIP="${GRAD_CLIP}"
        -x SCHEDULER="${SCHEDULER}"
        -x WARMUP_EPOCHS="${WARMUP_EPOCHS}"

        # Diagnostic: eliminate dataloader multiprocessing; later try 1–2
        -x NUM_WORKERS="0"
        -x PIN_MEMORY="true"

        -x DROP_LAST="${DROP_LAST}"
        -x PRETRAINED="${PRETRAINED}"
        -x CIFAR_STEM="${CIFAR_STEM}"
        -x INIT_CIFAR_STEM_FROM_PRETRAINED_CENTER="false"
        -x DATA_DIR="${DATA_DIR}"        
        -x CHECKPOINT_DIR="./checkpoints"
        -x SEED="42"
        -x BACKEND="${BACKEND}"
        -x COMM_ALGORITHM="${COMM_ALGORITHM}"
    )

    if [[ -n "${ZFP_RATE}" && "${ZFP_RATE}" != "${EXPERIMENT}" ]]; then
        MPI_ENV_ARGS+=(-x ZFP_RATE="${ZFP_RATE}")
    fi
       
      # Recommended binding for 1 node / 4 GPUs: bind each rank to a core set
    # If you later use dataloader workers >0, binding becomes even more important.
    mpirun -np "${NUM_PROCS}" \
        --map-by ppr:${PPN}:node:PE=8 \
        --bind-to core \
        "${MPI_ENV_ARGS[@]}" \
        python -u interface.py


    # Diagnostic: disable binding (binding + dataloader workers often kills throughput)
   # mpirun -np "${NUM_PROCS}" --map-by ppr:${PPN}:node --bind-to none \
    #    "${MPI_ENV_ARGS[@]}" \
   #     python interface.py

    status=$?
    echo "=== Finished training: model=${MODEL_NAME}, dataset=${DATASET}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}, scheduler=${SCHEDULER}, backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp_rate=${ZFP_RATE:-none}, status=${status} at $(date) ==="

    if [[ ${status} -ne 0 ]]; then
        echo "=== Stopping job because training failed ==="
        exit ${status}
    fi

done
done
done
done
done
done
done
done
done
done
done
done
done
done
done

echo "=== Job finished: $(date) ==="

# to request an interactive node:
# qsub -I -l select=1:ngpus=1 -l walltime=01:00:00 -l filesystems=home:eagle -q by-gpu -A UIC-HPC
