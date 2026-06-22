#!/bin/bash -l
#PBS -l select=1:ngpus=4
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -A UIC-HPC
#PBS -j oe
#PBS -o ddp_train.out
#PBS -N ddp-train

cd "${PBS_O_WORKDIR}"

# PBS does NOT substitute %j in #PBS -o. Use runtime jobid instead:
LOG="ddp_train.${PBS_JOBID}.out"
exec > >(tee -a "${LOG}") 2>&1

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== PBS_JOBID: ${PBS_JOBID} ==="

# Sophia env (replace Polaris env)
source "${PBS_O_WORKDIR}/envScriptSophia.sh"

# For enabling profiling time for communication hooks.
export DDP_HOOK_TIMING=1
export DDP_HOOK_TIMING_RANK0_ONLY=1

# Edit these lists to run multiple experiments inside one queued job.
# Leave only one active value in each list for a single run.

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
    # "2"
    "20"
    # "50"
    # "100"
)

BATCH_SIZES=(
    "32"       # strong global 128 on 4 GPUs
    # "64"    # strong global 256 on 4 GPUs
    # "128"   # weak scaling local batch 128
)

LEARNING_RATES=(
    # "0.0001"
    "0.001"    # paper-style CIFAR reproduction
    # "0.01"
    # "0.05"
    # "0.1"
)

SCHEDULERS=(
    "constant" # paper-style LR=0.001 reproduction
    # "cosine"
)

WARMUP_EPOCHS_LIST=(
    "0"
    # "5"      # ImageNet-style from scratch
)

GRAD_CLIPS=(
    #"none"
    # "0.5"
    "1.0"
)

PRETRAINED_VALUES=(
    "false"    # from scratch
    # "true"   # fine-tuning
)

CIFAR_STEM_VALUES=(
    "true"     # CIFAR from scratch
    # "false"  # ImageNet / ImageNet-like
)

DROP_LAST_VALUES=(
    "false"
    # "true"
)

BACKENDS=(
    "mpi"
    # "nccl"
)

# "default" = built-in DDP allreduce wrapped with NVTX timing.
# "none"    = no custom communication hook.
EXPERIMENTS=(
    # "default:"
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

        # Sophia/OpenMPI: use -x instead of MPICH-style -env, and drop MPICH_GPU_SUPPORT_ENABLED
        MPI_ENV_ARGS=(
            -x LD_PRELOAD
            -x LD_LIBRARY_PATH
            -x PYTHONPATH

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
            -x NUM_WORKERS="4"
            -x PIN_MEMORY="true"
            -x DROP_LAST="${DROP_LAST}"
            -x PRETRAINED="${PRETRAINED}"
            -x CIFAR_STEM="${CIFAR_STEM}"
            -x INIT_CIFAR_STEM_FROM_PRETRAINED_CENTER="false"
            -x DATA_DIR="./data"
            -x CHECKPOINT_DIR="./checkpoints"
            -x SEED="42"
            -x BACKEND="${BACKEND}"
            -x COMM_ALGORITHM="${COMM_ALGORITHM}"
            -x DDP_HOOK_TIMING
            -x DDP_HOOK_TIMING_RANK0_ONLY
        )

        if [[ -n "${ZFP_RATE}" && "${ZFP_RATE}" != "${EXPERIMENT}" ]]; then
            MPI_ENV_ARGS+=(-x ZFP_RATE="${ZFP_RATE}")
        fi

        mpirun -np "${NUM_PROCS}" --map-by ppr:${PPN}:node --bind-to core \
            "${MPI_ENV_ARGS[@]}" \
            python interface.py

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

#to request an interactive node
# qsub -I -l select=1:ngpus=4 -l walltime=01:00:00 -l filesystems=home:eagle -q debug -A UIC-HPC
