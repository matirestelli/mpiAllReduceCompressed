#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -J ddp-train             
#SBATCH -N 1  
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8                  
#SBATCH -t 01:30:00   
#SBATCH -C nvme            
#SBATCH -o ddp_train_frontier.%j.out       
#SBATCH -e ddp_train_frontier.%j.err       

# srun <your_program>
# Frontier modules

#ask for a compute node 
# salloc -A gen243 -p batch -J installingPyTorch -N 1 --gres=gpu:4 -t 01:00:00 -o installingPyTorch.%j.out -e installingPyTorch.%j.out

# salloc -A gen243 -p batch -J trytrain -N 1 -t 00:30:00
# NB to use nvme you need to request that with the allocation, if sbatch sh job launche out there if interactive terminal :
# salloc -A gen243 -p batch -J trytrain -N 1 -t 00:30:00 -C nvme

cd "${SLURM_SUBMIT_DIR}"

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source envScriptFrontier.sh

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
    #"0.0001"
    "0.001"    # paper-style CIFAR reproduction
    # "0.01"
    # "0.05"
    # "0.1"
)

SCHEDULERS=(
    # "constant" # paper-style LR=0.001 reproduction
    "cosine"
)

WARMUP_EPOCHS_LIST=(
    "0"
    # "5"      # ImageNet-style from scratch
)

GRAD_CLIPS=(
    "none"
    # "0.5"
    # "1.0"
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
    #"ring_zfp_online_coll:16"
    #"ring_zfp_online_coll:10"
    #"recursive_doubling:"
    #"recursive_doubling_zfp_naive:16"
    #"recursive_doubling_zfp_online_coll:16"
    #"recursive_doubling_zfp_online_coll:8"
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

        export MODEL_NAME="${MODEL_NAME}"
        export DATASET="${DATASET}"
        export NUM_CLASSES="${NUM_CLASSES}"
        export IMAGE_SIZE="${IMAGE_SIZE}"
        export NUM_EPOCHS="${NUM_EPOCHS}"
        export BATCH_SIZE="${BATCH_SIZE}"
        export LEARNING_RATE="${LEARNING_RATE}"
        export MOMENTUM="0.9"
        export WEIGHT_DECAY="5e-4"
        export GRAD_CLIP="${GRAD_CLIP}"
        export SCHEDULER="${SCHEDULER}"
        export WARMUP_EPOCHS="${WARMUP_EPOCHS}"
        export NUM_WORKERS="4"
        export PIN_MEMORY="true"
        export DROP_LAST="${DROP_LAST}"
        export PRETRAINED="${PRETRAINED}"
        export CIFAR_STEM="${CIFAR_STEM}"
        export INIT_CIFAR_STEM_FROM_PRETRAINED_CENTER="false"
        export DATA_DIR="./data"
        export CHECKPOINT_DIR="./checkpoints"
        export SEED="42"
        export BACKEND="${BACKEND}"
        export COMM_ALGORITHM="${COMM_ALGORITHM}"

        export MIOPEN_FIND_MODE=2
        export MIOPEN_FIND_ENFORCE=1

        export TORCH_DISTRIBUTED_DEBUG=DETAIL
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

        if [[ -n "${ZFP_RATE}" ]]; then
            export ZFP_RATE="${ZFP_RATE}"
        else
            unset ZFP_RATE
        fi

        unset SRUN_CPUS_PER_TASK
        unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

        #check of correct use for the first time of persostend MIOpen cache directory
        echo "MIOPEN_USER_DB_PATH=${MIOPEN_USER_DB_PATH:-UNSET}"
        echo "MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_CUSTOM_CACHE_DIR:-UNSET}"
        [ -n "${MIOPEN_USER_DB_PATH:-}" ] && ls -ld "$MIOPEN_USER_DB_PATH" || true

        echo "==== GPU visibility debug (batch shell) ===="
        hostname
        env | egrep 'ROCR_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL|SLURM_JOB_GPUS|SLURM_STEP_GPUS' || true
        command -v rocminfo >/dev/null 2>&1 && rocminfo | head -n 30 || true

        srun --exact -n "${NUM_PROCS}" --ntasks-per-node="${PPN}" --cpus-per-task=8 python interface.py

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

# salloc -A gen243 -p batch -J torch-build-and-pack -N 1 -t 00:30:00 -c 16 -C nvme
# squeue -j <jobid> -o "%.18i %.9P %.10T %.8u %.10M %.20S %.6D %R"
