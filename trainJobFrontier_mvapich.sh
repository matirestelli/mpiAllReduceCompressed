#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -J ddp-train-frontier_mvapich_8gpus_16b_64b
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --gpu-bind=closest
#SBATCH -t 02:00:00
#SBATCH -C nvme
#SBATCH -o ddp_train_frontier_mvapich_8gpus_16b_64b.%j.out
#SBATCH -e ddp_train_frontier_mvapich_8gpus_16b_64b.%j.err      


set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

#
# Load your MVAPICH-Plus / ROCm / Python env here.
# Replace this with your actual env setup file if needed.
#
source envScriptFrontier_mvapich.sh

#
# DDP / training diagnostics
#
export DDP_HOOK_TIMING=1
export DDP_HOOK_TIMING_RANK0_ONLY=1
export PRETRAINED_WEIGHTS_CACHE=/lustre/orion/gen243/proj-shared/matilderestelli/pretrained_weights_cache
export DDP_ITER_LOG=1
export DDP_PROFILE_BARRIER=0

#
# Important: match known-good Frontier NIC policy from your Cray MPICH script
#
export MPICH_OFI_NIC_POLICY=GPU

#
# MVAPICH-Plus GPU-aware settings: keep minimal for first debug pass
#
export MPICH_GPU_SUPPORT_ENABLED=1
export MVP_ENABLE_GPU=1
export MVP_CH4_OFI_ENABLE_HMEM=1
export MVP_CH4_OFI_ENABLE_MR_HMEM=1
export MV2_SHOW_ENV_INFO=1

#
# Avoid carrying over conflicting visibility/binding vars
#
unset SRUN_CPUS_PER_TASK
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

#
# Optional: leave these unset for the first stable test
# If envScriptFrontier_mvapich.sh sets them, consider commenting them out there too.
#
unset FI_MR_CACHE_MONITOR || true
unset FI_CXI_DEFAULT_CQ_SIZE || true
unset FI_CXI_DEFAULT_TX_SIZE || true
unset FI_CXI_RX_MATCH_MODE || true

NUM_PROCS=8
PPN=8

MODELS=(
    "wide_resnet50_2"
)

DATASETS=(
    "cifar10"
)

NUM_CLASSES_LIST=(
    "10"
)

IMAGE_SIZES=(
    "32"
)

NUM_EPOCHS_LIST=(
    "20"
)

BATCH_SIZES=(
    "64"
)

LEARNING_RATES=(
    "0.1"
)

MOMENTUMS=(
    "0.9"
)

OPTIMIZERS=(
    "sgd"
)

WEIGHT_DECAYS=(
    "5e-4"
)

NESTEROV_VALUES=(
    "true"
)

WD_ON_BN_BIAS_VALUES=(
    "false"
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
    "false"
)

CIFAR_STEM_VALUES=(
    "true"
)

DROP_LAST_VALUES=(
    "true"
)

BACKENDS=(
    "mpi"
)

#
# Start simple: use built-in DDP first.
# Once that is stable, re-enable custom hooks.
#
EXPERIMENTS=(
    "none:"
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
for OPTIMIZER in "${OPTIMIZERS[@]}"; do
for MOMENTUM in "${MOMENTUMS[@]}"; do
for WEIGHT_DECAY in "${WEIGHT_DECAYS[@]}"; do
for NESTEROV in "${NESTEROV_VALUES[@]}"; do
for WD_ON_BN_BIAS in "${WD_ON_BN_BIAS_VALUES[@]}"; do
for EXPERIMENT in "${EXPERIMENTS[@]}"; do

    COMM_ALGORITHM="${EXPERIMENT%%:*}"
    ZFP_RATE="${EXPERIMENT#*:}"

    echo "=== Starting: model=${MODEL_NAME}, ws=${NUM_PROCS}, bs=${BATCH_SIZE}, gb=$((BATCH_SIZE*NUM_PROCS)), lr=${LEARNING_RATE}, sched=${SCHEDULER}, warmup=${WARMUP_EPOCHS}, mom=${MOMENTUM}, wd=${WEIGHT_DECAY}, nesterov=${NESTEROV}, wd_bn_bias=${WD_ON_BN_BIAS}, backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp=${ZFP_RATE:-none} at $(date) ==="

    export MODEL_NAME="${MODEL_NAME}"
    export DATASET="${DATASET}"
    export NUM_CLASSES="${NUM_CLASSES}"
    export IMAGE_SIZE="${IMAGE_SIZE}"
    export NUM_EPOCHS="${NUM_EPOCHS}"
    export BATCH_SIZE="${BATCH_SIZE}"
    export LEARNING_RATE="${LEARNING_RATE}"
    export MOMENTUM="${MOMENTUM}"
    export WEIGHT_DECAY="${WEIGHT_DECAY}"
    export NESTEROV="${NESTEROV}"
    export WD_ON_BN_BIAS="${WD_ON_BN_BIAS}"
    export GRAD_CLIP="${GRAD_CLIP}"
    export SCHEDULER="${SCHEDULER}"
    export WARMUP_EPOCHS="${WARMUP_EPOCHS}"
    export NUM_WORKERS="4"
    export PIN_MEMORY="true"
    export DROP_LAST="${DROP_LAST}"
    export PRETRAINED="${PRETRAINED}"
    export OPTIMIZER="${OPTIMIZER}"
    export CIFAR_STEM="${CIFAR_STEM}"
    export INIT_CIFAR_STEM_FROM_PRETRAINED_CENTER="false"
    export DATA_DIR="./data"
    export CHECKPOINT_DIR="./checkpoints"
    export SEED="42"
    export BACKEND="${BACKEND}"
    export COMM_ALGORITHM="${COMM_ALGORITHM}"

    unset MIOPEN_FIND_MODE
    unset MIOPEN_FIND_ENFORCE

    export TORCH_DISTRIBUTED_DEBUG=DETAIL
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    if [[ -n "${ZFP_RATE}" ]]; then
        export ZFP_RATE="${ZFP_RATE}"
    else
        unset ZFP_RATE
    fi

    echo "==== Runtime env debug ===="
    env | egrep 'MPICH|MVP|MV2|FI_|ROCR_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|SLURM_JOB_GPUS|SLURM_STEP_GPUS' || true

    echo "==== GPU visibility debug (batch shell) ===="
    hostname
    command -v rocminfo >/dev/null 2>&1 && rocminfo | head -n 30 || true

    #
    # Match the working Cray launch shape as closely as possible
    #
    srun --exact \
        -n "${NUM_PROCS}" \
        --ntasks-per-node="${PPN}" \
        --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
        --gpus-per-task=1 \
        python -u interface.py

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
done
done
done
done
done

echo "=== Job finished: $(date) ==="