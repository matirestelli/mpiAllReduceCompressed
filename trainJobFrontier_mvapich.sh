#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -p extended
#SBATCH -J ddp-train-frontier_8gpus_64b_mvapich
#SBATCH -N 1  
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=1                  
#SBATCH -t 08:00:00   
#SBATCH -C nvme            
#SBATCH -o ddp_train_frontier_8gpus_64b_mvapich.%j.out       
#SBATCH -e ddp_train_frontier_8gpus_64b_mvapich.%j.err       


#ask for a compute node 
# salloc -A gen243 -p batch -J installingPyTorch -N 1 --gres=gpu:4 -t 01:00:00 -o installingPyTorch.%j.out -e installingPyTorch.%j.out

# salloc -A gen243 -p batch -J trytrain -N 1 -t 00:30:00
# NB to use nvme you need to request that with the allocation, if sbatch sh job launche out there if interactive terminal :
# salloc -A gen243 -p batch -J trytrain -N 1 -t 00:30:00 -C nvme

cd "${SLURM_SUBMIT_DIR}"

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source envScriptFrontier_mvapich.sh

unset MPICH_GPU_SUPPORT_ENABLED
export MV2_USE_ROCM=1
export MV2_SHOW_ENV_INFO=1
export MV2_USE_CUDA=1


# For enabling profiling time for communication hooks.
export DDP_HOOK_TIMING=1
export DDP_HOOK_TIMING_RANK0_ONLY=1
export PRETRAINED_WEIGHTS_CACHE=/lustre/orion/gen243/proj-shared/matilderestelli/pretrained_weights_cache
export DDP_ITER_LOG=0        # per-rank JSONL off (space); set 1 for a few configs
export DDP_PROFILE_BARRIER=0 # MUST stay 0 for timing runs — it kills bwd/comm overlap


# Edit these lists to run multiple experiments inside one queued job.
# Leave only one active value in each list for a single run.

NUM_PROCS=8
PPN=8
# processes per node

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
    #"224"   # ImageNet / ImageNet-like
)

NUM_EPOCHS_LIST=(
    "20"
    # "50"
    # "100"
)

BATCH_SIZES=(
    # "8"       # weak scaling local batch 8, total 128
    #"16"       # weak scaling local batch 16 (only one that works on Polaris supercompter)
    #"32"       # strong global scaling -> keep 32 fixed as local batch size, total 512 on 16 GPUs
    "64"    # strong global 256 on 4 GPUs
    # "128"   # weak scaling local batch 128
)

LEARNING_RATES=(
    #"0.0001"
    #"0.001"    # paper-style CIFAR reproduction
    #"0.003" 
    #"0.01"
    #"0.05"
    "0.1"
    #"0.2"
)

MOMENTUMS=(
    "0.9"      # standard
    # "0.95"   # B5 — only relevant at global batch >= 2048
)

OPTIMIZERS=(
    "sgd"        # A1–A5
    #"adamw"    # A6–A7 (paper's lr=0.001 is probably an Adam LR)
)

WEIGHT_DECAYS=(
    "5e-4"     # standard CIFAR WRN
    #"0.05"  #adam 
    # "1e-4"   # B3 and imagenet
)

NESTEROV_VALUES=(
    "true"     # B1 baseline
    # "false"  # B2
)

# false = BN gamma/beta and ALL biases get weight_decay=0.0  (B1, recommended)
# true  = weight decay applied to every parameter            (B4, your old behaviour)
WD_ON_BN_BIAS_VALUES=(
    "false"    # B1
    # "true"   # B4
)

SCHEDULERS=(
    # "constant" # paper-style LR=0.001 reproduction
    "cosine"
)

WARMUP_EPOCHS_LIST=(
    "0"
   # "1"      # paper-style CIFAR reproduction
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
    #"false"  # ImageNet / ImageNet-like
)

DROP_LAST_VALUES=(
    #"false"
    "true"
)

BACKENDS=(
    "mpi"
    # "nccl"
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
    #"default:"
    #"default_sync:"
    #"default_clone:"
    #"default_cpu_stage:"
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

        srun --exact -n "${NUM_PROCS}" --ntasks-per-node="${PPN}" \
            --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
            --gpus-per-task=1 \
            python -u interface.py

        status=$?
        echo "=== Finished training: model=${MODEL_NAME}, dataset=${DATASET}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}, scheduler=${SCHEDULER}, backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp_rate=${ZFP_RATE:-none}, status=${status} at $(date) ==="

        if [[ ${status} -ne 0 ]]; then
            echo "=== Stopping job because training failed ==="
            exit ${status}
        fi
    done   # EXPERIMENTS
done       # WD_ON_BN_BIAS
done       # NESTEROV
done       # WEIGHT_DECAY
done       # MOMENTUM
done       # OPTIMIZER
done       # BACKEND
done       # DROP_LAST
done       # CIFAR_STEM
done       # PRETRAINED
done       # GRAD_CLIP
done       # WARMUP_EPOCHS
done       # SCHEDULER
done       # LEARNING_RATE
done       # BATCH_SIZE
done       # NUM_EPOCHS
done       # IMAGE_SIZE
done       # NUM_CLASSES
done       # DATASET
done       # MODELS

echo "=== Job finished: $(date) ==="

# salloc -A gen243 -p batch -J torch-build-and-pack -N 1 -t 00:30:00 -c 16 -C nvme
# squeue -j <jobid> -o "%.18i %.9P %.10T %.8u %.10M %.20S %.6D %R"
