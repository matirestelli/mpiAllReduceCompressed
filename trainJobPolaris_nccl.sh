#!/bin/bash -l
#PBS -l select=8:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A UIC-HPC
#PBS -o ddp_train_PolarisNCCL_32gpus_4b_16b.%j.out
#PBS -e ddp_train_PolarisNCCL_32gpus_4b_16b.%j.err
#PBS -N ddp-train_PolarisNCCL_32gpus_4b_16b

cd ${PBS_O_WORKDIR}

# PBS does NOT substitute %j in #PBS -o. Use runtime jobid instead:
LOG="ddp_train.${PBS_JOBID}.out"
exec > >(tee -a "${LOG}") 2>&1

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

#source ${PBS_O_WORKDIR}/envScript3.sh
source ${PBS_O_WORKDIR}/envScript_nccl.sh

# For enabling profiling time for communication hooks.
export DDP_HOOK_TIMING=1
export DDP_HOOK_TIMING_RANK0_ONLY=1
export DDP_ITER_LOG=0        # per-rank JSONL off (space); set 1 for a few configs
export DDP_PROFILE_BARRIER=0 # MUST stay 0 for timing runs — it kills bwd/comm overlap

export MASTER_ADDR="$(head -n 1 "${PBS_NODEFILE}")"
export MASTER_PORT=29500

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"


# Edit these lists to run multiple experiments inside one queued job.
# Leave only one active value in each list for a single run.

NUM_PROCS=32
PPN=4
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
    "10"
    #"20"
    # "50"
    # "100"
)

BATCH_SIZES=(
    #"4"
    #"8"       # weak scaling local batch 8, total 128
    "16"       # weak scaling local batch 16 (only one that works on Polaris supercompter)
    #"32"       # strong global scaling -> keep 32 fixed as local batch size, total 512 on 16 GPUs
    #"64"    # strong global 256 on 4 GPUs
    # "128"   # weak scaling local batch 128
)

LEARNING_RATES=(
    #"0.0001"
    # "0.001"    # paper-style CIFAR reproduction
    # "0.003" 
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
    # "adamw"    # A6–A7 (paper's lr=0.001 is probably an Adam LR)
)

WEIGHT_DECAYS=(
    "5e-4"     # standard CIFAR sgd
    # "0.05"  #adam 
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
    # "mpi"
    "nccl"
)

# "default" = built-in DDP allreduce wrapped with NVTX timing.
# "none"    = no custom communication hook.
EXPERIMENTS=(
    "none:"
    "ring:"
    "ring_zfp_naive:16"
    "ring_zfp_online_coll:16"
    "ring_zfp_online_coll:8"
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

        MPI_ENV_ARGS=(
            #for multi node runs with nccl needs to pass master address and port to all nodes
            -env MASTER_ADDR="${MASTER_ADDR}"
            -env MASTER_PORT="${MASTER_PORT}"
            -env MODEL_NAME="${MODEL_NAME}"
            -env DATASET="${DATASET}"
            -env NUM_CLASSES="${NUM_CLASSES}"
            -env IMAGE_SIZE="${IMAGE_SIZE}"
            -env NUM_EPOCHS="${NUM_EPOCHS}"
            -env BATCH_SIZE="${BATCH_SIZE}"
            -env LEARNING_RATE="${LEARNING_RATE}"
            -env MOMENTUM="${MOMENTUM}"
            -env WEIGHT_DECAY="${WEIGHT_DECAY}"
            -env NESTEROV="${NESTEROV}"
            -env WD_ON_BN_BIAS="${WD_ON_BN_BIAS}"
            -env GRAD_CLIP="${GRAD_CLIP}"
            -env SCHEDULER="${SCHEDULER}"
            -env WARMUP_EPOCHS="${WARMUP_EPOCHS}"
            -env NUM_WORKERS="4"
            -env PIN_MEMORY="true"
            -env DROP_LAST="${DROP_LAST}"
            -env PRETRAINED="${PRETRAINED}"
            -env OPTIMIZER="${OPTIMIZER}"
            -env CIFAR_STEM="${CIFAR_STEM}"
            -env INIT_CIFAR_STEM_FROM_PRETRAINED_CENTER="false"
            -env DATA_DIR="./data"
            -env CHECKPOINT_DIR="./checkpoints"
            -env SEED="42"
            -env BACKEND="${BACKEND}"
            -env COMM_ALGORITHM="${COMM_ALGORITHM}"
        )

        if [[ -n "${ZFP_RATE}" ]]; then
            MPI_ENV_ARGS+=(-env ZFP_RATE="${ZFP_RATE}")
        fi

        mpiexec -np ${NUM_PROCS} --ppn ${PPN} --depth=8 --cpu-bind depth \
            "${MPI_ENV_ARGS[@]}" \
            python interface.py

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


