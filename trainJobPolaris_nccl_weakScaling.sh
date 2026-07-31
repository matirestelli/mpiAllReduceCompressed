#!/bin/bash -l
#PBS -l select=10:system=polaris
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:eagle
#PBS -q prod
#PBS -A UIC-HPC
#PBS -o ddp_train_polaris_nccl_16gpus_8b_32b_128b.%j.out
#PBS -e ddp_train_polaris_nccl_16gpus_8b_32b_128b.%j.err
#PBS -N ddp-train-polaris_nccl_16gpus_8b_32b_128b

# ── WEAK SCALING GRID (Polaris / PBS / A100) ─────────────────────────────────
# B_local fixed, B_global = B_local x P grows. LR = 0.1 * B_global / 128.
# Polaris node = 4x A100, so PPN=4 and NNODES = NUM_PROCS / 4.
#
#  P=8   -> select=2   NUM_EPOCHS=5    bs 8:lr 0.05 | bs 32:lr 0.20 | bs 128:lr 0.80
#  P=16  -> select=4   NUM_EPOCHS=8    bs 8:lr 0.10 | bs 32:lr 0.40 | bs 128:lr 1.60
#  P=32  -> select=8   NUM_EPOCHS=12   bs 8:lr 0.20 | bs 32:lr 0.80 | bs 128:lr 3.20
#
# Set NUM_PROCS below, uncomment the matching WEAK_CONFIGS + NUM_EPOCHS block,
# and change #PBS -l select=N to match (N = NUM_PROCS / 4).
#
# Metric = median of t_iter_median_ms over epochs >= 2. NEVER epoch time.
# Throughput only: do not report accuracy from bs=8, or from bs=128 at P>=32,
# and NOTE: NUM_EPOCHS varies per P (timing knob) -> accuracy NOT comparable across P.
# ─────────────────────────────────────────────────────────────────────────────

cd "${PBS_O_WORKDIR}"

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="

source ${PBS_O_WORKDIR}/envScript_nccl.sh

# For enabling profiling time for communication hooks.
export DDP_HOOK_TIMING=1
export DDP_HOOK_TIMING_RANK0_ONLY=1
export PRETRAINED_WEIGHTS_CACHE=/eagle/UIC-HPC/matilderestelli/pretrained_weights_cache
export DDP_ITER_LOG=0        # per-rank JSONL off (space); set 1 only for bs=8 at P=32
export DDP_PROFILE_BARRIER=0 # MUST stay 0 for timing runs — it kills bwd/comm overlap

# Polaris rendezvous (mpiexec does not set these for us)
export MASTER_ADDR="$(head -n 1 "${PBS_NODEFILE}")"
export MASTER_PORT=29500
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

# ── Polaris topology ─────────────────────────────────────────────────────────
NUM_PROCS=8
PPN=4                       # Polaris node = 4x A100
NNODES=$(( (NUM_PROCS + PPN - 1) / PPN ))
# Polaris: 32 CPU cores/node, 4 ranks/node -> 8 cores/rank (--depth=8).
DEPTH=8
NUM_WORKERS_VAL=$(( DEPTH - 1 ))   # leave 1 core for the main process

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

# BATCH_SIZE and LEARNING_RATE must be PAIRED here, not cross-producted.
WEAK_CONFIGS=(
    # P=8   (select=2)
    "8:0.05"     # gb 64
    "32:0.20"    # gb 256
    "128:0.80"   # gb 1024

    # P=16  (select=4)
    # "8:0.10"    # gb 128
    #"32:0.40"   # gb 512
    #"128:1.60"  # gb 2048

    # P=32  (select=8)
    #"8:0.20"    # gb 256
    # "32:0.80"   # gb 1024
    #"128:3.20"  # gb 4096
)

NUM_EPOCHS_LIST=(
    "5"        # P=8
    #"8"      # P=16
    #"12"     # P=32
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

# Keep 0: warmup_steps = steps_per_epoch * WARMUP_EPOCHS, and steps_per_epoch
# shrinks as P grows, so the same value means a different warmup at every point.
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
# Full list only at bs=32 (main operating point). bs=8 and bs=128 bracket the regime.
EXPERIMENTS=(
    "none:"
    "ring:"
    "ring_zfp_naive:16"
    "ring_zfp_online_coll:16"
    "ring_zfp_online_coll:10"
    "ring_zfp_online_coll:8"
    "recursive_doubling:"
    "recursive_doubling_zfp_naive:16"
    "recursive_doubling_zfp_online_coll:16"
    "recursive_doubling_zfp_online_coll:8"
    #"recursive_doubling_zfp_online_coll:4"
    #"ring_zfp_online_coll:8"
    #"ring_zfp_online_coll:4"
    # "default:"
    #"default_sync:"
    #"default_clone:"
    #"default_cpu_stage:"
)

# ── Sanity: every rank must see the right python / torchvision._C ─────────────
mpiexec -np "${NUM_PROCS}" --ppn "${PPN}" --depth="${DEPTH}" --cpu-bind depth \
    python - <<'PY'
import importlib.util, sys
spec = importlib.util.find_spec("torchvision._C")
print("python", sys.version)
print("torchvision._C spec:", spec)
if spec is not None:
    print("origin:", spec.origin)
PY

for MODEL_NAME in "${MODELS[@]}"; do
for DATASET in "${DATASETS[@]}"; do
for NUM_CLASSES in "${NUM_CLASSES_LIST[@]}"; do
for IMAGE_SIZE in "${IMAGE_SIZES[@]}"; do
for NUM_EPOCHS in "${NUM_EPOCHS_LIST[@]}"; do
for WEAK_CONFIG in "${WEAK_CONFIGS[@]}"; do
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

    BATCH_SIZE="${WEAK_CONFIG%%:*}"
    LEARNING_RATE="${WEAK_CONFIG#*:}"

    for EXPERIMENT in "${EXPERIMENTS[@]}"; do
        COMM_ALGORITHM="${EXPERIMENT%%:*}"
        ZFP_RATE="${EXPERIMENT#*:}"

        echo "=== Starting: model=${MODEL_NAME}, ws=${NUM_PROCS}, bs=${BATCH_SIZE}, gb=$((BATCH_SIZE*NUM_PROCS)), lr=${LEARNING_RATE}, sched=${SCHEDULER}, warmup=${WARMUP_EPOCHS}, epochs=${NUM_EPOCHS}, mom=${MOMENTUM}, wd=${WEIGHT_DECAY}, nesterov=${NESTEROV}, wd_bn_bias=${WD_ON_BN_BIAS}, backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp=${ZFP_RATE:-none} at $(date) ==="

        # Polaris passes env to ranks via mpiexec -env (NOT plain export).
        MPI_ENV_ARGS=(
            -env MASTER_ADDR="${MASTER_ADDR}"
            -env MASTER_PORT="${MASTER_PORT}"
            -env MPICH_GPU_SUPPORT_ENABLED=1
            -env LD_PRELOAD="${LD_PRELOAD}"
            -env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
            -env PYTHONPATH="${PYTHONPATH}"
            -env DDP_HOOK_TIMING="${DDP_HOOK_TIMING}"
            -env DDP_HOOK_TIMING_RANK0_ONLY="${DDP_HOOK_TIMING_RANK0_ONLY}"
            -env DDP_ITER_LOG="${DDP_ITER_LOG}"
            -env DDP_PROFILE_BARRIER="${DDP_PROFILE_BARRIER}"
            -env PRETRAINED_WEIGHTS_CACHE="${PRETRAINED_WEIGHTS_CACHE}"
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
            -env NUM_WORKERS="${NUM_WORKERS_VAL}"
            -env OMP_NUM_THREADS=1
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
            -env TORCH_DISTRIBUTED_DEBUG=DETAIL
            -env TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        )

        if [[ -n "${ZFP_RATE}" ]]; then
            MPI_ENV_ARGS+=(-env ZFP_RATE="${ZFP_RATE}")
        fi

        mpiexec -np "${NUM_PROCS}" --ppn "${PPN}" --depth="${DEPTH}" --cpu-bind depth \
            "${MPI_ENV_ARGS[@]}" \
            python -u interface.py

        status=$?
        echo "=== Finished: model=${MODEL_NAME}, dataset=${DATASET}, batch=${BATCH_SIZE}, lr=${LEARNING_RATE}, scheduler=${SCHEDULER}, backend=${BACKEND}, hook=${COMM_ALGORITHM}, zfp_rate=${ZFP_RATE:-none}, status=${status} at $(date) ==="

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
done       # WEAK_CONFIGS
done       # NUM_EPOCHS
done       # IMAGE_SIZE
done       # NUM_CLASSES
done       # DATASET
done       # MODELS

echo "=== Job finished: $(date) ==="

# submit:        qsub run_weak_polaris.sh
# interactive:   qsub -I -l select=4:system=polaris -l walltime=01:00:00 -l filesystems=home:eagle -q debug-scaling -A UIC-HPC