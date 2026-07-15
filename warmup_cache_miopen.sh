#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -p extended
#SBATCH -J miopen_warmup_cifar          
#SBATCH -N 1  
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8                  
#SBATCH -t 12:00:00   
#SBATCH -C nvme  
#SBATCH -o miopen_warmup_cifar.%j.out
#SBATCH -e miopen_warmup_cifar.%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

echo "=== Warmup job started: $(date) ==="
echo "=== Node: $(hostname) ==="

# --- Node identification, printed to both stdout and stderr ---
echo "=== Node: $(hostname) ===" | tee /dev/stderr
echo "=== Job: ${SLURM_JOB_ID} ===" | tee /dev/stderr

# --- Background GPU health poller: snapshot every 60s for the whole job ---
GPU_MONITOR_LOG="${SLURM_SUBMIT_DIR}/gpu_health_${SLURM_JOB_ID}.log"
(
  while true; do
    echo "--- $(date -Is) ---"
    rocm-smi --showclocks --showtemp --showpower --showuse 2>&1
    echo
    sleep 60
  done
) >> "${GPU_MONITOR_LOG}" 2>&1 &
GPU_MONITOR_PID=$!

# For warmup, ideally source your *warmup* env script (Option A),
# but if you still source envScriptFrontier.sh, make sure it does not force FAST/ENFORCE.
source envScriptFrontier_miopenWarmup.sh

# 1 process, 1 GPU
NUM_PROCS=1
PPN=1

export PYTHONUNBUFFERED=1

# Option A: build/populate cache (tune+compile on misses)
export MIOPEN_FIND_MODE=1
export MIOPEN_FIND_ENFORCE=1
# if you use 3 exhastive search and takes a lot

export MAX_TRAIN_BATCHES=50
export MAX_VAL_BATCHES=5

# Logging (can be large)
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_LOG_LEVEL=4
export MIOPEN_LOG_FILE="${SLURM_SUBMIT_DIR}/miopen_${SLURM_JOB_ID}.log"

export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128

export WARMUP_PRINT_EVERY=1
export WARMUP_PRINT_FIRST_N=20
export WARMUP_TIMING=1

# Sanity checks
: "${MIOPEN_LOCAL:?MIOPEN_LOCAL not set by env script}"
: "${MIOPEN_BASE:?MIOPEN_BASE not set by env script}"
: "${MIOPEN_USER_DB_PATH:?MIOPEN_USER_DB_PATH not set}"
: "${MIOPEN_CUSTOM_CACHE_DIR:?MIOPEN_CUSTOM_CACHE_DIR not set}"

echo "Warmup cache local: ${MIOPEN_LOCAL}"
echo "Warmup baseline:    ${MIOPEN_BASE}"
echo "Warmup DB dir:      ${MIOPEN_USER_DB_PATH}"
echo "Warmup cache dir:   ${MIOPEN_CUSTOM_CACHE_DIR}"
echo "MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE} MIOPEN_FIND_ENFORCE=${MIOPEN_FIND_ENFORCE}"

MODELS=("wide_resnet50_2")
DATASETS=("cifar10")
NUM_CLASSES_LIST=("10")
IMAGE_SIZES=("32")
NUM_EPOCHS_LIST=("1")
BATCH_SIZES=("8" "16" "32" "64")   # put all per-GPU batch sizes you will use, otherwise warmup will miss them and recompile/tune during real runs
LEARNING_RATES=("0.001")
SCHEDULERS=("cosine")
WARMUP_EPOCHS_LIST=("0")
GRAD_CLIPS=("none")
PRETRAINED_VALUES=("false")
CIFAR_STEM_VALUES=("true")
DROP_LAST_VALUES=("true")
BACKENDS=("mpi")
EXPERIMENTS=("none:")

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

  export MODEL_NAME DATASET NUM_CLASSES IMAGE_SIZE NUM_EPOCHS BATCH_SIZE
  export LEARNING_RATE SCHEDULER WARMUP_EPOCHS GRAD_CLIP PRETRAINED
  export CIFAR_STEM DROP_LAST BACKEND COMM_ALGORITHM
  export MOMENTUM="0.9"
  export WEIGHT_DECAY="5e-4"
  export NUM_WORKERS="4"
  export PIN_MEMORY="true"
  export INIT_CIFAR_STEM_FROM_PRETRAINED_CENTER="false"
  export DATA_DIR="./data"
  export CHECKPOINT_DIR="./checkpoints"
  export SEED="42"

  if [[ -n "${ZFP_RATE}" ]]; then export ZFP_RATE="${ZFP_RATE}"; else unset ZFP_RATE; fi

  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

  echo "=== Running interface.py at $(date) ==="
  srun --export=ALL --exact -n "${NUM_PROCS}" --ntasks-per-node="${PPN}" --cpus-per-task=8 \
    python -u interface.py

done; done; done; done; done; done; done; done; done; done; done; done; done; done; done

# Save back to Lustre baseline (single node).
echo "=== Saving cached DB/kernel cache back to Lustre baseline ==="
mkdir -p "${MIOPEN_BASE}"

# Optional: lock to avoid two warmup jobs copying at the same time
LOCKFILE="${MIOPEN_BASE}/.update.lock"
(
  flock -x 200
  cp -a "${MIOPEN_LOCAL}/." "${MIOPEN_BASE}/"
) 200>"${LOCKFILE}"

echo "=== Warmup job finished: $(date) ==="
kill "${GPU_MONITOR_PID}" 2>/dev/null || true
