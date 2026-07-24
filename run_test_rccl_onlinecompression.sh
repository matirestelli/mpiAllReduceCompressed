#!/bin/bash -l
#SBATCH -A gen243
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -J test_rccl_onlinecompression
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --gpu-bind=closest
#SBATCH -t 02:00:00
#SBATCH -C nvme
#SBATCH -o test_rccl_onlinecompression.%j.out
#SBATCH -e test_rccl_onlinecompression.%j.err
 
set -euo pipefail

NUM_PROCS=16
PPN=8
export PPN

# Use the same environment as the training job.
cd "${SLURM_SUBMIT_DIR}"
source envScriptFrontier.sh

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29577
export PYTHONUNBUFFERED=1

run() {
  echo ""
  echo "############################################################"
  echo "# $*"
  echo "############################################################"
  srun --exact -n "${NUM_PROCS}" --ntasks-per-node="${PPN}" \
       --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
       --gpus-per-task=1 \
       --gpu-bind=map_gpu:0,1,2,3,4,5,6,7 \
       python -u test_hier_algo.py "$@" || echo ">>> STAGE SET FAILED (continuing)"
}

# ── 1. Algorithm only. Identity codec, no streams. MUST be exact (0.0). ──────
#    A failure here is a pure algorithm/topology bug.
run --backend nccl --codec identity --numel 1000 --chunks 1
run --backend nccl --codec identity --numel 1000 --chunks 4

# ── 2. Stream ordering. Identity codec + hook-identical side streams. ────────
#    Passes above but fails here  =>  stream/event ordering bug.
run --backend nccl --codec identity --numel 1000 --chunks 4 --streams

# ── 3. ZFP codec, no streams. Isolates the compressor. ──────────────────────
#    Watch S2b: if the round-trip alone is broken/non-finite, the rate is the
#    problem, not the algorithm.
run --backend nccl --codec zfp --rate 16 --numel 1000 --chunks 1 --tol 1e-2
run --backend nccl --codec zfp --rate 8  --numel 1000 --chunks 1 --tol 1e-1
run --backend nccl --codec zfp --rate 4  --numel 1000 --chunks 1 --tol 5e-1

# ── 4. Everything together: ZFP + streams, realistic bucket size. ────────────
run --backend nccl --codec zfp --rate 16 --numel 1000000 --chunks 4 --streams --tol 1e-2

echo ""
echo "ALL STAGE SETS COMPLETE"