#!/bin/bash -l
#PBS -l select=1:ngpus=4
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -A UIC-HPC
#PBS -N verify-cuda-aware-mpi
#PBS -o verify_cuda_aware_mpi.out
#PBS -e verify_cuda_aware_mpi.err
#PBS -j n

set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"

echo "=== JOB ==="
echo "date: $(date -u)"
echo "host: $(hostname)"
echo "pwd : $(pwd)"
echo "job : ${PBS_JOBID:-<no PBS_JOBID>}"
echo

# ----------------------------
# PRRTE temp on node-local disk
# ----------------------------
export TMPDIR="/tmp/${USER}/${PBS_JOBID:-nojobid}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
export OMPI_MCA_prte_tmpdir_base="$TMPDIR"
export OMPI_MCA_prte_silence_shared_fs=1

# ----------------------------
# Test parameters (exported)
# ----------------------------
export NRANKS="${NRANKS:-4}"
export TENSOR_MIB="${TENSOR_MIB:-256}"
export WARMUP_ITERS="${WARMUP_ITERS:-5}"
export ITERS="${ITERS:-20}"

# CUDA-aware MPI/UCX settings (KEEP CUDA-AWARE ON)
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_opal_cuda_support=true

# Choose transports explicitly (you can override on qsub -v UCX_TLS=...)
export UCX_TLS="${UCX_TLS:-self,cuda_ipc,cuda_copy,posix}"
export UCX_MEMTYPE_CACHE="${UCX_MEMTYPE_CACHE:-n}"

# Make UCX produce per-rank logs so a hang shows WHERE
export UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-info}"
export UCX_LOG_FILE="${UCX_LOG_FILE:-${PBS_O_WORKDIR}/ucx.%h.%p.log}"

# Optional: ensure UCX 1.17 is first
UCX_PREFIX="/soft/libraries/ucx/1.17.0"
export PATH="${UCX_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${UCX_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Limit CPU noise
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# ----------------------------
# Source your environment
# ----------------------------
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

if [[ -f "${PBS_O_WORKDIR}/envScriptSophia.sh" ]]; then
  source "${PBS_O_WORKDIR}/envScriptSophia.sh"
elif [[ -f "${SCRIPT_DIR}/envScriptSophia.sh" ]]; then
  source "${SCRIPT_DIR}/envScriptSophia.sh"
fi

echo "=== EFFECTIVE SETTINGS ==="
echo "NRANKS=${NRANKS}"
echo "TENSOR_MIB=${TENSOR_MIB}"
echo "WARMUP_ITERS=${WARMUP_ITERS}"
echo "ITERS=${ITERS}"
echo "OMPI_MCA_pml=${OMPI_MCA_pml}"
echo "OMPI_MCA_osc=${OMPI_MCA_osc}"
echo "OMPI_MCA_opal_cuda_support=${OMPI_MCA_opal_cuda_support}"
echo "UCX_TLS=${UCX_TLS}"
echo "UCX_MEMTYPE_CACHE=${UCX_MEMTYPE_CACHE}"
echo "UCX_LOG_LEVEL=${UCX_LOG_LEVEL}"
echo "UCX_LOG_FILE=${UCX_LOG_FILE}"
echo "TMPDIR=${TMPDIR}"
echo

echo "=== VERSIONS ==="
command -v mpirun
mpirun --version | head -n 5 || true
command -v ucx_info || true
echo

echo "=== MPI SMOKE TEST ==="
mpirun --tag-output --timestamp-output -np "${NRANKS}" \
  -x PATH -x LD_LIBRARY_PATH \
  -x OMPI_MCA_prte_tmpdir_base -x OMPI_MCA_prte_silence_shared_fs \
  hostname
echo "MPI smoke test OK"
echo

# ----------------------------
# Run the actual test with:
# - per-rank tagged output
# - hard timeout (so it never hangs forever)
# - faulthandler for Python stack dumps on timeout/SIGUSR1
# ----------------------------
echo "=== CUDA-AWARE MPI TEST ==="
echo "If it hangs, see: ${UCX_LOG_FILE} and the python stack dumps in stdout/err."
echo

# 6 minutes timeout (adjust as needed). If timeout triggers, you get failure + logs.
timeout 360s mpirun --tag-output --timestamp-output -np "${NRANKS}" \
  -x PATH -x LD_LIBRARY_PATH \
  -x OMPI_MCA_prte_tmpdir_base -x OMPI_MCA_prte_silence_shared_fs \
  -x OMPI_MCA_pml -x OMPI_MCA_osc -x OMPI_MCA_opal_cuda_support \
  -x UCX_TLS -x UCX_MEMTYPE_CACHE -x UCX_LOG_LEVEL -x UCX_LOG_FILE \
  -x NRANKS -x TENSOR_MIB -x WARMUP_ITERS -x ITERS \
  -x OMP_NUM_THREADS \
  python -u - <<'PY'
import os, time, statistics, faulthandler, signal, sys
import torch
import torch.distributed as dist

faulthandler.enable(all_threads=True)
faulthandler.register(signal.SIGUSR1, all_threads=True)

rank_env = os.environ.get("OMPI_COMM_WORLD_RANK", "?")
print(f"[python] started pid={os.getpid()} host={os.uname().nodename} OMPI_COMM_WORLD_RANK={rank_env}", flush=True)

# Prove CUDA context works early
assert torch.cuda.is_available(), "CUDA not available"
local_gpu = int(rank_env) % torch.cuda.device_count() if rank_env != "?" else 0
torch.cuda.set_device(local_gpu)
torch.zeros(1, device="cuda")  # touch cuda
print(f"[python] rank_env={rank_env} using cuda:{local_gpu}", flush=True)

print("[python] before dist.init_process_group('mpi')", flush=True)
t0 = time.time()
dist.init_process_group(backend="mpi")
dt = time.time() - t0
rank = dist.get_rank()
world = dist.get_world_size()
print(f"[python] rank {rank}/{world} after init_process_group in {dt:.3f}s", flush=True)

tensor_mib = int(os.environ.get("TENSOR_MIB", "256"))
warmup = int(os.environ.get("WARMUP_ITERS", "5"))
iters = int(os.environ.get("ITERS", "20"))

nbytes = tensor_mib * 1024 * 1024
nelem = nbytes // 4  # float32
x = torch.ones(nelem, device="cuda", dtype=torch.float32)

# First barrier to ensure everyone reached here
dist.barrier()
if rank == 0:
    print("[python] starting warmup allreduce", flush=True)

# Warmup: time the first allreduce separately (often where it hangs)
torch.cuda.synchronize()
t_first0 = time.perf_counter()
dist.all_reduce(x)
torch.cuda.synchronize()
t_first = time.perf_counter() - t_first0
dist.barrier()
print(f"[python] rank {rank}: FIRST allreduce {t_first:.6f}s", flush=True)

for _ in range(max(0, warmup - 1)):
    dist.all_reduce(x)
torch.cuda.synchronize()
dist.barrier()

times = []
for _ in range(iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    times.append(time.perf_counter() - t0)

med = statistics.median(times)
mn, mx = min(times), max(times)

p = world
moved = 2.0 * (p - 1) / p * nbytes
bw_gib = (moved / med) / (1024**3)

print(f"[python] rank {rank}: median {med:.6f}s min {mn:.6f}s max {mx:.6f}s est_bw {bw_gib:.2f} GiB/s", flush=True)

dist.barrier()
dist.destroy_process_group()
PY

rc=$?
echo
echo "=== DONE (rc=$rc) $(date -u) ==="

# If timeout killed it, make it obvious
if [[ $rc -eq 124 ]]; then
  echo "[ERROR] Timed out. Collecting quick UCX log summary:"
  ls -lh ucx.*.log 2>/dev/null || true
  echo "Last 50 lines of each UCX log:"
  for f in ucx.*.log; do
    echo "----- $f -----"
    tail -n 50 "$f" || true
  done
  exit 124
fi
