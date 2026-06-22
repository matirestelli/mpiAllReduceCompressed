#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N check-mpi-gpuaware
#PBS -q by-gpu
#PBS -A UIC-HPC
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o check-mpi-gpuaware.pbs.out

set -euo pipefail

CONDA_PREFIX="/lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build_sophia/conda/pt-sophia-openmpi"

module purge
module use /soft/modulefiles
module load compilers/openmpi/5.0.10

export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

echo "Node: $(hostname)"
echo "PBS_JOBID: ${PBS_JOBID:-}"
which mpirun
mpirun --version | head -n 2 || true

python - <<'PY'
import torch, torch.distributed as dist
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("cuda available/devices:", torch.cuda.is_available(), torch.cuda.device_count())
print("dist.is_mpi_available:", dist.is_mpi_available())
PY

SCRIPT="${PBS_O_WORKDIR:-$PWD}/mpi_gpuaware_check.py"

cat > "${SCRIPT}" <<'PY'
import torch
import torch.distributed as dist

dist.init_process_group(backend="mpi")
rank = dist.get_rank()
world = dist.get_world_size()

def allreduce_check(device):
    x = torch.ones(1024*1024, device=device, dtype=torch.float32) * (rank + 1)
    dist.all_reduce(x)  # sum
    expected = (world * (world + 1)) / 2.0
    ok = torch.allclose(x.mean(), torch.tensor(expected, device=device), rtol=0, atol=1e-3)
    if rank == 0:
        print(f"[{device}] all_reduce mean={x.mean().item():.3f} expected={expected:.3f} ok={ok}")
    return ok

ok_cpu = allreduce_check("cpu")

ok_cuda = None
if torch.cuda.is_available():
    torch.cuda.set_device(rank % torch.cuda.device_count())
    try:
        ok_cuda = allreduce_check("cuda")
    except Exception as e:
        if rank == 0:
            print("[cuda] all_reduce exception (likely not GPU-aware MPI):", repr(e))
        ok_cuda = False
else:
    if rank == 0:
        print("CUDA not available; skipping GPU-aware check")

if rank == 0:
    print("RESULT cpu_ok =", ok_cpu)
    print("RESULT cuda_ok =", ok_cuda)
PY

echo "== Running 2-rank MPI check =="
mpirun -np 2 python "${SCRIPT}"

