#!/bin/bash
#SBATCH -A gen243
#SBATCH -p batch 
#SBATCH -q debug 
#SBATCH -J torch_verify_mpi_gpuaware
#SBATCH -o %x-%j.out
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -C nvme
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=7

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

TARBALL=/lustre/orion/gen243/proj-shared/matilderestelli/pytorch/conda_env.tar.gz
BB=/mnt/bb/${USER}
ENV_TGZ=${BB}/torch_env.tar.gz
ENV_DIR=${BB}/torch_env
PY=${ENV_DIR}/bin/python

module purge
module load cpe/26.03
module load PrgEnv-gnu
module load gcc-native/13.2
module load cray-mpich
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan
module load miniforge3/23.11.0-0
module unload darshan-runtime || true

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_OFI_NIC_POLICY=GPU
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "${BB}"

echo "=== sbcast env tarball to NVMe ==="
sbcast -pf "${TARBALL}" "${ENV_TGZ}"

echo "=== untar env on NVMe (1 task per node) ==="
srun -N "${SLURM_NNODES}" --ntasks-per-node 1 mkdir -p "${ENV_DIR}"
srun -N "${SLURM_NNODES}" --ntasks-per-node 1 -c "${SLURM_CPUS_PER_TASK}" \
  tar --use-compress-program=pigz -xf "${ENV_TGZ}" -C "${ENV_DIR}"

echo "=== conda-unpack on each node (1 task per node) ==="
srun -N "${SLURM_NNODES}" --ntasks-per-node 1 "${ENV_DIR}/bin/conda-unpack"

echo "=== single-rank torch sanity ==="
"${PY}" - <<'PY'
import torch
print("torch:", torch.__file__)
print("torch version:", torch.__version__)
print("torch.version.hip:", getattr(torch.version, "hip", None))
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
PY

echo "=== MPI backend + GPU tensor all_reduce (4 ranks, 1 GPU per rank) ==="
srun --unbuffered -l \
  --ntasks-per-node=4 \
  --gpus-per-task=1 \
  --gpu-bind=closest \
  bash -lc "
set -euo pipefail
'${PY}' - <<'PY'
import os, time, socket
import torch
import torch.distributed as dist

rank = int(os.environ['SLURM_PROCID'])
world = int(os.environ['SLURM_NTASKS'])
host = socket.gethostname()

# With --gpus-per-task=1, each task typically sees exactly 1 GPU.
# So valid device is cuda:0 regardless of SLURM_LOCALID.
dev = torch.device('cuda:0')
torch.cuda.set_device(0)

print(f'[rank {rank}/{world} on {host}] '
      f'ROCR_VISIBLE_DEVICES={os.environ.get(\"ROCR_VISIBLE_DEVICES\")} '
      f'CUDA_VISIBLE_DEVICES={os.environ.get(\"CUDA_VISIBLE_DEVICES\")} '
      f'device_count={torch.cuda.device_count()} using={dev}',
      flush=True)

if rank == 0:
    print('MPICH_GPU_SUPPORT_ENABLED=', os.environ.get('MPICH_GPU_SUPPORT_ENABLED'))
    print('MPICH_OFI_NIC_POLICY=', os.environ.get('MPICH_OFI_NIC_POLICY'))

dist.init_process_group(backend='mpi')

n = 128 * 1024 * 1024 // 4  # ~128 MiB float32
x = torch.ones(n, device=dev, dtype=torch.float32)

torch.cuda.synchronize()
t0 = time.time()
dist.all_reduce(x)
torch.cuda.synchronize()
dt = time.time() - t0

expected = float(world)
ok = abs(x[0].item() - expected) < 1e-5
print(f'[rank {rank}] allreduce ok={ok} time={dt:.3f}s first={x[0].item()}', flush=True)

dist.barrier()
dist.destroy_process_group()
PY
"
echo "=== done ==="
