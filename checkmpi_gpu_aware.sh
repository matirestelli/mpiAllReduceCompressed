#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N mpi-gpu-allreduce-perf
#PBS -q by-gpu
#PBS -A UIC-HPC
#PBS -l select=1:ngpus=4
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o mpi-gpu-allreduce-perf.out

set -euo pipefail

CONDA_PREFIX="/lus/eagle/projects/UIC-HPC/mrest/pytorch_mpi_build_sophia/conda/pt-sophia-openmpi"

module purge
module use /soft/modulefiles
module load compilers/openmpi/5.0.10
source envScriptSophia.sh

export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# ---- Useful debug knobs (keep for first run) ----
export OMPI_MCA_opal_cuda_support=true
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export OMPI_MCA_btl="^openib,uct,smcuda,vader,tcp"


# UCX: start with a "reasonable" set; adjust later if needed
export UCX_TLS=rc,sm,self,cuda_copy,cuda_ipc
export UCX_MEMTYPE_CACHE=n

#logging 
export UCX_LOG_LEVEL=info
export UCX_TLS=rc,sm,self,cuda_copy,cuda_ipc
export UCX_INFO=all


export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Try forcing intra-node to prefer shm + cuda_ipc and exclude rc (IB) entirely for this 1-node test
export UCX_TLS=self,sm,cuda_copy,cuda_ipc
export UCX_NET_DEVICES=none
export OMPI_MCA_pml=ucx

echo "Node: $(hostname)"
which mpirun
mpirun --version | head -n 2 || true

cat > allreduce_perf.py <<'PY'
import os, time
import torch
import torch.distributed as dist

dist.init_process_group("mpi")
rank = dist.get_rank()
world = dist.get_world_size()

ng = torch.cuda.device_count()
torch.cuda.set_device(rank % ng)

def bench(n_elems, iters=200, warmup=20):
    x = torch.ones(n_elems, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    for _ in range(warmup):
        dist.all_reduce(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t1 = time.time()

    dt = (t1 - t0) / iters
    nbytes = x.numel() * x.element_size()
    # effective bandwidth for allreduce ~ 2*(p-1)/p * nbytes / dt (ring model)
    bw = (2*(world-1)/world) * (nbytes / dt) / 1e9
    if rank == 0:
        print(f"world={world} bytes={nbytes} dt={dt*1e3:.3f} ms  est_bw={bw:.2f} GB/s")

for n in [256*1024, 1024*1024, 16*1024*1024]:  # 1MB, 4MB, 64MB (float32)
    bench(n)
PY

echo "== Running 4-rank GPU allreduce perf =="
mpirun -np 4 --bind-to core --map-by ppr:4:node python allreduce_perf.py
