#!/usr/bin/env python3
import os
import socket

from mpi4py import MPI
import torch
import torch.distributed as dist

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()
    host = socket.gethostname()

    # Show what mpi4py is linked against (good sanity check)
    mpi_ver = MPI.Get_library_version().strip().replace("\n", " | ")
    if rank == 0:
        print(f"[mpi4py] MPI library version: {mpi_ver}", flush=True)

    # Bind 1 rank -> 1 GPU using SLURM_LOCALID
    local_rank = int(os.environ.get("SLURM_LOCALID", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    print(f"[rank {rank}/{world} on {host}] local_rank={local_rank} dev={dev} "
          f"torch={torch.__version__} hip={getattr(torch.version,'hip',None)}",
          flush=True)

    # PyTorch distributed MPI availability
    print(f"[rank {rank}] dist.is_mpi_available()={dist.is_mpi_available()}", flush=True)

    # Initialize process group using MPI backend
    dist.init_process_group(backend="mpi")

    # GPU AllReduce test
    x = torch.tensor([rank + 1.0], device=dev, dtype=torch.float32)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = world * (world + 1) / 2  # sum 1..world
    ok = abs(x.item() - expected) < 1e-5
    print(f"[rank {rank}] GPU all_reduce sum: got={x.item()} expected={expected} ok={ok}",
          flush=True)

    # GPU point-to-point send/recv test (often the first to fail if not GPU-aware)
    if world >= 2:
        if rank == 0:
            t = torch.arange(16, device=dev, dtype=torch.int32)
            dist.send(t, dst=1)
            print(f"[rank 0] sent GPU tensor -> rank 1 (first4={t[:4].tolist()})", flush=True)
        elif rank == 1:
            t = torch.empty(16, device=dev, dtype=torch.int32)
            dist.recv(t, src=0)
            print(f"[rank 1] recv GPU tensor <- rank 0 (first4={t[:4].tolist()})", flush=True)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        print("NOTE: torch doesn't expose a definitive 'GPU-aware MPI = True' flag.\n"
              "      Successful GPU tensor send/recv + collectives is the practical validation.",
              flush=True)

if __name__ == "__main__":
    main()
