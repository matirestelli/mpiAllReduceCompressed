from mpi4py import MPI
import torch
import torch

import torch.distributed as dist

import os

def main():
    print("torch:", torch.__version__)
    print("hip:", torch.version.hip)
    print("nccl:", torch.cuda.nccl.version() if torch.cuda.is_available() else None)
    print("cuda.is_available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device0:", torch.cuda.get_device_name(0))

    print("\nDistributed backends:")
    print("  gloo:", dist.is_gloo_available())
    print("  nccl:", dist.is_nccl_available())
    print("  mpi :", dist.is_mpi_available())

    # If launched under torchrun, try to init and allreduce
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if dist.is_nccl_available() and torch.cuda.is_available() else "gloo"
        print(f"\nInitializing process group with backend={backend} ...")
        dist.init_process_group(backend=backend)

        rank = dist.get_rank()
        world = dist.get_world_size()

        if backend == "nccl":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
            t = torch.tensor([rank + 1.0], device="cuda")
        else:
            t = torch.tensor([rank + 1.0])

        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        print(f"rank {rank}/{world}: allreduce result = {t.item()}")

        dist.destroy_process_group()

if __name__ == "__main__":
    main()
