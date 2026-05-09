"""
Correctness test for DDP communication hooks.

Each rank r produces gradient (r+1) on every weight element.
After a correct allreduce the expected value on every element is:
    (1 + 2 + ... + world_size) / world_size = (world_size + 1) / 2
With 4 ranks: 2.5 exactly. Any other value means the hook is wrong.

Usage:
    mpirun -np 4 ... python toy_train.py [algorithm]

algorithm: ring, recursive_doubling, ring_zfp, recursive_doubling_zfp, etc.
           Omit to use DDP built-in allreduce (no hook).
"""

import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from communication_strategy import get_comm_hook

DIM = 4096


def main():
    algorithm = sys.argv[1] if len(sys.argv) > 1 else None

    dist.init_process_group(backend="mpi")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model = nn.Linear(DIM, 1, bias=False)
    nn.init.zeros_(model.weight)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
                find_unused_parameters=False, static_graph=True)

    result = get_comm_hook("mpi", algorithm)
    if result is not None:
        hook, state = result
        model.register_comm_hook(state=state, hook=hook)
        if rank == 0:
            print(f"[Setup] Hook: mpi/{algorithm}", flush=True)
    else:
        if rank == 0:
            print("[Setup] No hook — DDP built-in MPI allreduce", flush=True)

    # Rank r: x=ones(1,DIM), y=-(r+1)/2  =>  grad = (r+1)*ones after backward
    x = torch.ones(1, DIM, device=device)
    y = torch.tensor([[-(rank + 1) / 2.0]], device=device)

    output = model(x)
    loss = nn.MSELoss()(output, y)
    loss.backward()

    grad = model.module.weight.grad  # shape (1, DIM)
    expected = (world_size + 1) / 2.0
    max_err = (grad - expected).abs().max().item()
    tol = 1e-3

    dist.barrier()
    for r in range(world_size):
        if rank == r:
            g4 = grad[0, :4].tolist()
            status = "PASS" if max_err < tol else "FAIL"
            print(
                f"[Rank {rank}] grad[:4]={[f'{v:.6f}' for v in g4]} "
                f"expected={expected:.4f} max_err={max_err:.2e} --> {status}",
                flush=True,
            )
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
