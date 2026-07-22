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
