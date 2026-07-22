import os, time, statistics
import torch
import torch.distributed as dist

def envflag(name, default=""):
    v = os.environ.get(name, default)
    return v

def main():
    # MPI backend: rank/world are provided by MPI.
    dist.init_process_group(backend="mpi")
    rank = dist.get_rank()
    world = dist.get_world_size()

    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device("cuda")

    if rank == 0:
        print("torch:", torch.__version__)
        print("torch.version.cuda:", torch.version.cuda)
        print("world:", world)
        print("cuda devices:", torch.cuda.device_count())
        print("OMPI_MCA_opal_cuda_support:", envflag("OMPI_MCA_opal_cuda_support"))
        print("OMPI_MCA_pml:", envflag("OMPI_MCA_pml"))
        print("UCX_TLS:", envflag("UCX_TLS"))
        print("UCX_MEMTYPE_CACHE:", envflag("UCX_MEMTYPE_CACHE"))

    # ---- Allreduce microbench on CUDA tensor ----
    # Choose a size large enough to expose staging.
    # 256 MiB per rank (float32): 256*2^20 / 4 bytes = 67,108,864 elements
    n_elems = 67_108_864
    x = torch.ones(n_elems, device=dev, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    iters = 50
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.time()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    # Report per-rank times
    t_med = statistics.median(times)
    t_min = min(times)
    t_max = max(times)

    # Effective bandwidth estimate (very rough):
    # ring allreduce bytes moved per rank ~ 2*(p-1)/p * Nbytes
    # (standard approximation; good enough to compare Polaris vs Sophia)
    p = world
    nbytes = x.numel() * x.element_size()
    factor = 2.0 * (p - 1) / p
    gb = 1024**3
    bw_gbps = (factor * nbytes / t_med) / gb

    # Gather medians to rank0
    t_tensor = torch.tensor([t_med, t_min, t_max, bw_gbps], device=dev)
    out = [torch.empty_like(t_tensor) for _ in range(world)]
    dist.all_gather(out, t_tensor)

    if rank == 0:
        print("\n=== CUDA allreduce benchmark ===")
        print(f"tensor bytes: {nbytes} ({nbytes/gb:.3f} GiB)")
        for r, v in enumerate(out):
            v = v.cpu().tolist()
            print(f"rank {r}: median {v[0]:.6f}s  min {v[1]:.6f}s  max {v[2]:.6f}s  est_bw {v[3]:.2f} GiB/s")

        # A simple "smoke" check that the allreduce actually happened
        # x should become world after SUM of ones.
        # Validate on rank0 only (avoid extra sync cost).
        y = x[0].item()
        print("post-check x[0] (should be world_size):", y)

    # ---- Toy DDP-ish step (compute + allreduce) ----
    # This mimics training: matmul + gradient-like allreduce.
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)

    # fake "grad" buffer (256 MiB float32 again)
    g = torch.randn(n_elems, device=dev, dtype=torch.float32)

    for _ in range(10):
        c = a @ b
        dist.all_reduce(g)
    torch.cuda.synchronize()

    step_times = []
    steps = 30
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.time()
        c = a @ b
        dist.all_reduce(g)
        torch.cuda.synchronize()
        step_times.append(time.time() - t0)

    st_med = statistics.median(step_times)
    st_tensor = torch.tensor([st_med], device=dev)
    out2 = [torch.empty_like(st_tensor) for _ in range(world)]
    dist.all_gather(out2, st_tensor)
    if rank == 0:
        print("\n=== Toy step (matmul + allreduce) median step time ===")
        for r, v in enumerate(out2):
            print(f"rank {r}: {v.item():.6f}s")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
