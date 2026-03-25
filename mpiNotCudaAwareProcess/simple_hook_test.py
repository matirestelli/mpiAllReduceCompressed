"""
Test: Register hook AFTER moving to GPU

Question: Does hook registration timing matter?
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI

def cpu_staging_hook(state, bucket):
    """CPU staging hook"""
    rank = dist.get_rank()
    tensor = bucket.buffer()
    
    print(f"\n[RANK {rank}] HOOK FIRED!", flush=True)
    print(f"[RANK {rank}]   Tensor device: {tensor.device}", flush=True)
    
    # If on GPU, move to CPU for AllReduce
    if tensor.device.type == 'cuda':
        cpu_tensor = tensor.cpu()
    else:
        cpu_tensor = tensor
    
    arr = cpu_tensor.numpy()
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, arr, op=MPI.SUM)
    
    world_size = dist.get_world_size()
    cpu_tensor.div_(world_size)
    
    if tensor.device.type == 'cuda':
        tensor.copy_(cpu_tensor.to(tensor.device))
    else:
        tensor.copy_(cpu_tensor)
    
    future = torch.futures.Future()
    future.set_result(tensor)
    return future

def main():
    dist.init_process_group(backend="mpi")
    rank = dist.get_rank()
    
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"[Rank {rank}] Device: {device}")
    
    # Step 1: Create on CPU
    model = nn.Linear(10, 10)
    print(f"[Rank {rank}] Model created on CPU")
    
    # Step 2: Wrap with DDP on CPU
    model = DDP(model)
    print(f"[Rank {rank}] Wrapped with DDP on CPU")
    
    # Step 3: Move to GPU
    model = model.to(device)
    print(f"[Rank {rank}] Moved to {device}")
    
    # Step 4: Register hook AFTER moving to GPU
    print(f"[Rank {rank}] Registering hook AFTER moving to GPU...")
    model.register_comm_hook(state=None, hook=cpu_staging_hook)
    print(f"[Rank {rank}] Hook registered")
    
    dist.barrier()
    
    # Forward + backward
    x = torch.randn(4, 10, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    
    criterion = nn.CrossEntropyLoss()
    outputs = model(x)
    loss = criterion(outputs, y)
    
    print(f"[Rank {rank}] About to backward...")
    loss.backward()
    print(f"[Rank {rank}] Backward done")
    
    dist.barrier()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()