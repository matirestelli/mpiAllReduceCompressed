"""
Simple comparison: MPI backend vs NCCL backend
Same test, same hook with logging, different backends
Purpose: Understand why hooks work/don't work
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys


# Global tracking
hook_calls = 0


def logging_hook(state, bucket):
    """Minimal hook - just log that it was called"""
    global hook_calls
    hook_calls += 1
    
    rank = dist.get_rank()
    print(f"\n[RANK {rank}] *** HOOK CALLED (call #{hook_calls}) ***", flush=True)
    print(f"[RANK {rank}] Bucket size: {bucket.buffer().numel()}", flush=True)
    
    tensor = bucket.buffer()
    future = torch.futures.Future()
    future.set_result(tensor)
    return future


def main():
    # Detect which backend to use
    backend = "mpi"  # Change to "nccl" to test NCCL backend
    
    if len(sys.argv) > 1:
        backend = sys.argv[1]
    
    print("\n" + "="*70)
    print(f"TESTING BACKEND: {backend.upper()}")
    print("="*70 + "\n", flush=True)
    
    try:
        if backend == "nccl":
            # NCCL with mpirun: Set environment variables for rank/world_size
            from mpi4py import MPI
            mpi_rank = MPI.COMM_WORLD.Get_rank()
            mpi_world_size = MPI.COMM_WORLD.Get_size()
            
            import os
            os.environ['RANK'] = str(mpi_rank)
            os.environ['WORLD_SIZE'] = str(mpi_world_size)
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        
        dist.init_process_group(backend=backend)
    except RuntimeError as e:
        print(f"ERROR initializing {backend} backend: {e}")
        import traceback
        traceback.print_exc()
        return
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Device setup depends on backend
    if backend == "nccl":
        # NCCL: Model on GPU
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {rank}] Using device: {device}", flush=True)
    else:
        # MPI: For now, model on CPU (non-CUDA-aware MPI)
        device = torch.device("cpu")
        print(f"[Rank {rank}] Using device: {device}", flush=True)
    
    dist.barrier()
    
    # Create model
    if rank == 0:
        print(f"\n[Setup] Creating model on {device}...")
    
    torch.manual_seed(42)
    model = nn.Linear(10, 10, bias=False)
    model = model.to(device)
    
    # Wrap with DDP
    if backend == "nccl":
        model = DDP(model, device_ids=[local_rank])
    else:
        model = DDP(model)
    
    if rank == 0:
        print(f"[Setup] Model wrapped with DDP")
    
    # Register hook
    if rank == 0:
        print(f"[Setup] Registering logging hook...")
    
    model.register_comm_hook(state=None, hook=logging_hook)
    
    if rank == 0:
        print(f"[Setup] Hook registered")
        print(f"[Setup] Starting training...\n")
    
    dist.barrier()
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    torch.manual_seed(42)
    if backend == "nccl":
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 10, (4,), device=device)
    else:
        x = torch.randn(4, 10)
        y = torch.randint(0, 10, (4,))
    
    if rank == 0:
        print("[Training] Forward pass...")
    
    outputs = model(x)
    loss = criterion(outputs, y)
    
    if rank == 0:
        print(f"[Training] Loss: {loss.item():.6f}")
        print(f"[Training] Backward pass (HOOK SHOULD FIRE HERE)...\n")
    
    loss.backward()
    
    dist.barrier()
    
    if rank == 0:
        print(f"\n[Training] Backward complete")
    
    optimizer.step()
    
    dist.barrier()
    
    # Results
    if rank == 0:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
    
    dist.barrier()
    for r in range(world_size):
        if rank == r:
            print(f"[Rank {rank}] Total hook calls: {hook_calls}")
        dist.barrier()
    
    # Check gradient sync
    if rank == 0:
        print("\n[Verify] Gradient values:")
    
    dist.barrier()
    first_param = next(model.parameters())
    grad_val = first_param.grad[0, 0].item() if first_param.grad is not None else None
    
    for r in range(world_size):
        if rank == r:
            print(f"[Rank {rank}] grad[0,0] = {grad_val:.8f}")
        dist.barrier()
    
    if rank == 0:
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        print("\nIf hook_calls > 0: Hooks ARE being called ✓")
        print("If hook_calls = 0: Hooks NOT being called ✗")
        print("\nIf all ranks have SAME grad value: Synchronized ✓")
        print("If ranks have DIFFERENT values: NOT synchronized ✗")
        print("="*70 + "\n")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()