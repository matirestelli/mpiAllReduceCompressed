"""
Diagnostic script for DDP communication hooks with NON-CUDA-AWARE MPI

Key insight: With non-CUDA-aware MPI, the model should stay on CPU
for communication, then move to GPU for computation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
import sys
import traceback


def simple_allreduce_hook(state, bucket):
    """Minimal hook to detect if it's called"""
    rank = dist.get_rank()
    print(f"\n{'='*60}")
    print(f"[RANK {rank}] *** HOOK WAS CALLED ***")
    print(f"{'='*60}\n", flush=True)
    
    tensor = bucket.buffer()
    future = torch.futures.Future()
    future.set_result(tensor)
    return future


def main():
    dist.init_process_group(backend="mpi")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # For non-CUDA-aware MPI: Keep model on CPU for communication
    device = torch.device("cpu")
    if torch.cuda.is_available():
        gpu_device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        gpu_device = torch.device("cpu")
    
    if rank == 0:
        print("\n" + "="*70)
        print("DDP COMMUNICATION HOOK DIAGNOSTIC")
        print("NON-CUDA-AWARE MPI MODE")
        print("="*70 + "\n")
        print(f"Communication device: {device} (CPU for MPI)")
        if gpu_device.type == 'cuda':
            print(f"Compute device: {gpu_device} (GPU after sync)")
        print()
    
    # TEST 1: Check if DDP uses hooks with MPI backend (model on CPU)
    if rank == 0:
        print("\n[TEST 1] DDP with MPI backend (model on CPU for communication)")
        print("-" * 70)
    
    dist.barrier()
    
    configs = [
        {
            "name": "Config A: Default (broadcast_buffers=True)",
            "kwargs": {"broadcast_buffers": True}
        },
        {
            "name": "Config B: broadcast_buffers=False",
            "kwargs": {"broadcast_buffers": False}
        },
        {
            "name": "Config C: find_unused_parameters=True",
            "kwargs": {"broadcast_buffers": True, "find_unused_parameters": True}
        },
    ]
    
    for config in configs:
        if rank == 0:
            print(f"\n>>> Testing: {config['name']}")
        
        dist.barrier()
        
        # Create fresh model for each config
        torch.manual_seed(42)
        test_model = nn.Linear(10, 10, bias=False)
        
        try:
            # For non-CUDA-aware MPI: DDP wraps CPU model, stays on CPU for AllReduce
            # Then we move to GPU for actual forward/backward computation
            test_model = DDP(test_model, **config['kwargs'])
            
            if rank == 0:
                print(f"    ✓ DDP wrapping successful (model on CPU)")
            
            # Register hook BEFORE any forward pass
            if rank == 0:
                print(f"    Registering hook...")
            test_model.register_comm_hook(state=None, hook=simple_allreduce_hook)
            
            if rank == 0:
                print(f"    ✓ Hook registered successfully")
            
            # Create input on CPU (for MPI communication)
            torch.manual_seed(42)
            x = torch.randn(4, 10)
            y_target = torch.zeros(4, dtype=torch.long)
            
            # Forward + backward on CPU
            if rank == 0:
                print(f"    Running forward/backward on CPU...")
            
            criterion = nn.CrossEntropyLoss()
            outputs = test_model(x)
            loss = criterion(outputs, y_target)
            loss.backward()
            
            if rank == 0:
                print(f"    ✓ Forward/backward completed")
                print(f"    ? Did you see [RANK X] *** HOOK WAS CALLED *** ?")
            
        except Exception as e:
            if rank == 0:
                print(f"    ✗ ERROR: {e}")
                traceback.print_exc()
        
        dist.barrier()
    
    # TEST 2: Verify hook registration
    if rank == 0:
        print("\n\n[TEST 2] Verifying hook registration (CPU model)")
        print("-" * 70)
    
    dist.barrier()
    
    torch.manual_seed(42)
    model = nn.Linear(10, 10, bias=False)
    model = DDP(model)
    
    # Check internal state BEFORE registering
    if rank == 0:
        print(f"\nBEFORE hook registration:")
        print(f"  model._communication_hook: {getattr(model, '_communication_hook', 'NOT FOUND')}")
    
    dist.barrier()
    
    model.register_comm_hook(state=None, hook=simple_allreduce_hook)
    
    if rank == 0:
        print(f"\nAFTER hook registration:")
        print(f"  model._communication_hook: {getattr(model, '_communication_hook', 'NOT FOUND')}")
    
    dist.barrier()
    
    # TEST 3: Check what backend is being used
    if rank == 0:
        print("\n\n[TEST 3] Backend information")
        print("-" * 70)
        print(f"\ndist.get_backend(): {dist.get_backend()}")
        print(f"model.process_group: {model.process_group if hasattr(model, 'process_group') else 'NOT FOUND'}")
    
    dist.barrier()
    
    # TEST 4: Full training loop on CPU
    if rank == 0:
        print("\n\n[TEST 4] Full training loop on CPU")
        print("-" * 70)
    
    dist.barrier()
    
    torch.manual_seed(42)
    model = nn.Linear(10, 10, bias=False)
    model = DDP(model)
    model.register_comm_hook(state=None, hook=simple_allreduce_hook)
    
    dist.barrier()
    
    if rank == 0:
        print("\nRunning training on CPU with hook...")
    
    torch.manual_seed(42)
    x = torch.randn(4, 10)
    y_target = torch.zeros(4, dtype=torch.long)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    if rank == 0:
        print("Forward pass...")
    outputs = model(x)
    loss = criterion(outputs, y_target)
    
    if rank == 0:
        print("Backward pass (HOOK SHOULD FIRE HERE)...")
    
    loss.backward()
    
    if rank == 0:
        print("Backward completed.\n")
    
    dist.barrier()
    
    # TEST 5: Check if model needs to be GPU-aware for DDP to use hooks
    if rank == 0:
        print("\n\n[TEST 5] Testing with GPU tensors but CPU model")
        print("-" * 70)
        print("(This simulates: model on CPU for AllReduce, but working with GPU data)")
    
    dist.barrier()
    
    torch.manual_seed(42)
    model = nn.Linear(10, 10, bias=False)
    model = DDP(model)
    model.register_comm_hook(state=None, hook=simple_allreduce_hook)
    
    dist.barrier()
    
    if rank == 0:
        print("\nRunning training with GPU inputs but CPU model...")
    
    try:
        torch.manual_seed(42)
        x_gpu = torch.randn(4, 10, device=gpu_device)
        y_gpu = torch.zeros(4, dtype=torch.long, device=gpu_device)
        
        criterion = nn.CrossEntropyLoss()
        
        if rank == 0:
            print("Forward pass with GPU inputs...")
        
        # This might fail because CPU model can't take GPU inputs
        outputs = model(x_gpu)
        loss = criterion(outputs, y_gpu)
        
        if rank == 0:
            print("Backward pass...")
        
        loss.backward()
        
        if rank == 0:
            print("✓ Completed with GPU inputs\n")
    
    except RuntimeError as e:
        if rank == 0:
            print(f"✗ Expected error: {e}\n")
    
    dist.barrier()
    
    # TEST 6: Explore DDP internals
    if rank == 0:
        print("\n\n[TEST 6] DDP internals exploration")
        print("-" * 70)
        
        torch.manual_seed(42)
        model = nn.Linear(10, 10, bias=False)
        model = DDP(model)
        
        print("\nDDP hook-related attributes:")
        for attr in sorted(dir(model)):
            if 'hook' in attr.lower() or 'comm' in attr.lower():
                try:
                    val = getattr(model, attr)
                    if not callable(val):
                        print(f"  {attr}: {val}")
                except:
                    pass
    
    if rank == 0:
        print("\n" + "="*70)
        print("DIAGNOSIS COMPLETE")
        print("="*70 + "\n")
        print("KEY FINDINGS:")
        print("  1. Non-CUDA-aware MPI requires:")
        print("     - Model stays on CPU for AllReduce")
        print("     - DDP wrapping without device_ids")
        print("     - CPU tensors for gradient synchronization")
        print("\n  2. If you see [RANK X] *** HOOK WAS CALLED ***:")
        print("     → Hooks work with MPI backend! (surprise!)")
        print("     → Your custom hook strategy can work")
        print("\n  3. If you DON'T see hook messages:")
        print("     → MPI backend ignores hooks (as suspected)")
        print("     → Use manual AllReduce approach instead")
        print("\n  4. Next steps:")
        print("     - Run manual_allreduce_solution.py for working alternative")
        print("     - Or switch to NCCL/Gloo backend if available")
        print("\n")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()