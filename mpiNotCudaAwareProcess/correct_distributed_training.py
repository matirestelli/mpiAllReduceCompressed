"""
Enhanced test script for DDP communication hooks with DETAILED DIAGNOSTICS
Purpose: Definitively verify if hooks are actually being called during backward pass
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI
import sys
import time
from datetime import datetime


# Global state to track hook calls
hook_call_log = {
    'calls': 0,
    'timestamps': [],
    'buckets_processed': 0,
}


def get_timestamp():
    """Get formatted timestamp for logging"""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def log_message(rank, level, message):
    """Structured logging with rank and timestamp"""
    timestamp = get_timestamp()
    prefix = f"[{timestamp}] [RANK-{rank:02d}] [{level:^10}]"
    print(f"{prefix} {message}", flush=True)
    sys.stderr.flush()


def cpu_staging_allreduce_hook(state, bucket):
    """
    CPU staging hook for non-CUDA-aware MPI.
    With EXTENSIVE logging to verify it's called and understand flow.
    """
    global hook_call_log
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Increment call counter
    hook_call_log['calls'] += 1
    hook_call_log['timestamps'].append(get_timestamp())
    call_num = hook_call_log['calls']
    
    tensor = bucket.buffer()
    original_device = tensor.device
    bucket_size = tensor.numel()
    
    # ===== ENTRY POINT LOGGING =====
    log_message(rank, "HOOK", f"╔{'='*60}")
    log_message(rank, "HOOK", f"║ HOOK EXECUTION #{call_num}")
    log_message(rank, "HOOK", f"╠{'='*60}")
    
    log_message(rank, "INFO", f"Tensor device: {original_device}")
    log_message(rank, "INFO", f"Tensor shape: {tensor.shape}")
    log_message(rank, "INFO", f"Tensor dtype: {tensor.dtype}")
    log_message(rank, "INFO", f"Tensor elements: {bucket_size}")
    log_message(rank, "INFO", f"Tensor requires_grad: {tensor.requires_grad}")
    log_message(rank, "INFO", f"World size: {world_size}")
    
    # Get gradient stats BEFORE allreduce
    log_message(rank, "INFO", f"BEFORE allreduce stats:")
    log_message(rank, "INFO", f"  - min: {tensor.min().item():.6e}")
    log_message(rank, "INFO", f"  - max: {tensor.max().item():.6e}")
    log_message(rank, "INFO", f"  - mean: {tensor.mean().item():.6e}")
    log_message(rank, "INFO", f"  - sum: {tensor.sum().item():.6e}")
    
    # ===== STAGE 1: CPU TRANSFER =====
    log_message(rank, "INFO", f"Stage 1: Transferring to CPU...")
    try:
        cpu_tensor = tensor.cpu()
        log_message(rank, "OK", f"Successfully moved tensor to CPU")
    except Exception as e:
        log_message(rank, "ERROR", f"Failed to move to CPU: {e}")
        raise
    
    # ===== STAGE 2: CONVERT TO NUMPY =====
    log_message(rank, "INFO", f"Stage 2: Converting to numpy array...")
    try:
        #arr = cpu_tensor.numpy()
        arr = np.ascontiguousarray(cpu_tensor.numpy())  # Explicit copy
        log_message(rank, "OK", f"Numpy array shape: {arr.shape}, dtype: {arr.dtype}")
    except Exception as e:
        log_message(rank, "ERROR", f"Failed to convert to numpy: {e}")
        raise
    
    # ===== STAGE 3: MPI ALLREDUCE =====
    log_message(rank, "INFO", f"Stage 3: Calling MPI.Allreduce on CPU buffer...")
    log_message(rank, "INFO", f"  - Calling MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, arr, MPI.SUM)...")
    
    try:
        start_time = time.time()
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, arr, op=MPI.SUM)
        elapsed = time.time() - start_time
        log_message(rank, "OK", f"MPI.Allreduce completed in {elapsed:.4f}s")
    except Exception as e:
        log_message(rank, "ERROR", f"MPI.Allreduce failed: {e}")
        raise
    
    # ===== STAGE 4: AVERAGING =====
    log_message(rank, "INFO", f"Stage 4: Averaging gradients by {world_size}...")
    try:
        cpu_tensor.div_(world_size)
        log_message(rank, "OK", f"Averaging complete")
    except Exception as e:
        log_message(rank, "ERROR", f"Averaging failed: {e}")
        raise
    
    # ===== STAGE 5: TRANSFER BACK TO GPU =====
    log_message(rank, "INFO", f"Stage 5: Transferring result back to GPU ({original_device})...")
    try:
        tensor.copy_(cpu_tensor)
        log_message(rank, "OK", f"Successfully copied back to GPU")
    except Exception as e:
        log_message(rank, "ERROR", f"Failed to copy back to GPU: {e}")
        raise
    
    # Get gradient stats AFTER allreduce
    log_message(rank, "INFO", f"AFTER allreduce stats:")
    log_message(rank, "INFO", f"  - min: {tensor.min().item():.6e}")
    log_message(rank, "INFO", f"  - max: {tensor.max().item():.6e}")
    log_message(rank, "INFO", f"  - mean: {tensor.mean().item():.6e}")
    log_message(rank, "INFO", f"  - sum: {tensor.sum().item():.6e}")
    
    # ===== CREATE FUTURE =====
    log_message(rank, "INFO", f"Stage 6: Creating and returning Future...")
    try:
        future = torch.futures.Future()
        future.set_result(tensor)
        log_message(rank, "OK", f"Future created and set")
    except Exception as e:
        log_message(rank, "ERROR", f"Failed to create Future: {e}")
        raise
    
    hook_call_log['buckets_processed'] += 1
    
    log_message(rank, "HOOK", f"║ HOOK EXECUTION #{call_num} - SUCCESS")
    log_message(rank, "HOOK", f"╚{'='*60}\n")
    
    return future


def main():
    # ===== DISTRIBUTED INITIALIZATION =====
    print("\n" + "="*70)
    print("INITIALIZING DISTRIBUTED TRAINING")
    print("="*70 + "\n", flush=True)
    
    dist.init_process_group(backend="mpi")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        log_message(rank, "INFO", f"World size: {world_size}")
        log_message(rank, "INFO", f"CUDA devices available: {torch.cuda.device_count()}")
        log_message(rank, "INFO", f"Rank 0 assigned to: {device}")
    
    dist.barrier()
    
    # ===== MODEL SETUP =====
    if rank == 0:
        print("\n" + "="*70)
        print("MODEL SETUP")
        print("="*70 + "\n", flush=True)
    
    torch.manual_seed(42)
    model = nn.Linear(10, 10)
    
    log_message(rank, "INFO", f"Linear(10,10) model created on CPU")
    
    # Wrap with DDP
    model = DDP(model, broadcast_buffers=True)
    log_message(rank, "INFO", f"Model wrapped with DDP (broadcast_buffers=True)")
    
    # Move to GPU
    model = model.to(device)
    log_message(rank, "INFO", f"Model moved to {device}")
    
    # Print model structure on rank 0
    if rank == 0:
        log_message(rank, "INFO", f"Model structure:\n{model}")
    
    dist.barrier()
    
    # ===== HOOK REGISTRATION =====
    if rank == 0:
        print("\n" + "="*70)
        print("COMMUNICATION HOOK REGISTRATION")
        print("="*70 + "\n", flush=True)
    
    log_message(rank, "INFO", f"Registering CPU staging hook...")
    model.register_comm_hook(state=None, hook=cpu_staging_allreduce_hook)
    log_message(rank, "OK", f"Hook registered successfully!")
    
    if rank == 0:
        log_message(rank, "WARN", f"** Hook will fire DURING backward pass **")
        log_message(rank, "WARN", f"** Watch for [HOOK] messages below **")
    
    dist.barrier()
    
    # ===== TRAINING SETUP =====
    if rank == 0:
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70 + "\n", flush=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create identical input on all ranks
    torch.manual_seed(42)
    x = torch.randn(4, 10, device=device)
    y_target = torch.randint(0, 10, (4,), device=device)
    
    log_message(rank, "INFO", f"Input data created:")
    log_message(rank, "INFO", f"  - x.shape: {x.shape}, device: {x.device}")
    log_message(rank, "INFO", f"  - y.shape: {y_target.shape}, device: {y_target.device}")
    log_message(rank, "INFO", f"  - x.sum(): {x.sum().item():.6f}")
    
    dist.barrier()
    
    # ===== FORWARD PASS =====
    if rank == 0:
        log_message(rank, "INFO", f"Starting forward pass...")
    
    outputs = model(x)
    loss = criterion(outputs, y_target)
    
    log_message(rank, "OK", f"Forward pass complete, loss: {loss.item():.6f}")
    
    dist.barrier()
    
    # ===== BACKWARD PASS - THIS IS WHERE HOOK SHOULD FIRE =====
    if rank == 0:
        print("\n" + "-"*70)
        print(">>> BACKWARD PASS STARTING - HOOK SHOULD FIRE HERE <<<")
        print("-"*70 + "\n", flush=True)
    
    log_message(rank, "WARN", f"About to call loss.backward()...")
    time.sleep(0.1)  # Small delay to ensure log order
    
    loss.backward()  # <-- THIS TRIGGERS THE HOOK
    
    if rank == 0:
        print("\n" + "-"*70)
        print(">>> BACKWARD PASS COMPLETED <<<")
        print("-"*70 + "\n", flush=True)
    
    log_message(rank, "OK", f"Backward pass complete")
    
    dist.barrier()
    
    # ===== OPTIMIZER STEP =====
    log_message(rank, "INFO", f"Optimizer step...")
    optimizer.step()
    log_message(rank, "OK", f"Optimizer step complete")
    
    dist.barrier()
    
    # ===== VERIFICATION =====
    if rank == 0:
        print("\n" + "="*70)
        print("HOOK EXECUTION SUMMARY")
        print("="*70 + "\n")
    
    # Collect hook statistics
    dist.barrier()
    for r in range(world_size):
        if rank == r:
            log_message(rank, "STAT", f"Total hook calls: {hook_call_log['calls']}")
            log_message(rank, "STAT", f"Buckets processed: {hook_call_log['buckets_processed']}")
            if hook_call_log['calls'] > 0:
                log_message(rank, "STAT", f"Call timestamps: {', '.join(hook_call_log['timestamps'])}")
        dist.barrier()
    
    # Collect gradient values from all ranks
    if rank == 0:
        print("\n" + "="*70)
        print("GRADIENT SYNCHRONIZATION CHECK")
        print("="*70 + "\n")
    
    dist.barrier()
    first_param = next(model.module.parameters())  # Note: use .module for DDP
    grad_value = first_param.grad[0, 0].item() if first_param.grad is not None else None
    grad_sum = first_param.grad.sum().item() if first_param.grad is not None else None
    
    for r in range(world_size):
        if rank == r:
            if grad_value is not None:
                log_message(rank, "DATA", f"grad[0,0]={grad_value:.8f}, sum={grad_sum:.8f}")
            else:
                log_message(rank, "WARN", f"No gradients found!")
        dist.barrier()
    
    if rank == 0:
        print("\n" + "="*70)
        print("INTERPRETATION")
        print("="*70)
        print("\n✓ If [HOOK] messages appeared above:")
        print("  → Hook IS being called (good!)")
        print("\n✓ If all ranks show SAME gradient values:")
        print("  → Gradients ARE synchronized (hook works!)")
        print("\n✗ If NO [HOOK] messages appeared:")
        print("  → Hook NOT being called (problem!)")
        print("\n✗ If ranks show DIFFERENT gradient values:")
        print("  → Gradients NOT synchronized (hook failed!)")
        print("\n" + "="*70 + "\n")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()