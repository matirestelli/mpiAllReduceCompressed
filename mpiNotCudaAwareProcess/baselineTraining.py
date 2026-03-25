"""
Training with detailed logging to understand what's happening during gradient sync.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import datasets, transforms, models
import argparse
import sys


# ============================================================================
# LOGGING SETUP - Intercept all_reduce calls
# ============================================================================

original_all_reduce = dist.all_reduce
all_reduce_log = []

def logged_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None):
    """Log all_reduce calls to see what's being synchronized"""
    rank = dist.get_rank()
    
    log_entry = {
        'rank': rank,
        'tensor_device': str(tensor.device),
        'tensor_shape': tuple(tensor.shape),
        'tensor_dtype': str(tensor.dtype),
        'tensor_requires_grad': tensor.requires_grad,
        'operation': str(op),
    }
    all_reduce_log.append(log_entry)
    
    # Print immediately for real-time monitoring
    print(f"[Rank {rank}] all_reduce called: device={tensor.device}, shape={tensor.shape}, "
          f"dtype={tensor.dtype}, requires_grad={tensor.requires_grad}", flush=True)
    
    try:
        # Call the original all_reduce
        return original_all_reduce(tensor, op, group)
    except RuntimeError as e:
        print(f"[Rank {rank}] all_reduce FAILED: {e}", flush=True)
        raise

# Monkey-patch
dist.all_reduce = logged_all_reduce


# ============================================================================
# CUSTOM COMM HOOK - Log what DDP is doing
# ============================================================================

def logging_allreduce_hook(state, bucket):
    """
    Custom hook to log DDP's gradient synchronization in detail.
    
    This hook intercepts each gradient bucket and shows:
    - Where the tensor is (CPU or GPU)
    - What it contains
    - When synchronization happens
    """
    rank = dist.get_rank()
    tensor = bucket.buffer()
    
    print(f"\n[Rank {rank}] ========== GRADIENT BUCKET SYNC ==========", flush=True)
    print(f"[Rank {rank}] Bucket device: {tensor.device}", flush=True)
    print(f"[Rank {rank}] Bucket shape: {tensor.shape}", flush=True)
    print(f"[Rank {rank}] Bucket dtype: {tensor.dtype}", flush=True)
    print(f"[Rank {rank}] Bucket requires_grad: {tensor.requires_grad}", flush=True)
    print(f"[Rank {rank}] Bucket sample values: {tensor.flatten()[:5]}", flush=True)
    
    # Call the default hook to do actual synchronization
    print(f"[Rank {rank}] Calling dist.all_reduce on bucket...", flush=True)
    
    try:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] ✓ all_reduce succeeded!", flush=True)
    except RuntimeError as e:
        print(f"[Rank {rank}] ✗ all_reduce FAILED: {e}", flush=True)
        raise
    
    # Average gradients
    tensor /= dist.get_world_size()
    
    print(f"[Rank {rank}] After reduce - sample values: {tensor.flatten()[:5]}", flush=True)
    print(f"[Rank {rank}] ==========================================\n", flush=True)
    
    # Return a Future as DDP expects
    future = torch.futures.Future()
    future.set_result(tensor)
    return future


# ============================================================================
# MODEL AND TRAINING FUNCTIONS
# ============================================================================

def get_model(name):
    """Load model from torchvision"""
    if name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    elif name == "resnext101":
        model = models.resnext101_32x8d(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    elif name == "convnext_base":
        model = models.convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 10)
        return model
    else:
        raise ValueError(f"Unknown model: {name}")


def train_epoch(model, train_loader, optimizer, criterion, device, rank, num_batches=1):
    """Train for specified number of batches"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"BATCH {batch_idx + 1}")
            print(f"{'='*70}", flush=True)
        
        inputs = inputs.to(device)
        targets = targets.to(device)

        if rank == 0:
            print(f"[Rank {rank}] Forward pass...", flush=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        if rank == 0:
            print(f"[Rank {rank}] Loss computed: {loss.item():.4f}", flush=True)
            print(f"[Rank {rank}] Loss device: {loss.device}", flush=True)
            print(f"[Rank {rank}] Starting backward()...", flush=True)
        
        # BACKWARD - This triggers gradient synchronization
        loss.backward()
        
        if rank == 0:
            print(f"[Rank {rank}] Backward() completed", flush=True)
            print(f"[Rank {rank}] Starting optimizer.step()...", flush=True)
        
        optimizer.step()
        
        if rank == 0:
            print(f"[Rank {rank}] Optimizer step completed", flush=True)

        total_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["resnet50"],
                        choices=["resnet50", "resnext101", "convnext_base"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batches", type=int, default=1,
                        help="Number of batches to train (for logging)")
    parser.add_argument("--use-hook", type=int, default=0,
                        help="1=use logging hook, 0=default sync")
    args = parser.parse_args()

    # Initialize distributed training
    dist.init_process_group(backend="mpi")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("\n" + "="*70)
        print("DDP TRAINING WITH DETAILED LOGGING")
        print("="*70)
        print(f"Training on {world_size} GPUs with MPI backend")
        print(f"Models: {args.models}")
        print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
        print(f"Training {args.batches} batch(es) for logging")
        print(f"Using logging hook: {bool(args.use_hook)}")
        print("="*70 + "\n")

    # Data loading
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform
    )

    per_rank_batch = args.batch_size // world_size
    if per_rank_batch <= 0:
        raise ValueError(f"Batch size {args.batch_size} too small for {world_size} GPUs")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_rank_batch,
        sampler=train_sampler,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss()

    # Train each model
    for model_name in args.models:
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Training {model_name}")
            print(f"{'='*70}\n")

        # Create model
        model = get_model(model_name)
        
        if rank == 0:
            print(f"[Rank {rank}] Created model on CPU")
            print(f"[Rank {rank}] Model parameters device: {next(model.parameters()).device}")
        
        # Wrap with DDP (on CPU)
        model = DDP(model, broadcast_buffers=False)
        
        if rank == 0:
            print(f"[Rank {rank}] Wrapped with DDP (broadcast_buffers=False)")
            print(f"[Rank {rank}] Model parameters device after DDP: {next(model.parameters()).device}")
        
        # Move to GPU
        model = model.to(device)
        
        if rank == 0:
            print(f"[Rank {rank}] Moved model to {device}")
            print(f"[Rank {rank}] Model parameters device after .to(device): {next(model.parameters()).device}")
        
        # Register logging hook if requested
        if args.use_hook:
            if rank == 0:
                print(f"[Rank {rank}] Registering logging comm hook")
            model.register_comm_hook(state=None, hook=logging_allreduce_hook)
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        # Training loop
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            
            if rank == 0:
                print(f"\nStarting epoch {epoch + 1}...")
            
            try:
                avg_loss, accuracy = train_epoch(
                    model, train_loader, optimizer, criterion, device, rank,
                    num_batches=args.batches
                )
                
                if rank == 0:
                    print(f"\n{'='*70}")
                    print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
                    print(f"{'='*70}\n")
                    
            except RuntimeError as e:
                if rank == 0:
                    print(f"\n{'='*70}")
                    print(f"ERROR during training: {e}")
                    print(f"{'='*70}\n")
                raise

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Summary
    if rank == 0:
        print("\n" + "="*70)
        print("TRAINING COMPLETE - SUMMARY OF all_reduce CALLS")
        print("="*70)
        print(f"Total all_reduce calls: {len(all_reduce_log)}")
        
        if all_reduce_log:
            print("\nall_reduce calls:")
            for i, call in enumerate(all_reduce_log):
                print(f"  Call {i+1}: device={call['tensor_device']}, shape={call['tensor_shape']}, "
                      f"requires_grad={call['tensor_requires_grad']}")
        
        print("="*70 + "\n")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()