"""
Complete distributed training for ResNet50 on CIFAR-10
Using CPU staging hook for non-CUDA-aware MPI - WORKING VERSION
Based on successful test script that uses broadcast_buffers=True
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import models, transforms, datasets
from mpi4py import MPI
import numpy as np
import os
import time


def cpu_staging_allreduce_hook(state, bucket):
    """
    CPU staging hook for non-CUDA-aware MPI.
    
    Explicitly moves gradients GPU → CPU → MPI → CPU → GPU
    This is the WORKING version from the test script.
    """
    tensor = bucket.buffer()
    original_device = tensor.device
    world_size = dist.get_world_size()
    
    rank = dist.get_rank()
    
    # Move to CPU
    cpu_tensor = tensor.cpu()
    arr = np.ascontiguousarray(cpu_tensor.numpy())
    
    if rank == 0:
        print(f"[Hook] Syncing gradient bucket: shape={tensor.shape}")
    
    # Call non-CUDA-aware MPI on CPU array
    MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, arr, op=MPI.SUM)
    
    # Average gradients
    cpu_tensor.div_(world_size)
    
    # Move back to GPU
    tensor.copy_(cpu_tensor)
    
    # Return Future for DDP
    future = torch.futures.Future()
    future.set_result(tensor)
    return future


def get_data_loaders(batch_size, data_dir="./data"):
    """Load CIFAR-10 with distributed sampler"""
    
    # Normalize CIFAR-10
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    
    # Training augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=val_transform
    )
    
    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=42
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False,
        seed=42
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler


def train_epoch(model, train_loader, criterion, optimizer, epoch, rank, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    if rank == 0:
        print(f"\n[Epoch {epoch}] Training...")
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward - hook fires here
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if rank == 0 and (batch_idx + 1) % 50 == 0:
            accuracy = 100.0 * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, rank, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def main():
    # Initialize distributed training
    dist.init_process_group(backend="mpi")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Training hyperparameters
    num_epochs = 10
    batch_size = 128
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    
    if rank == 0:
        print("\n" + "="*80)
        print(f"Distributed Training: ResNet50 + CIFAR-10 (MPI Backend)")
        print("="*80)
        print(f"Ranks: {world_size}, Epochs: {num_epochs}, Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}, Device: {device}\n")
    
    # Create model with identical seed on all ranks
    torch.manual_seed(42)
    model = models.resnet50(num_classes=10)
    
    if rank == 0:
        print(f"[Setup] ResNet50 model created")
    
    # Step 1: Wrap DDP on CPU with broadcast_buffers=True (like working test script)
    model = DDP(model, broadcast_buffers=True)
    
    if rank == 0:
        print(f"[Setup] Model wrapped with DDP on CPU")
    
    # Step 2: Move to GPU
    model = model.to(device)
    
    if rank == 0:
        print(f"[Setup] Model moved to {device}")
    
    # Step 3: Register CPU staging hook (like working test script)
    model.register_comm_hook(state=None, hook=cpu_staging_allreduce_hook)
    
    if rank == 0:
        print(f"[Setup] CPU staging hook registered\n")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Load data
    if rank == 0:
        print("[Data] Loading CIFAR-10...")
    
    train_loader, val_loader, train_sampler = get_data_loaders(batch_size)
    
    if rank == 0:
        print(f"[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
    
    # Training loop
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, rank, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, rank, device)
        
        # Update scheduler
        scheduler.step()
        
        if rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(checkpoint, "checkpoints/resnet50_best.pth")
                print(f"[Checkpoint] Best model saved (Val Acc: {val_acc:.2f}%)\n")
    
    # Final stats
    elapsed_time = time.time() - start_time
    
    if rank == 0:
        print("="*80)
        print("Training Completed!")
        print("="*80)
        print(f"Total time: {elapsed_time:.2f}s")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Model saved to: checkpoints/resnet50_best.pth")
        print("="*80 + "\n")
    
    # Verify parameter synchronization across ranks
    print("\n" + "="*80)
    print("PARAMETER SYNCHRONIZATION CHECK")
    print("="*80)
    first_param = list(model.module.parameters())[0].detach().cpu()
    first_param_values = first_param.flatten()[:5].tolist()
    
    dist.barrier()
    for r in range(world_size):
        if rank == r:
            print(f"[Rank {rank}] First 5 param values: {[f'{v:.6f}' for v in first_param_values]}")
        dist.barrier()
    
    if rank == 0:
        print("If all ranks show SAME values → ✓ Gradients ARE synchronized")
        print("If all ranks show DIFFERENT values → ✗ Gradients NOT synchronized")
        print("="*80 + "\n")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()