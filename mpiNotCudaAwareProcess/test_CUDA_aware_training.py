"""
Distributed training ResNet50 on CIFAR-10
Native CUDA-aware MPI — NO CPU staging hook.

This script tests whether the patched PyTorch build correctly bypasses
the CUDA-aware MPI whitelist check and passes GPU tensors directly to MPI
without any CPU staging. If training converges and parameter values are
identical across ranks → CUDA-aware MPI is working.

Run with:
    source ~/mpiAllReduceCompressed/envScriptV2.sh
    mpirun -np 4 --ppn 4 --depth=8 --cpu-bind depth \
        -env MPICH_GPU_SUPPORT_ENABLED=1 \
        python test_CUDA_aware_training.py
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, models, transforms
 
 
def get_data_loaders(batch_size, rank, world_size, data_dir="./data"):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
 
    train_dataset = datasets.CIFAR10(root=data_dir, train=True,  download=False, transform=train_transform)
    val_dataset   = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=val_transform)
 
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False, seed=42)
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, sampler=val_sampler,   num_workers=0, pin_memory=True)
 
    return train_loader, val_loader, train_sampler
 
 
def train_epoch(model, loader, criterion, optimizer, sampler, epoch, device, rank):
    model.train()
    sampler.set_epoch(epoch)
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        #check to see ddp works
        if epoch == 1 and batch_idx == 0:  # first batch of first epoch
            grad = list(model.parameters())[0].grad.detach().cpu().flatten()[:5]
            print(f"[Rank {rank}] grad[:5] = {grad.tolist()}", flush=True)
    
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return total_loss / len(loader), 100. * correct / total
 
 
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return total_loss / len(loader), 100. * correct / total
 
 
def main():
    dist.init_process_group(backend="mpi")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
 
    # Hyperparams
    num_epochs    = 5
    batch_size    = 128
    learning_rate = 0.1
    momentum      = 0.9
    weight_decay  = 5e-4
 
    # Model — GPU first, then DDP
    torch.manual_seed(42)
    model = models.resnet50(num_classes=10).to(device)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=True)
    
 
    # Optimizer + scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
 
    # Data
    train_loader, val_loader, train_sampler = get_data_loaders(batch_size, rank, world_size)
 
    if rank == 0:
        print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | "
              f"Ranks {world_size} | Epochs {num_epochs} | Batch {batch_size}")
 
    # Training loop
    start = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, train_sampler, epoch, device, rank)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step()
        if rank == 0:
            print(f"[Epoch {epoch}] train loss={train_loss:.4f} acc={train_acc:.2f}% | "
                  f"val loss={val_loss:.4f} acc={val_acc:.2f}%")
 
    if rank == 0:
        print(f"Done in {time.time()-start:.1f}s")
 
 
    dist.destroy_process_group()
 
 
if __name__ == "__main__":
    main()
 