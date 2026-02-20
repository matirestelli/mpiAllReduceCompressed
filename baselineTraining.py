#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnext101_32x8d  # pip install timm for ConvNeXt
import timm  # pip install timm
import argparse
import pandas as pd  # For CSV
from mpi4py import MPI  # MPI rank source

# the steps are from the guide https://docs.pytorch.org/docs/stable/distributed.html
def get_model(model_name):
    if model_name == "wideresnet":
        from torchvision.models import wide_resnet50_2
        return wide_resnet50_2(num_classes=10)
    elif model_name == "resnext":
        return resnext101_32x8d(weights=None, num_classes=10)
    elif model_name == "convnext":
        return timm.create_model("convnext_base", pretrained=False, num_classes=10)
    raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', default=["wideresnet"], choices=["wideresnet", "resnext", "convnext"])
    parser.add_argument("--lr", type=float, default=0.001)  # From paper
    parser.add_argument("--batch-size", type=int, default=128)  # Global BS from paper
    parser.add_argument("--epochs", type=int, default=100)  # From accuracy plots
    args = parser.parse_args()
    
    # STEP 0: Get ranks FROM MPI (BEFORE torch init!) - paper reproduction
    # mpi4py provides Cray MPICH ranks → sets Torch env vars
    comm = MPI.COMM_WORLD
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29500 + comm.rank)
    os.environ['WORLD_SIZE'] = str(comm.size)
    os.environ['RANK'] = str(comm.rank)
    os.environ['LOCAL_RANK'] = str(comm.rank % 4)  # 4 GPUs/node
    
    # STEP 1: Initialize process group (ONCE for all models)
    # PyTorch DDP tutorial: "Initialize process group across all processes."
    # Enables all-to-all sync. Uses MPI→env vars (LOCAL_RANK, WORLD_SIZE).
    # Backend="nccl" for GPU (fastest); MPI ranks for paper reproduction.
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])

    #STEP 2: Set per-process GPU
    # Tutorial: "Each process has its own GPU."
    torch.cuda.set_device(local_rank)

    # Results tracking (rank 0 only)
    if local_rank == 0:
        results = []  # Track all models
    else:
        results = None
    
    # LOOP: One model at a time
    for model_name in args.models:
        if local_rank == 0:
            print(f"Starting {model_name} on {world_size} GPUs")
        
        #STEP 3: Distributed data loading
        # Tutorial: "Use DistributedSampler to shard data across processes."
        # Ensures each GPU sees different data shard (no duplicates).
        # the transform matrix is the usual for cifar10 gave by torch
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size // world_size,  # Per-GPU batch (global=128)
            sampler=train_sampler, 
            shuffle=False,  # Sampler handles shuffle
            num_workers=4, 
            pin_memory=True,  # Async GPU copy (tutorial best practice)
        )

        #STEP 4: Wrap model in DDP
        # Tutorial exact: "Wrap model with DDP."
        # Auto-syncs gradients via AllReduce in backward/step().
        model = get_model(model_name).cuda(local_rank)
        model = DDP(model, device_ids=[local_rank])  # Paper: "adding a DistributedDataParallel wrapper"

        # STEP 5: Optimizer + Loss 
        # Tutorial: "Create optimizer *after* DDP (uses ddp_model.parameters())."
        # the optimizer and decay where not present in the paper, I used the more common onces for this models and dataset
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        # STEP 6: Training loop
        model.train()
        epoch_times = []
        for epoch in range(args.epochs):
            # Tutorial: "Call set_epoch() each epoch to reshuffle data shards."
            train_sampler.set_epoch(epoch)
            
            # Precise epoch timing (matches paper Fig. 11)
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

            for inputs, targets in train_loader:
                # Tutorial: "Move to device with non_blocking=True for perf."
                inputs, targets = inputs.cuda(local_rank, non_blocking=True), targets.cuda(local_rank, non_blocking=True)
                
                # Core DDP steps (tutorial loop exact) 
                optimizer.zero_grad()     # Clear grads
                outputs = model(inputs)   # Forward
                loss = criterion(outputs, targets)
                loss.backward()           # Backward → DDP auto-syncs grads
                optimizer.step()          # AllReduce gradients across GPUs → paper's focus!

            end_time.record()
            torch.cuda.synchronize()   # Wait for GPU events
            epoch_time = start_time.elapsed_time(end_time) / 1000.0  # Seconds
            epoch_times.append(epoch_time)
            
            # Tutorial: "Print only from rank 0."
            if local_rank == 0:
                print(f"{model_name} Epoch {epoch}: {epoch_time:.2f}s")
        
        # Per-model cleanup
        del model, optimizer, train_loader, train_dataset, train_sampler
        
        if local_rank == 0:
            results.append({
                "model": model_name,
                "world_size": world_size,
                "avg_epoch_time": sum(epoch_times) / len(epoch_times),
                "epochs": args.epochs
            })
            print(f"{model_name} avg: {sum(epoch_times)/len(epoch_times):.2f}s")
    
    # STEP 7: Final cleanup
    # Tutorial: "Destroy process group at end."
    if local_rank == 0:
        pd.DataFrame(results).to_csv(f"results_{world_size}gpus_{args.epochs}epochs.csv", index=False)
        print("Results saved!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
