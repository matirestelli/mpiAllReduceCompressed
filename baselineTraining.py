import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from torchvision import datasets, transforms

def main():
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    if not dist.is_mpi_available():
        raise RuntimeError("MPI backend not available")

    dist.init_process_group(backend="mpi")  # Inits MPI here
    
    from mpi4py import MPI  # Now safe
    comm = MPI.COMM_WORLD
    mpi_rank = comm.rank  # Optional if using dist ranks
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % 4
    torch.cuda.set_device(local_rank) 

    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=128 // world_size,
                              sampler=train_sampler, shuffle=False)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    model = nn.Linear(32*32*3, 10).to(device)
    model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        train_sampler.set_epoch(epoch)
        for inputs, targets in train_loader:
            inputs = inputs.view(inputs.size(0), -1).to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
