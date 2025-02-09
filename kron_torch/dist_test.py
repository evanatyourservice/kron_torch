"""
cd kron_torch && torchrun --standalone --nproc_per_node=N dist_test.py
where N is the number of GPUs you want to use
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torchvision import datasets, transforms

from dist_kron import Kron as DistributedKron
from dist_onesidedkron import OneSidedKron as DistributedOneSidedKron

torch.set_float32_matmul_precision('high')


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

def verify_layer_updates(model, initial_params):
    """Verify that all layers have been updated from their initial values."""
    all_layers_updated = True
    for (name, param), (_, initial_param) in zip(model.named_parameters(), initial_params):
        if torch.allclose(param.data, initial_param):
            print(f"Warning: {name} parameters remained unchanged")
            all_layers_updated = False
    return all_layers_updated

def train(model, device, train_loader, optimizer, epoch, rank, initial_params=None):
    model.train()
    initial_loss = None
    final_loss = None
    layers_updated = True

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 100:
            break

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        if batch_idx == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0 and rank == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

    if rank == 0:
        print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
        if final_loss >= initial_loss:
            print("Warning: Loss did not decrease!")

        if initial_params is not None:
            layers_updated = verify_layer_updates(model.module, initial_params)
            if not layers_updated:
                print("Warning: Some layers were not updated!")

    success = torch.tensor(final_loss < initial_loss and layers_updated, device=device)
    dist.broadcast(success, src=0)
    
    return bool(success.item())

def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    batch_size = max(1, int(64 // world_size))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = DDP(SimpleNet().to(device), device_ids=[rank])
    initial_params = [(name, param.clone().detach()) for name, param in model.module.named_parameters()]

    lr = 0.001
    optimizer = DistributedKron(
        model.parameters(),
        lr=lr,
        preconditioner_update_probability=1.0,
        rank=rank,
        world_size=world_size
    )

    try:
        print(f"Training with DistributedKron on rank {rank}")
        success = train(model, device, loader, optimizer, 1, rank, initial_params)
        if rank == 0 and not success:
            print("DistributedKron test failed!")

        dist.barrier()
        torch.cuda.synchronize()

        model = DDP(SimpleNet().to(device), device_ids=[rank])
        initial_params = [(name, param.clone().detach()) for name, param in model.module.named_parameters()]

        optimizer = DistributedOneSidedKron(
            model.parameters(),
            lr=lr,
            preconditioner_update_probability=1.0,
            rank=rank,
            world_size=world_size
        )

        print(f"Training with DistributedOneSidedKron on rank {rank}")
        success = train(model, device, loader, optimizer, 1, rank, initial_params)
        if rank == 0 and not success:
            print("DistributedOneSidedKron test failed!")

        dist.barrier()
        torch.cuda.synchronize()

    except Exception as e:
        print(f"Rank {rank} encountered error: {e}")
        raise e
    finally:
        try:
            dist.barrier()
            torch.cuda.synchronize()
            dist.destroy_process_group()
        except Exception as e:
            print(f"Rank {rank} cleanup error: {e}")

if __name__ == "__main__":
    main()
