"""
cd kron_torch && python kron_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from kron import Kron

torch.set_float32_matmul_precision('high')


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(F.max_pool2d(x, 2)))
        x = x.view(-1, 16 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64)

    model_kron = ConvNet().to(device)
    model_sgd = ConvNet().to(device)
    model_sgd.load_state_dict(model_kron.state_dict())

    if hasattr(torch, "compile"):
        try:
            model_kron = torch.compile(model_kron)
            model_sgd = torch.compile(model_sgd)
            print("Using compiled models")
        except Exception as e:
            print(f"Model compilation failed: {e}")

    optimizer_kron = Kron(
        model_kron.parameters(),
        lr=0.0001,
        weight_decay=1e-6,
        memory_save_mode="one_diag",
    )
    optimizer_sgd = torch.optim.SGD(
        model_sgd.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001
    )

    print("\nTraining with Kron optimizer:")
    for epoch in range(1, 3):
        train(model_kron, device, train_loader, optimizer_kron, epoch)

    print("\nTraining with SGD optimizer:")
    for epoch in range(1, 3):
        train(model_sgd, device, train_loader, optimizer_sgd, epoch)


if __name__ == "__main__":
    main()
