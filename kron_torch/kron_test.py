import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from kron import Kron


class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.scalar = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.ln1 = nn.LayerNorm([10, 12, 12])
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.ln2 = nn.LayerNorm([20, 4, 4])
        self.fc1 = nn.Linear(320, 50)
        self.ln3 = nn.LayerNorm(50)
        self.fc2 = nn.Linear(50, 10)
        self.static_param = nn.Parameter(torch.randn(1), requires_grad=False)

    def forward(self, x):
        x = self.scalar * x
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.ln1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.ln2(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.ln3(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def print_model_summary(model):
    print("Model Summary:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}, requires_grad=True")
            total_params += param.numel()
        else:
            print(f"{name}: {param.shape}, requires_grad=False")
    print(f"Total trainable parameters: {total_params}")


def train(model, device, train_loader, optimizer, scheduler, epoch):
    initial_static_param = model.static_param.clone()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if batch_idx % 50 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
    assert torch.allclose(
        model.static_param, initial_static_param
    ), "Static parameter changed during training!"


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: "
        f"{correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("../data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    model_kron = SimpleConvNet().to(device)
    model_sgd = SimpleConvNet().to(device)
    model_sgd.load_state_dict(model_kron.state_dict())

    print_model_summary(model_kron)

    print(f"Initial static param (Kron): {model_kron.static_param.item():.4f}")
    print(f"Initial static param (SGD): {model_sgd.static_param.item():.4f}")

    optimizer_kron = Kron(model_kron.parameters(), lr=0.001, weight_decay=0.0001)
    optimizer_sgd = torch.optim.SGD(
        model_sgd.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001
    )

    num_epochs = 1
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    scheduler_kron = torch.optim.lr_scheduler.LinearLR(
        optimizer_kron, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    scheduler_sgd = torch.optim.lr_scheduler.LinearLR(
        optimizer_sgd, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )

    print("Training with Kron optimizer:")
    for epoch in range(1, num_epochs + 1):
        train(model_kron, device, train_loader, optimizer_kron, scheduler_kron, epoch)
    kron_accuracy = test(model_kron, device, test_loader)

    print("\nTraining with SGD optimizer:")
    for epoch in range(1, num_epochs + 1):
        train(model_sgd, device, train_loader, optimizer_sgd, scheduler_sgd, epoch)
    sgd_accuracy = test(model_sgd, device, test_loader)

    print(f"\nFinal results:")
    print(f"Kron accuracy: {kron_accuracy:.2f}%")
    print(f"SGD accuracy: {sgd_accuracy:.2f}%")
    print(f"Scalar value (Kron): {model_kron.scalar.item():.4f}")
    print(f"Scalar value (SGD): {model_sgd.scalar.item():.4f}")
    print(f"Final static param (Kron): {model_kron.static_param.item():.4f}")
    print(f"Final static param (SGD): {model_sgd.static_param.item():.4f}")


if __name__ == "__main__":
    main()
