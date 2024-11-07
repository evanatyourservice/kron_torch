import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time

from kron_torch import Kron


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.scalar = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([16, 28, 28])
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([32, 28, 28])
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.ln3 = nn.LayerNorm([64, 14, 14])
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.ln4 = nn.LayerNorm([128, 7, 7])
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.ln5 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 10)
        self.static_param = nn.Parameter(torch.randn(1), requires_grad=False)

    def forward(self, x):
        x = self.scalar * x
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
        x = F.relu(self.ln3(self.conv3(F.max_pool2d(x, 2))))
        x = F.relu(self.ln4(self.conv4(F.max_pool2d(x, 2))))
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.ln5(self.fc1(x)))
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
    start_time = time.time()

    num_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if batch_idx % 5 == 0:
            print(
                f"Epoch: {epoch}, Batch: {batch_idx + 1}/{num_batches}, "
                f"Loss: {loss.item():.6f}"
            )

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")

    assert torch.allclose(
        model.static_param, initial_static_param
    ), "Static parameter changed during training!"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0  # Use 0 workers for CPU to avoid overhead
        pin_memory = False

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model_kron = ConvNet().to(device)
    model_sgd = ConvNet().to(device)
    model_sgd.load_state_dict(model_kron.state_dict())

    if hasattr(torch, "compile"):
        try:
            model_kron = torch.compile(model_kron)
            model_sgd = torch.compile(model_sgd)
            print("Using compiled models")
        except Exception as e:
            print(f"Model compilation not supported on this system: {e}")

    print_model_summary(model_kron)

    print(f"Initial static param (Kron): {model_kron.static_param.item():.4f}")
    print(f"Initial static param (SGD): {model_sgd.static_param.item():.4f}")

    optimizer_kron = Kron(
        model_kron.parameters(),
        lr=0.0001,
        weight_decay=1e-6,
        preconditioner_update_probability=1.0,
        memory_save_mode="one_diag",
    )
    optimizer_sgd = torch.optim.SGD(
        model_sgd.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001
    )

    num_epochs = 3
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

    print("\nKron optimizer states:")
    for group_idx, group in enumerate(optimizer_kron.param_groups):
        print(f"Parameter group {group_idx}:")
        for p in group["params"]:
            if p.requires_grad:
                state = optimizer_kron.state[p]
                print(f"  Parameter: shape={p.shape}, dtype={p.dtype}")
                if state:
                    for key, value in state.items():
                        if key == "Q":
                            print(f"    Q: list of {len(value)} tensors")
                            for i, q_tensor in enumerate(value):
                                print(
                                    f"      Q[{i}]: shape={q_tensor.shape}, dtype={q_tensor.dtype}"
                                )
                        elif key == "exprs":
                            print(f"    exprs: tuple of {len(value)} expressions")
                        elif isinstance(value, torch.Tensor):
                            print(
                                f"    {key}: shape={value.shape}, dtype={value.dtype}"
                            )
                        else:
                            print(f"    {key}: {type(value)}")
                else:
                    print("    No state")

    print("\nOptimizer attributes:")
    for attr_name in dir(optimizer_kron):
        if not attr_name.startswith("_") and attr_name not in ["state", "param_groups"]:
            attr_value = getattr(optimizer_kron, attr_name)
            if not callable(attr_value):
                if isinstance(attr_value, torch.Tensor):
                    print(
                        f"  {attr_name}: shape={attr_value.shape}, dtype={attr_value.dtype}"
                    )
                else:
                    print(f"  {attr_name}: {type(attr_value)}")

    print("\nTraining with SGD optimizer:")
    for epoch in range(1, num_epochs + 1):
        train(model_sgd, device, train_loader, optimizer_sgd, scheduler_sgd, epoch)

    print(f"\nFinal results:")
    print(f"Scalar value (Kron): {model_kron.scalar.item():.4f}")
    print(f"Scalar value (SGD): {model_sgd.scalar.item():.4f}")
    print(f"Final static param (Kron): {model_kron.static_param.item():.4f}")
    print(f"Final static param (SGD): {model_sgd.static_param.item():.4f}")


if __name__ == "__main__":
    main()
