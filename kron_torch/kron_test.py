import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from kron import Kron
import time


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
    start_time = time.time()
    last_print_time = start_time
    total_steps = 0
    steps_since_last_print = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_steps += 1
        steps_since_last_print += 1
        if batch_idx % 50 == 0:
            current_time = time.time()
            elapsed_time = current_time - last_print_time
            steps_per_second = steps_since_last_print / elapsed_time
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                f"\tSteps/sec: {steps_per_second:.2f}"
            )
            last_print_time = current_time
            steps_since_last_print = 0

    total_time = time.time() - start_time
    average_steps_per_second = total_steps / total_time
    print(f"Epoch {epoch} completed. Average steps/sec: {average_steps_per_second:.2f}")

    assert torch.allclose(
        model.static_param, initial_static_param
    ), "Static parameter changed during training!"
    
    return average_steps_per_second  # Return the average steps per second


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
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    model_kron = SimpleConvNet().to(device)
    model_sgd = SimpleConvNet().to(device)
    model_sgd.load_state_dict(model_kron.state_dict())

    model_kron = torch.compile(model_kron)
    model_sgd = torch.compile(model_sgd)

    print_model_summary(model_kron)

    print(f"Initial static param (Kron): {model_kron.static_param.item():.4f}")
    print(f"Initial static param (SGD): {model_sgd.static_param.item():.4f}")

    optimizer_kron = Kron(
        model_kron.parameters(), lr=0.0001, weight_decay=0.0001, max_skew_triangular=1,preconditioner_update_probability=1
    )  # should be largest dim diag, rest tri
    
    # Additional Kron optimizer configurations
    optimizer_kron_no_skew = Kron(
        model_kron.parameters(), lr=0.0001, weight_decay=0.0001,preconditioner_update_probability=1
    )  # No max_skew_triangular setting

    optimizer_kron_max_size_0 = Kron(
        model_kron.parameters(), lr=0.0001, weight_decay=0.0001, max_skew_triangular=0,preconditioner_update_probability=1
    )  # max_skew_triangular set to 0
    optimizer_sgd = torch.optim.SGD(
        model_sgd.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001
    )

    num_epochs = 4
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch

    scheduler_kron = torch.optim.lr_scheduler.LinearLR(
        optimizer_kron, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    scheduler_kron_no_skew = torch.optim.lr_scheduler.LinearLR(
        optimizer_kron_no_skew, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    scheduler_kron_max_size_0 = torch.optim.lr_scheduler.LinearLR(
        optimizer_kron_max_size_0, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    scheduler_sgd = torch.optim.lr_scheduler.LinearLR(
        optimizer_sgd, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )

    print("Training with Kron optimizer:")
    kron_steps_per_second = []
    for epoch in range(1, num_epochs + 1):
        steps_per_second = train(model_kron, device, train_loader, optimizer_kron, scheduler_kron, epoch)
        if epoch > 2:
            kron_steps_per_second.append(steps_per_second)
    kron_accuracy = test(model_kron, device, test_loader)

    print("Training with Kron optimizer (no skew setting):")
    kron_no_skew_steps_per_second = []
    for epoch in range(1, num_epochs + 1):
        steps_per_second = train(model_kron, device, train_loader, optimizer_kron_no_skew, scheduler_kron_no_skew, epoch)
        if epoch > 2:
            kron_no_skew_steps_per_second.append(steps_per_second)
    kron_no_skew_accuracy = test(model_kron, device, test_loader)

    print("Training with Kron optimizer (max skew triangular = 0):")
    kron_max_size_0_steps_per_second = []
    for epoch in range(1, num_epochs + 1):
        steps_per_second = train(model_kron, device, train_loader, optimizer_kron_max_size_0, scheduler_kron_max_size_0, epoch)
        if epoch > 2:
            kron_max_size_0_steps_per_second.append(steps_per_second)
    kron_max_size_0_accuracy = test(model_kron, device, test_loader)

    print("\nTraining with SGD optimizer:")
    sgd_steps_per_second = []
    for epoch in range(1, num_epochs + 1):
        steps_per_second = train(model_sgd, device, train_loader, optimizer_sgd, scheduler_sgd, epoch)
        if epoch > 2:
            sgd_steps_per_second.append(steps_per_second)
    sgd_accuracy = test(model_sgd, device, test_loader)

    # Print average steps per second for each optimizer after skipping first two epochs
    def print_avg_steps_per_second(name, steps_per_second):
        if steps_per_second:
            avg = sum(steps_per_second) / len(steps_per_second)
            print(f"Average steps per second ({name}, after compiling): {avg:.2f}")
        else:
            print(f"Not enough epochs to calculate average steps per second for {name} optimizer")

    print("\nPerformance summary:")
    print_avg_steps_per_second("Kron", kron_steps_per_second)
    print_avg_steps_per_second("Kron no skew", kron_no_skew_steps_per_second)
    print_avg_steps_per_second("Kron max skew 0", kron_max_size_0_steps_per_second)
    print_avg_steps_per_second("SGD", sgd_steps_per_second)

    print(f"\nFinal results:")
    print(f"Kron accuracy: {kron_accuracy:.2f}%")
    print(f"Kron no skew accuracy: {kron_no_skew_accuracy:.2f}%")
    print(f"Kron max skew 0 accuracy: {kron_max_size_0_accuracy:.2f}%")
    print(f"SGD accuracy: {sgd_accuracy:.2f}%")
    print(f"Scalar value (Kron): {model_kron.scalar.item():.4f}")
    print(f"Scalar value (SGD): {model_sgd.scalar.item():.4f}")
    print(f"Final static param (Kron): {model_kron.static_param.item():.4f}")
    print(f"Final static param (SGD): {model_sgd.static_param.item():.4f}")


if __name__ == "__main__":
    main()