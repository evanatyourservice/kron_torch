import torch
import torch.nn as nn
from kron import Kron
import matplotlib.pyplot as plt
import math

torch.manual_seed(42)


class QuadraticModel(nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([a]))
        self.b = nn.Parameter(torch.tensor([b]))
        self.c = nn.Parameter(torch.tensor([c]))

    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c


def generate_data(num_samples=1000):
    x = torch.linspace(-10, 10, num_samples).unsqueeze(1)
    true_a, true_b, true_c = 2.0, -3.0, 5.0
    y = true_a * x**2 + true_b * x + true_c + torch.randn_like(x) * 0.1
    return x, y, (true_a, true_b, true_c)


def train(model, optimizer, criterion, X, y, num_epochs=1000):
    initial_lr = optimizer.param_groups[0]["lr"]
    losses = []

    for epoch in range(num_epochs):
        current_lr = initial_lr * (1 - epoch / num_epochs)
        optimizer.param_groups[0]["lr"] = current_lr

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch+1:4d}: Loss = {loss.item():.4f}, LR = {current_lr:.6f}, "
                f"a = {model.a.item():.4f}, b = {model.b.item():.4f}, c = {model.c.item():.4f}"
            )

    return losses


def evaluate_model(model, criterion, X, y):
    return criterion(model(X), y).item()


def plot_results(X, y, true_params, model_kron, model_sgd):
    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, alpha=0.1, label="Data")
    x_plot = torch.linspace(-10, 10, 200).unsqueeze(1)
    plt.plot(
        x_plot,
        true_params[0] * x_plot**2 + true_params[1] * x_plot + true_params[2],
        "r-",
        label="True",
    )
    plt.plot(x_plot, model_kron(x_plot).detach(), "g-", label="Kron")
    plt.plot(x_plot, model_sgd(x_plot).detach(), "b-", label="SGD")
    plt.legend()
    plt.title("Quadratic Function Fitting")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("quadratic_fitting.png")
    plt.close()


def main():
    X_train, y_train, true_params = generate_data()

    criterion = nn.MSELoss()
    num_epochs = 1000

    model_kron = QuadraticModel(1.0, 1.0, 1.0)
    model_sgd = QuadraticModel(1.0, 1.0, 1.0)

    optimizer_kron = Kron(model_kron.parameters(), lr=0.02)
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=1.0)

    initial_loss_kron = evaluate_model(model_kron, criterion, X_train, y_train)
    initial_loss_sgd = evaluate_model(model_sgd, criterion, X_train, y_train)

    print("Training Kron")
    losses_kron = train(
        model_kron, optimizer_kron, criterion, X_train, y_train, num_epochs
    )
    print("Training SGD")
    losses_sgd = train(
        model_sgd, optimizer_sgd, criterion, X_train, y_train, num_epochs
    )

    final_loss_kron = evaluate_model(model_kron, criterion, X_train, y_train)
    final_loss_sgd = evaluate_model(model_sgd, criterion, X_train, y_train)

    kron_decrease = (initial_loss_kron - final_loss_kron) / initial_loss_kron * 100
    sgd_decrease = (initial_loss_sgd - final_loss_sgd) / initial_loss_sgd * 100

    print(f"Loss Decrease: Kron: {kron_decrease:.2f}%, SGD: {sgd_decrease:.2f}%")
    print(f"True params: {true_params}")
    print(
        f"Kron params: {model_kron.a.item():.4f}, {model_kron.b.item():.4f}, {model_kron.c.item():.4f}"
    )
    print(
        f"SGD params:  {model_sgd.a.item():.4f}, {model_sgd.b.item():.4f}, {model_sgd.c.item():.4f}"
    )

    if not math.isnan(final_loss_sgd) and not math.isnan(final_loss_kron):
        print(
            "Kron outperformed SGD"
            if kron_decrease > sgd_decrease
            else "SGD outperformed Kron"
        )
    else:
        print("Warning: NaN values encountered")

    plot_results(X_train, y_train, true_params, model_kron, model_sgd)

    plt.figure(figsize=(10, 6))
    plt.plot(losses_kron, label="Kron")
    plt.plot(losses_sgd, label="SGD")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("training_loss.png")
    plt.close()


if __name__ == "__main__":
    main()
