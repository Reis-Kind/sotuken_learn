import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class BinaryConnectMnist(nn.Module):
    def __init__(self):
        super().__init__()

        # 実数の重み
        self.fc1 = nn.Linear(784, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)
        self.layers = nn.ModuleList([self.fc1, self.fc2])

        # 二値化の重み
        self.b_fc1 = nn.Linear(784, 128, bias=False)
        self.b_fc2 = nn.Linear(128, 10, bias=False)
        self.b_layers = nn.ModuleList([self.b_fc1, self.b_fc2])

        # 神の一手
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(10)

        self.relu = nn.ReLU()


    def binarize(self):
        for layers, b_layers in zip(self.layers, self.b_layers):
            b_layers.weight.data = torch.where(layers.weight.data >= 0, 1.0, -1.0)
            if layers.bias is not None and b_layers.bias is not None:
                b_layers.bias.data = layers.bias.data.clone()


    def forward(self, x):
        x = x.view(x.size(0), -1)
        self.binarize()
        x = self.b_fc1(x)
        x = self.relu(self.bn1(x))
        x = self.b_fc2(x)
        x = self.bn2(x)
        return x

    def set_grad(self):
        for layers, b_layers in zip(self.layers, self.b_layers):
            if b_layers.weight.grad is not None:
                layers.weight.grad = b_layers.weight.grad.clone()
            if b_layers.bias is not None and b_layers.bias.grad is not None:
                layers.bias.grad = b_layers.bias.grad.clone()

    def clipping(self):
        for layers in self.layers:
            layers.weight.data.clamp_(-1.0, 1.0)

    def update(self, optimizer, loss):
        optimizer.zero_grad()
        for b_layer in self.b_layers:
            if b_layer.weight.grad is not None:
                b_layer.weight.grad.zero_()
            if b_layer.bias is not None and b_layer.bias.grad is not None:
                b_layer.bias.grad.zero_()

        loss.backward()
        self.set_grad()
        optimizer.step()
        self.clipping()


def evaluate(model, loader, device):
    """
    正答率の計算(train_loader / test_loader どちらにも使える汎用版)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def plot(train_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', color='tab:red', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'o-', color='tab:green', label='Train Accuracy')
    plt.plot(epochs, test_accuracies, 's-', color='tab:blue', label='Test Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    os.makedirs('./output', exist_ok=True)
    plt.savefig('./output/binaryconnect_mnist_result.png')
    print("\nグラフを 'binaryconnect_mnist_result.png' として保存した．")

def main():
    torch.manual_seed(42)

    epochs = 10
    batch_size = 64
    learning_rate = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    model = BinaryConnectMnist().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    optimizer = optim.Adam(
        list(model.layers.parameters()) +
        list(model.bn1.parameters()) +
        list(model.bn2.parameters()),
        lr=learning_rate
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print("学習開始")
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Train Acc':>10} | {'Test Acc':>10} | {'Gap':>6}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            model.update(optimizer, loss)
            running_loss += loss.item() * data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # 訓練データ・テストデータ両方で精度を測る
        epoch_train_acc = evaluate(model, train_loader, device)
        epoch_test_acc = evaluate(model, test_loader, device)
        gap = epoch_train_acc - epoch_test_acc

        scheduler.step()

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_train_acc)
        test_accuracies.append(epoch_test_acc)

        print(f"{epoch:5d} | {epoch_loss:8.4f} | {epoch_train_acc:9.2f}% | {epoch_test_acc:9.2f}% | {gap:5.2f}pt")

    plot(train_losses, train_accuracies, test_accuracies)

if __name__ == '__main__':
    main()