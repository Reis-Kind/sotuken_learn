import os
import copy
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class BinarizeSTE(torch.autograd.Function):
    """
    順方向の計算では値を 1.0 と -1.0 に制限，逆方向の計算では勾配をそのまま通すことで，
    微分不可能な関数を含むネットワークであっても通常の勾配降下法で学習できる
    
    """

    @staticmethod
    def forward(ctx, x):
        """
        入力xの各要素が 0 以上であれば 1.0、0未満であれば -1.0 に変換
        
        """
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        二値化処理は本来微分できないため、
        そのままでは勾配を計算できない。
        そこでSTEを用いて、一定の範囲では
        出力側の勾配をそのまま入力側へ伝える。

        ただし、入力値の絶対値が 1.0 を超えている場合は
        勾配を 0 とする。
        これにより、実数重みが二値化の範囲から大きく外れた場合に
        更新され続けることを防ぐ。
        
        """
        # forward() で保存しておいた入力値を取得
        x, = ctx.saved_tensors
        # 出力側から伝わってきた勾配をコピー
        grad = grad_output.clone()
        # |x| > 1.0 の領域では勾配を0にして更新を止める
        grad[x.abs() > 1.0] = 0.0

        return grad


class BinarizedNeuroEvo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)
        self.bn1 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.view(-1, 784)
        w1 = BinarizeSTE.apply(self.fc1.weight)
        x = nn.functional.linear(x, w1, self.fc1.bias)
        x = self.bn1(x)
        x = BinarizeSTE.apply(x)
        w2 = BinarizeSTE.apply(self.fc2.weight)
        x = nn.functional.linear(x, w2, self.fc2.bias)
        return x

    def clipping(self):
        self.fc1.weight.data.clamp_(-1.0, 1.0)
        self.fc2.weight.data.clamp_(-1.0, 1.0)


def cross_entropy_loss(model, x, y):
    output = model(x)
    loss = nn.functional.cross_entropy(output, y)
    return loss


def train_plain(model, train_loader, test_loader, device):
    """
    GA摂動なしの、通常のBinaryConnect学習(比較用ベースライン)
    """
    n_epoch = 20
    lr = 0.0005

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

    train_losses = []
    test_accuracies = []

    for epoch in range(1, n_epoch + 1):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = cross_entropy_loss(model, data, target)
            loss.backward()
            optimizer.step()
            model.clipping()

            running_loss += loss.item() * data.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                predict = model(data).argmax(dim=1)
                correct += (predict == target).sum().item()
                total += target.size(0)

        acc = 100.0 * correct / total

        train_losses.append(epoch_loss)
        test_accuracies.append(acc)
        print(f"Epoch [{epoch}/{n_epoch}] Loss: {epoch_loss:.4f} | Test Acc: {acc:.2f}%")

    return train_losses, test_accuracies


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")

    model = BinarizedNeuroEvo()
    train_losses, test_accuracies = train_plain(model, train_loader, test_loader, device)

    # --- グラフ描画 ---
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(11, 6))

    color1 = 'tab:red'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', color=color1, fontsize=12)
    line1 = ax1.plot(epochs, train_losses, color=color1, marker='o',
                      markersize=5, linewidth=1.8, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax1.minorticks_on()
    ax1.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.6)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Test Accuracy (%)', color=color2, fontsize=12)
    line2 = ax2.plot(epochs, test_accuracies, color=color2, marker='s',
                      markersize=5, linewidth=1.8, label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 100)

    plt.title('Training Loss & Test Accuracy (No GA Rescue)', fontsize=14, pad=12)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.9)

    ax1.set_xticks(list(epochs))

    fig.tight_layout()
    os.makedirs('./output', exist_ok=True)
    plt.savefig('./output/plain_result.png', dpi=150)
    print("\nグラフを ./output/plain_result.png に保存した")


if __name__ == '__main__':
    main()