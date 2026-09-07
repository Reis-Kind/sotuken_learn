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
        return torch.where(x >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        出力側の勾配をそのまま入力側に通過
        
        """
        return grad_output


class BinarizedNeuroEvo(nn.Module):
    """
    
    
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64, bias=True)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        # 重みを２値化
        w1 = BinarizeSTE.apply(self.fc1.weight)
        # ２値化した重みとバイアスを線型結合
        x = nn.functional.linear(x, w1, self.fc1.bias)
        x = BinarizeSTE.apply(x)
        w2 = BinarizeSTE.apply(self.fc2.weight)
        x = nn.functional.linear(x, w2, self.fc2.bias)
        return x

def evaluate(model, x, y):
    model.eval()
    output = model(x)
    predict = output.argmax(dim=1)
    acc = (predict == y).float().mean().item()

    return acc

def cross_entropy_loss(model, x, y):
    output = model(x)
    loss =  nn.functional.cross_entropy(output, y)

    return loss


def genetic_algorithm(model, x_batch, y_batch):
    """
    局所解を何とかするためのGA(突然変異とエリート選択)
    
    """
    n_candidate = 8
    mutation_rate = 0.02

    origin_loss = cross_entropy_loss(model, x_batch, y_batch).item()
    best_loss = origin_loss
    best_state = copy.deepcopy(model.state_dict())

    for i in range(n_candidate):
        candidate = copy.deepcopy(model.state_dict())
        for key in ['fc1.weight', 'fc2.weight']:
            w = candidate[key]
            mask = torch.rand_like(w) < mutation_rate
            w[mask] = -w[mask]

        model.load_state_dict(candidate)
        loss = cross_entropy_loss(model, x_batch, y_batch).item()
        if loss < best_loss:
            best_loss = loss
            best_state = copy.deepcopy(candidate)

    model.load_state_dict(best_state)
    improved = best_loss < origin_loss

    return best_loss, improved

def train_with_ga(model, train_loader, test_loader, device):
    n_epoch = 20
    lr = 0.001

    # 何エポック停滞したらGAを使うか
    ga_act = 2
    # 停滞カウント
    count = 0

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)

    # GA摂動の評価に使う固定バッチ(訓練データから少し多めに確保)
    help_x, help_y = next(iter(DataLoader(train_loader.dataset, batch_size=500, shuffle=True)))
    help_x, help_y = help_x.to(device), help_y.to(device)

    # 最小損失を記録する変数を無限で初期化
    best_loss_so_far = float('inf')
    rescue_events = []
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

            with torch.no_grad():
                model.fc1.weight.clamp_(-1.0, 1.0)
                model.fc2.weight.clamp_(-1.0, 1.0)

            running_loss += loss.item() * data.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)

        # 停滞検知
        # 前エポックまでの最小損失 best_loss_so_farから今回のepoch_lossを引いた低下量が0.001より大きいかを検証
        if best_loss_so_far - epoch_loss > 1e-3:
                best_loss_so_far = epoch_loss
                count = 0
        else:
            count += 1

        if count >= ga_act:
            help_loss, improved = genetic_algorithm(model, help_x, help_y)
            count = 0
            rescue_events.append(epoch)

            print(f"  → Epoch {epoch}: 停滞検知、GA摂動を実施 "
                  f"(改善={'あり' if improved else 'なし'}, "
                  f"rescue_loss={help_loss:.4f})")

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

    return train_losses, test_accuracies, rescue_events


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
    train_losses, test_accuracies, rescue_events = train_with_ga(model, train_loader, test_loader, device)

    # --- main 内でグラフの描画 ---
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 1. 訓練損失 (Loss) のプロット（左Y軸）
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    line1 = ax1.plot(epochs, train_losses, color=color, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. テスト精度 (Accuracy) のプロット（右Y軸）
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Test Accuracy (%)', color=color)
    line2 = ax2.plot(epochs, test_accuracies, color=color, marker='s', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # 3. GAレスキュー発生エポックに垂直線を表示
    if rescue_events:
        for i, ev in enumerate(rescue_events):
            ax1.axvline(x=ev, color='purple', linestyle=':', linewidth=2, 
                        label='GA Rescue Event' if i == 0 else "")

    # 凡例の集約表示
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Training Loss, Test Accuracy & GA Rescue Events')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()