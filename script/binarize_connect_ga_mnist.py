import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class BinaryConnectMnist(nn.Module):
    def __init__(self):
        super().__init__()

        # 実数の重み
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2,  bias=False)
        self.layers = nn.ModuleList([self.conv1, self.conv2])

        # 二値化の重み
        self.b_conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2, bias=False)
        self.b_conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2,  bias=False)
        self.b_layers = nn.ModuleList([self.b_conv1, self.b_conv2])

        # 神の一手
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(32 * 7 * 7, 10)

    def binarize(self):
        """
        実数の重みを二値化

        """

        for layers, b_layers in zip(self.layers, self.b_layers):
            # torch.signでもいいけど、0 のときに 0 を返してしまい，重みが +1 でも -1 でもなくなってしまう_
            b_layers.weight.data = torch.where(layers.weight.data >= 0, 1.0, -1.0)

            # 実数層 (layers) のバイアスを 二値化層 (b_layer) にそのまま複製
            if layers.bias is not None and b_layers.bias is not None:
                b_layers.bias.data = layers.bias.data.clone()


    def forward(self, x):
        """
        順伝播：二値化された重みで計算
        予測値をだす？誤差から逆算する前に一旦今の結果を見る感じ

        """

        self.binarize()

        x = self.b_conv1(x)
        x = self.relu(self.bn1(x))
        x = self.pool(x)

        x = self.b_conv2(x)
        x = self.relu(self.bn2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_grad(self):
        """
        勾配を実数層にコピー

        """
        for layers, b_layers in zip(self.layers, self.b_layers):
            if b_layers.weight.grad is not None:
                layers.weight.grad = b_layers.weight.grad.clone()
            if b_layers.bias is not None and b_layers.bias.grad is not None:
                layers.bias.grad = b_layers.bias.grad.clone()

    def clipping(self):
        """
        実数の重みが大きくなりすぎて，二値化重み（+1,-1のみなので）が変化しにくくなるのを防ぐ
        為に実数重みを-1.0 ~ 1.0に制限

        """
        for layers in self.layers:
            layers.weight.data.clamp_(-1.0, 1.0)

    def update(self, optimizer, loss):
        """
        一回分の学習更新を一括処理する関数

        """
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


def evaluate(model, x, y, device):
    """
    正答率の計算
    
    """
    model.eval()

    x = x.to(device)
    y = y.to(device)

    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1)

        correct = (pred == y).sum().item()
        total = y.size(0)

    accuracy = 100.0 * correct / total

    return accuracy


def cross_entropy_loss(model, x, y):
    output = model(x)
    loss =  nn.functional.cross_entropy(output, y)

    return loss


def genetic_algorithm(model, x_batch, y_batch, device):
    """
    局所解を何とかするためのGA(突然変異とエリート選択)
    
    """
    n_candidate = 8
    mutation_rate = 0.001

    model.eval()

    # 現状のモデルでの正答率を計算
    origin_acc = evaluate(model, x_batch, y_batch, device)
    best_acc = origin_acc
    best_state = copy.deepcopy(model.state_dict())

    # 複数の変異候補を作成して評価
    for i in range(n_candidate):
        candidate = copy.deepcopy(model.state_dict())
        for key in ['conv1.weight', 'conv2.weight']:
            w = candidate[key]
            mask = torch.rand_like(w) < mutation_rate
            w[mask] = -w[mask]

        # 変異モデルの正答率を計算して評価
        model.load_state_dict(candidate)
        acc = evaluate(model, x_batch, y_batch, device)
        # より良い正答率が得られた場合は最適状態を更新
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(candidate)

    # 最も優れていた状態をモデルに反映
    model.load_state_dict(best_state)
    improved = best_acc > origin_acc

    return best_acc, improved



def plot(train_losses, test_accuracies):
    """
    グラフ描画

    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Lossのグラフ
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', color='tab:red', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Accuracyのグラフ
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, 'o-', color='tab:blue', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    os.makedirs('./output', exist_ok=True) # フォルダがなければ作成
    plt.savefig('./output/binaryconnect_mnist_result.png')
    print("\nグラフを 'binaryconnect_mnist_result.png' として保存した．")

def main():

    torch.manual_seed(42)

    epochs = 10
    batch_size = 64
    learning_rate = 0.001
    ga_act=4
    use_ga=False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイス判定
    print(f"使用デバイス: {device}")
    model = BinaryConnectMnist().to(device) # モデルを GPU へ転送

    # データセットの準備（正規化も）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # --- 訓練用とGA検証用に分割 ---
    # 訓練: 55000枚、GA検証用: 5000枚(学習には一切使わない)
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = 55000
    val_size = len(full_train_dataset) - train_size
    train_dataset, ga_val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42) )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # GA検証用データをテンソルにまとめておく(毎回同じプールからサンプリングする)
    ga_val_x = torch.stack([ga_val_dataset[i][0] for i in range(len(ga_val_dataset))])
    ga_val_y = torch.tensor([ga_val_dataset[i][1] for i in range(len(ga_val_dataset))])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # テストデータをTensorにまとめる
    test_x = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_y = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    optimizer = optim.Adam(
        list(model.layers.parameters()) +
        list(model.bn1.parameters()) +
        list(model.bn2.parameters()) +
        list(model.fc.parameters()),
        lr=learning_rate
    )

    # 学習率をcos関数の波形に沿って徐々に小さくしていく
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc_so_far = 0.0
    count = 0
    help_events = []
    train_losses = []
    test_accuracies = []

    print("学習開始")

    for epoch in range(1, epochs + 1):
        # モデルを学習モードに
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device) # データを GPU へ転送
            loss = cross_entropy_loss(model, data, target)
            model.update(optimizer, loss)
            # そのエポック全体のloss
            running_loss += loss.item() * data.size(0)
        # 平均loss
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = evaluate(model, test_x, test_y, device)

        scheduler.step()

        train_losses.append(epoch_loss)
        test_accuracies.append(epoch_acc)

        if use_ga:

            # 停滞検知
            # 前エポックまでの最小損失 best_loss_so_farから今回のepoch_lossを引いた低下量が0.001より大きいかを検証
            if epoch_acc - best_acc_so_far > 0.01:
                best_acc_so_far = epoch_acc
                count = 0
            else:
                count += 1

            if count >= ga_act:

                # GA検証用プール(訓練にも評価にも使っていない5000枚)から500枚サンプリング
                perm = torch.randperm(len(ga_val_x))[:500]
                help_x, help_y = ga_val_x[perm], ga_val_y[perm]

                help_acc, improved = genetic_algorithm(model, help_x, help_y, device)
                count = 0
                help_events.append(epoch)
            
                print(f"  → Epoch {epoch}: 停滞検知、GA摂動を実施 "
                      f"(改善={'あり' if improved else 'なし'}, "
                      f"rescue_acc={help_acc:.2f}%)")
            

        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.2f}%")

    # グラフ描画実行
    plot(train_losses, test_accuracies)

if __name__ == '__main__':
    main()






