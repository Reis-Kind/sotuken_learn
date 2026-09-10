import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class BinaryConnectCifar10(nn.Module):
    def __init__(self):
        super().__init__()

        # 実数の重み
        # RGBで3チャネルあるから3（Mnistは白黒だから1）
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2,  bias=False)
        self.layers = nn.ModuleList([self.conv1, self.conv2])

        # 二値化の重み
        self.b_conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False)
        self.b_conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2,  bias=False)
        self.b_layers = nn.ModuleList([self.b_conv1, self.b_conv2])

        # 神の一手
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # padding=2だから32*32→16*16→8*8
        self.fc = nn.Linear(32 * 8 * 8, 10)

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


def genetic_algorithm(
    model, x_batch, y_batch, check_x, check_y, device
):
    """
    GA選択用500枚と、その回の選択に使わない4500枚で、
    GA前後の正答率を確認する。
    """
    n_candidate = 8
    mutation_rate = 0.001

    model.eval()

    # GA開始前の二値重みを保存
    origin_signs = [
        (layer.weight.detach() >= 0).clone()
        for layer in model.layers
    ]

    def count_changed_weights():
        # マスクの個数ではなく、実際に符号が変わった重みを数える
        return [
            ((layer.weight.detach() >= 0) != origin).sum().item()
            for layer, origin in zip(model.layers, origin_signs)
        ]

    # 同じ画像を使って、GA前後を比較する
    origin_acc = evaluate(model, x_batch, y_batch, device)
    check_before = evaluate(model, check_x, check_y, device)

    best_acc = origin_acc
    origin_state = copy.deepcopy(model.state_dict())
    best_state = copy.deepcopy(origin_state)
    best_candidate = 0

    print(
        f"    GA前: 選択用500枚={origin_acc:.2f}% | "
        f"確認用4500枚={check_before:.2f}%"
    )

    for i in range(n_candidate):
        # 毎回、GA開始時のモデル(origin_state)から独立した候補を作る
        candidate = copy.deepcopy(origin_state)
        mutation_count = 0

        for key in ['conv1.weight', 'conv2.weight']:
            w = candidate[key]
            mask = torch.rand_like(w) < mutation_rate
            mutation_count += mask.sum().item()
            w[mask] = -w[mask]

        model.load_state_dict(candidate)
        acc = evaluate(model, x_batch, y_batch, device)

        changed = count_changed_weights()

        print(
            f"    候補{i + 1}: 今回の反転指定={mutation_count}個 | "
            f"GA開始時からの符号変化={sum(changed)}個 | "
            f"選択用正答率={acc:.2f}% "
            f"({acc - origin_acc:+.2f}ポイント)"
        )

        # 候補の選択には500枚の正答率だけを使う
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            best_candidate = i + 1

    model.load_state_dict(best_state)

    # 採用・復元後の状態を実際に再評価
    selected_acc = evaluate(model, x_batch, y_batch, device)
    check_after = evaluate(model, check_x, check_y, device)
    changed = count_changed_weights()

    selected_name = (
        "元のモデル" if best_candidate == 0
        else f"候補{best_candidate}"
    )

    print(f"    選択結果: {selected_name}")
    print(
        f"    選択用500枚: {origin_acc:.2f}% → {selected_acc:.2f}% "
        f"({selected_acc - origin_acc:+.2f}ポイント)"
    )
    print(
        f"    確認用4500枚: {check_before:.2f}% → {check_after:.2f}% "
        f"({check_after - check_before:+.2f}ポイント)"
    )
    print(
        f"    採用後の符号変化: "
        f"conv1={changed[0]}個, conv2={changed[1]}個, "
        f"合計={sum(changed)}個"
    )

    improved = selected_acc > origin_acc
    return selected_acc, improved



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
    plt.savefig('./output/binaryconnect_cifar_result.png')
    print("\nグラフを 'binaryconnect_cifar_result.png' として保存した．")

def main():

    torch.manual_seed(42)

    epochs = 200
    batch_size = 64
    learning_rate = 0.001
    ga_act=4
    use_ga=True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイス判定
    print(f"使用デバイス: {device}")
    model = BinaryConnectCifar10().to(device) # モデルを GPU へ転送

    # データセットの準備（正規化も）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # --- 訓練用とGA検証用に分割 ---
    # 訓練: 45000枚、GA検証用: 5000枚(学習には一切使わない)
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = 45000
    val_size = len(full_train_dataset) - train_size
    train_dataset, ga_val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42) )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # GA検証用データをテンソルにまとめておく(毎回同じプールからサンプリングする)
    ga_val_x = torch.stack([ga_val_dataset[i][0] for i in range(len(ga_val_dataset))])
    ga_val_y = torch.tensor([ga_val_dataset[i][1] for i in range(len(ga_val_dataset))])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

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
                # 同じ5000枚を、選択用500枚と確認用4500枚に分ける
                perm = torch.randperm(len(ga_val_x))
                select_indices = perm[:500]
                check_indices = perm[500:]

                help_x = ga_val_x[select_indices]
                help_y = ga_val_y[select_indices]

                check_x = ga_val_x[check_indices]
                check_y = ga_val_y[check_indices]

                print(f"  → Epoch {epoch}: 停滞検知、GA開始")

                help_acc, improved = genetic_algorithm(
                    model,
                    help_x, help_y,
                    check_x, check_y,
                    device
                )

                count = 0
                help_events.append(epoch)

                print(
                    f"  → Epoch {epoch}: GA終了 "
                    f"(改善={'あり' if improved else 'なし'}, "
                    f"rescue_acc={help_acc:.2f}%)"
                )
            

        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.2f}%")

    # グラフ描画実行
    plot(train_losses, test_accuracies)

if __name__ == '__main__':
    main()






