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
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.layers = nn.ModuleList([self.fc1, self.fc2])

        # 二値化の重み
        self.b_fc1 = nn.Linear(784, 128)
        self.b_fc2 = nn.Linear(128, 10)
        self.b_layers = nn.ModuleList([self.b_fc1, self.b_fc2])

        self.relu = nn.ReLU()


    def binarize(self):
        """
        実数の重みを二値化

        """
        # 一時的に勾配の追跡オフ
        with torch.no_grad():

            for layers, b_layers in zip(self.layers, self.b_layers):
                # torch.signでもいいけど、0 のときに 0 を返してしまい，重みが +1 でも -1 でもなくなってしまう_
                b_layers.weight.copy_(torch.where(layers.weight.data >= 0, 1.0, -1.0))

                # 実数層 (layers) のバイアスを 二値化層 (b_layer) にそのまま複製
                if layers.bias is not None:
                    b_layers.bias.data = layers.bias.data.clone()


    def forward(self, x):
        """
        順伝播：二値化された重みで計算
        予測値をだす？誤差から逆算する前に一旦今の結果を見る感じ

        """
        # 全結合層に入力するために一次元に
        x = x.view(x.size(0), -1)

        self.binarize()
        x = self.relu(self.b_fc1(x))
        # 0 ~ 9を判別させたいのでこの層には非線形関数はいれない
        x = self.b_fc2(x)

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


def evaluate(model, test_loader, device):
    """
    正答率の計算
    
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


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
    epochs = 10
    batch_size = 64
    learning_rate = 0.0005

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイス判定
    print(f"使用デバイス: {device}")
    model = BinaryConnectMnist().to(device) # モデルを GPU へ転送

    # データセットの準備
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    optimizer = optim.Adam(model.layers.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


    train_losses = []
    test_accuracies = []

    print("学習開始")

    for epoch in range(1, epochs + 1):
        # モデルを学習モードに
        model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device) # データを GPU へ転送
            output = model(data)
            loss = criterion(output, target)
            model.update(optimizer, loss)
            # そのエポック全体のloss
            running_loss += loss.item() * data.size(0)
        # 平均loss
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = evaluate(model, test_loader, device)

        train_losses.append(epoch_loss)
        test_accuracies.append(epoch_acc)

        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.2f}%")

    # グラフ描画実行
    plot(train_losses, test_accuracies)

if __name__ == '__main__':
    main()






