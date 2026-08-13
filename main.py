import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 先生作成のライブラリをインポート
import nn_tools


# ==========================================
# 1. 先生の設計による BinaryConnect クラス
# ==========================================
class BinaryConnect(nn.Module):
    def __init__(self, input_ch=1):
        super(BinaryConnect, self).__init__()

        # BinaryConnectのネットワーク
        _b_order = {
            'conv1': 8,
            'bn2d1': None,
            'relu1': None,
            'conv2': 8,
            'bn2d2': None,
            'relu2': None,
            'max_pool2': 2,
            'conv3': 16,
            'bn2d3': None,
            'relu3': None,
            'conv4': 16,
            'bn2d4': None,
            'relu4': None,
            'max_pool4': 2,
            'global_avg_pool': None,
            'view': None,
            'fc1': 10
        }

        # 浮動小数点の重みを保存しておくネットワークの構成を初期化
        _fp_order = {}
        # 2値化する層の番号を格納するリストを初期化
        self.binarize_layer_number_list = []
        # 2値化する層を検索するループ
        for i, _item in enumerate(_b_order.items()):
            _key, _value = _item
            # 畳み込み層か全結合層の場合
            if 'conv' in _key or 'fc' in _key:
                # 2値化する層の番号をリストに追加
                self.binarize_layer_number_list.append(i)
                # 2値化する層の番号をリストに追加
                _fp_order[_key] = _value

        self.b_layers = nn_tools.generate_model(_b_order, input_ch, bias=False)
        self.fp_layers = nn_tools.generate_model(_fp_order, input_ch, bias=False)

    def forward(self, x):
        """順伝播."""
        self.binarize()
        x = self.b_layers(x)
        return x

    def binarize(self):
        """層の2値化を行う関数."""
        for i, j in enumerate(self.binarize_layer_number_list):
            self.b_layers[j].weight.data = torch.sign(self.fp_layers[i].weight.data)

    def set_grad(self):
        """2値化した畳み込み層の勾配を実数の勾配へコピーする関数."""
        for i, j in enumerate(self.binarize_layer_number_list):
            if self.b_layers[j].weight.grad is not None:
                self.fp_layers[i].weight.grad = self.b_layers[j].weight.grad.clone()

    def clipping(self):
        """重みのクリッピングを行う関数."""
        for i in range(len(self.fp_layers)):
            self.fp_layers[i].weight.data.clamp_(-1, 1)

    def update(self, optimizer, loss):
        """重みの更新を行う関数."""
        # 勾配をリセット
        optimizer.zero_grad()
        # 誤差逆伝播
        loss.backward()
        # 勾配をセット
        self.set_grad()
        # 重みの更新
        optimizer.step()
        # 重みのclipping
        self.clipping()

    def get_binarize_weight_list(self):
        """2値化した各層の重みをリストで取得する関数."""
        weight_list = []
        # 重みを取得するループ
        for i in self.binarize_layer_number_list:
            weight_list.append(self.b_layers[i].weight.data)
        return weight_list

    def get_weight_map(self):
        """各層の重みの分布を取得する関数."""
        # 二値化
        self.binarize()
        # 各層の重みをリストで取得
        weight_list = self.get_binarize_weight_list()

        weight_map = [[0, 0] for i in range(len(weight_list))]
        # 各層の重みを取り出すループ
        for i, w in enumerate(weight_list):
            for j, n in enumerate([-1, 1]):
                weight_map[i][j] += (w == n).sum().item()

        return weight_map


# ==========================================
# 2. 実行用メイン処理
# ==========================================
def main():
    # 実行デバイス（GPUが使えるならGPUを優先）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用中のデバイス: {device}")

    # MNISTデータのダウンロード・前処理の設定
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # モデルのインスタンス化
    model = BinaryConnect(input_ch=1).to(device)

    # 【重要】オプティマイザの登録パラメータ
    # 2値化しない BatchNorm などのパラメータも学習に含めるために
    # model.fp_layers と、model.b_layersから重みを除いたものを別々に登録する
    optimizer = optim.Adam([
        {'params': model.fp_layers.parameters()},
        {'params': [p for name, p in model.b_layers.named_parameters() if 'weight' not in name]}
    ], lr=0.01)

    criterion = nn.CrossEntropyLoss()

    print("学習を開始します．")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 順伝播
        output = model(data)
        
        # 誤差（損失）の計算
        loss = criterion(output, target)

        # 逆伝播・勾配コピー・実数更新・クリッピングを一括実行
        model.update(optimizer, loss)

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    # 学習後の各層における 2値（-1, 1）の分布を出力
    weight_map = model.get_weight_map()
    print("\n--- 各層の2値化重み分布 [-1の数, 1の数] ---")
    for layer_idx, counts in enumerate(weight_map):
        print(f"レイヤー {layer_idx}: -1の個数 = {counts[0]}, 1の個数 = {counts[1]}")


if __name__ == '__main__':
    main()