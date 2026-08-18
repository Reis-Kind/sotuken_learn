import torch
import torch.nn as nn

class BinaryConnectMnist(nn.Module):
    def __init__(self):
        super().__init__()

        # 実数の重み
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.layers = [self.fc1, self.fc2]

        # 二値化の重み
        self.b_fc1 = nn.Linear(784, 128)
        self.b_fc2 = nn.Linear(128, 10)
        self.b_layers = [self.b_fc1, self.b_fc2]

        self.relu = nn.ReLU()


    def binarize(self):
        """
        実数の重みを二値化

        """
        for layers, b_layers in zip(self.layers, self.b_layers):
            b_layers.weight.data = torch.sign(layers.weight.data)


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

    def clipping(self):
        """
        実数の重みが大きくなりすぎて，二値化重み（+1,-1のみなので）が変化しにくくなるのを防ぐ
        為に実数重みを-1.0 ~ 1.0に制限

        """
        for layers in self.layers:
            layers.weight.data.clam_(-1.0, 1.0)

    def update(self, optimizer, loss):
        """
        一回分の学習更新を一括処理する関数

        """
        optimizer.zero_grad()
        loss.backward()
        self.set_grad()
        optimizer.step()
        self.clipping()

        






