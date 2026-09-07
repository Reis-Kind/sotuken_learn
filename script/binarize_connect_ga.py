import numpy as np
import torch 
import torch.nn as nn

class BainarizeSTE(torch.autograd.Function):
    """
    順方向の計算では値を 1.0 と -1.0 に制限，逆方向の計算では勾配をそのまま通すことで，
    微分不可能な関数を含むネットワークであっても通常の勾配降下法で学習できる
    
    """

    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
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
        x = self.fc1(x)
        x = torch.where(x >= 0, 1.0, -1.0)
        # 出力層は二値化しない
        x = self.fc2(x)
        return x
    