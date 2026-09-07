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
    