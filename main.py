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

        


