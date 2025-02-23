from torch import nn
import torch


class Linear(nn.Module):
    def __init__(self, input, output, bias=True):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input, output))
        self.bias = nn.Parameter(torch.randn(output)) if bias else torch.zeros(output)

    def forward(self, x):
        # y = xA^T + b
        # matrix multiplication preserves all dimentions except the last one
        return x @ self.weights + self.bias
