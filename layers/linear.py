from torch import nn
import torch
import math


class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        # Use Kaiming initialization for weights
        std = math.sqrt(2.0 / input_dim)
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * std)

        # Initialize bias to zeros
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            # register the parameter for pytorch to know it exists
            self.register_parameter("bias", None)

    def forward(self, x):
        # y = xA^T + b
        output = x @ self.weights
        if self.bias is not None:
            output = output + self.bias
        return output
