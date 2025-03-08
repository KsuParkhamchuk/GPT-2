import torch
from torch import nn
from hyperparams import EMBEDDING_DIM


class LayerNormalization(nn.Module):
    def __init__(self, dim=EMBEDDING_DIM, eps=1e-5):
        super().__init__()
        # Initialize gamma to ones and beta to zeros
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

    def __call__(self, x):
        return self.forward(x)
