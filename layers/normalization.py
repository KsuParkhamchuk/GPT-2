import torch
from torch import nn
from model.hyperparams import EMBEDDING_DIM


class NormalizationLayer(nn.Module):
    def __init__(self, dim=EMBEDDING_DIM, eps=1e-5):
        super().__init__()
        # Initialize gamma to ones and beta to zeros
        # beta, gamma params shape [EMBEDDING_DIM]
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    # x- mean - substructing the mean centers the distribution around zero
    # var - variance - how much our data varies from the average
    # mean and variance are calculated per token under the hood of mean and var methods
    # normalization is applied element-wise
    def forward(self, x):
        # shape [BATCH_SIZE, SEQUENCE_LENGTH, 1]
        mean = x.mean(dim=-1, keepdim=True)
        # shape [BATCH_SIZE, SEQUENCE_LENGTH, 1]
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

    def __call__(self, x):
        return self.forward(x)
