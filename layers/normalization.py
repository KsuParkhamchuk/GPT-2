import torch
from hyperparams import EMBEDDING_DIM


class LayerNormalization:
    def __init__(self):
        self.b = torch.zeros(EMBEDDING_DIM)
        self.y = torch.ones(EMBEDDING_DIM)

        self.row_mean = lambda row: torch.mean(row)
        self.st_d = lambda row, m: torch.sqrt(
            sum([(x - m) ** 2 for x in row]) / EMBEDDING_DIM
        )  # might be torch.std(row)
        self.row_norm = lambda row, m, st_d: (row - m) / st_d
        self.row_scaled_and_shifted = lambda row_norm: row_norm * self.y + self.b

    def forward(self, x):
        normalized_output = torch.zeros_like(x)

        for i in range(x.size(0)):
            row = x[i]
            row_mean = self.row_mean(row)
            row_st_d = self.st_d(row, row_mean)
            row_norm = self.row_norm(row, row_mean, row_st_d)
            row_scaled_and_shifted = self.row_scaled_and_shifted(row_norm)
            normalized_output[i] = row_scaled_and_shifted

        return normalized_output

    def __call__(self, x):
        return self.forward(x)
