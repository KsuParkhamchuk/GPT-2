import torch
from torch import nn
from .normalization import NormalizationLayer
from hyperparams import EMBEDDING_DIM, ATTENTION_HEADS, HEAD_DIM
from .linear import Linear


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiheadAttention()
        self.MLP = MLP()
        self.attn_layer_norm = NormalizationLayer()
        self.mlp_layer_norm = NormalizationLayer()

    def forward(self, x):
        x = x + self.self_attention(self.attn_layer_norm(x))
        x = x + self.MLP(self.mlp_layer_norm(x))
        return x

    def __call__(self, x):
        return self.forward(x)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = Linear(EMBEDDING_DIM, HEAD_DIM)
        self.W_k = Linear(EMBEDDING_DIM, HEAD_DIM)
        self.W_v = Linear(EMBEDDING_DIM, HEAD_DIM)

        # helps prevent attention scores from being too extreme initially
        self.W_q.weights.data.normal_(mean=0.0, std=0.02)
        self.W_k.weights.data.normal_(mean=0.0, std=0.02)
        self.W_v.weights.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        # [BATCH_SIZE, CONTEXT_SIZE, HEAD_DIM]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # transpose(-2, -1) swaps sequence length dimentions but preserve batch
        # [BATCH_SIZE, CONTEXT_SIZE, CONTEXT_SIZE]
        attn_scores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(HEAD_DIM))
        # creates a mask with ones on an upper triangle and 0 on diagonal and below, then convert to bool
        mask = torch.triu(
            torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=x.device),
            diagonal=1,
        ).bool()
        # replaces positions where mask=True with -inf
        masked_scores = attn_scores.masked_fill(mask, float("-inf"))
        # compute softmax with last dimension
        attn_weights = torch.softmax(masked_scores, dim=-1)
        # [Batch_SIZE, CONTEXT_SIZE, HEAD_DIM]
        output = attn_weights @ V
        return output


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(ATTENTION_HEADS)])
        self.proj_weights = Linear(EMBEDDING_DIM, EMBEDDING_DIM)

        # Initialize attention projection matrices with small values
        # This helps prevent attention scores from being too extreme initially
        self.proj_weights.weights.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        heads_outputs = []
        for head in self.heads:
            heads_outputs.append(head.forward(x))

        attn_res = torch.cat(heads_outputs, dim=-1)
        attn_res = self.proj_weights(attn_res)
        return attn_res


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = Linear(EMBEDDING_DIM, 4 * EMBEDDING_DIM)
        self.gelu = nn.GELU()
        self.proj = Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM)

        # Initialize with appropriate scaling
        self.fc_1.weights.data.normal_(mean=0.0, std=0.02)
        self.proj.weights.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        fc_1 = self.fc_1(x)
        gelu_output = self.gelu(fc_1)
        proj_output = self.proj(gelu_output)
        return proj_output
