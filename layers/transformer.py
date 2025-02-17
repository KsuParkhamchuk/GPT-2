import torch
from torch import nn
from .normalization import LayerNormalization
from hyperparams import EMBEDDING_DIM, CONTEXT_SIZE, ATTENTION_HEADS


class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiheadAttention()
        self.MLP = MLP()
        self.layer_norm = LayerNormalization()

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm(x))
        x = x + self.MLP(self.layer_norm(x))
        return x


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM // ATTENTION_HEADS)
        self.W_k = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM // ATTENTION_HEADS)
        self.W_v = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM // ATTENTION_HEADS)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attn_scores = Q @ K.T / torch.sqrt(EMBEDDING_DIM // ATTENTION_HEADS)
        # creates a mask with ones on an upper triangle and 0 on diagonal and below, then convert to bool
        mask = torch.triu(
            torch.ones(attn_scores.size(-2), attn_scores.size(-1)), diagonal=1
        ).bool()
        # replaces positions where mask=True with -inf
        masked_scores = attn_scores.masked_fill(mask, float("-inf"))
        # compute softmax with last dimension
        attn_weights = torch.softmax(masked_scores, dim=-1)
        output = attn_weights @ V
        return output


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        heads_outputs = []
        for _ in range(ATTENTION_HEADS):
            heads_outputs.append(Head().forward(x))

        attn_res = torch.cat(heads_outputs, dim=-1)
        return attn_res


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(CONTEXT_SIZE, 4 * EMBEDDING_DIM)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(CONTEXT_SIZE, EMBEDDING_DIM)

    def forward(self, x):
        fc_1 = self.fc_1(x)
        gelu_output = self.gelu(fc_1)
        proj_output = self.proj(gelu_output)
        return proj_output
