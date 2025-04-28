import torch
from torch import nn
from .linear import Linear
from model.hyperparams import EMBEDDING_DIM, HEAD_DIM


class KVCacheOptimizedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_q = Linear(EMBEDDING_DIM, HEAD_DIM)
        self.W_k = Linear(EMBEDDING_DIM, HEAD_DIM)
        self.W_v = Linear(EMBEDDING_DIM, HEAD_DIM)

        # helps prevent attention scores from being too extreme initially
        self.W_q.weights.data.normal_(mean=0.0, std=0.02)
        self.W_k.weights.data.normal_(mean=0.0, std=0.02)
        self.W_v.weights.data.normal_(mean=0.0, std=0.02)

        # Store cached keys and values
        self.cache_k = None
        self.cache_v = None
        self.cached_seq_len = 0

    def forward(self, x):

        # [BATCH_SIZE, SEQ_LEN, HEAD_DIM]
        Q = self.W_q(x)
        seq_len = x.size(1)

        if self.cache_k is not None:

            if seq_len == 1:
                # [BATCH_SIZE, 1, HEAD_DIM]
                new_k = self.W_k(x)
                new_v = self.W_v(x)
            else:
                # [BATCH_SIZE, NEW_TOKENS, HEAD_DIM]
                new_tokens = x[:, self.cached_seq_len :, :]
                new_k = self.W_k(new_tokens)
                new_v = self.W_v(new_tokens)

            # Concatenating existing and new keys and values
            K = torch.cat([self.cache_k, new_k], dim=1)
            V = torch.cat([self.cache_v, new_v], dim=1)

            # Update cache
            self.cache_k = K
            self.cache_v = V
            self.cached_seq_len = K.size(1)
        else:
            # [BATCH_SIZE, SEQ_LEN, HEAD_DIM]
            K = self.W_k(x)
            V = self.W_v(x)
            self.cache_k = K
            self.cache_v = V
            self.cached_seq_len = K.size(1)

        # transpose(-2, -1) swaps sequence length dimentions but preserve batch
        # [BATCH_SIZE, SEQ_LEN, SEQ_LEN]
        attn_scores = (
            Q @ self.cache_k.transpose(-2, -1) / torch.sqrt(torch.tensor(HEAD_DIM))
        )
        # creates a mask with ones on an upper triangle and 0 on diagonal and below, then convert to bool
        mask = torch.triu(
            torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=x.device),
            diagonal=1,
        ).bool()
        # replaces positions where mask=True with -inf
        masked_scores = attn_scores.masked_fill(mask, float("-inf"))
        # compute softmax with last dimension
        attn_weights = torch.softmax(masked_scores, dim=-1)
        # [BATCH_SIZE, SEQ_LEN, HEAD_DIM]
        output = attn_weights @ V
        return output
