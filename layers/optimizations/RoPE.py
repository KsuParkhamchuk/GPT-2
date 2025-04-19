from torch import nn
import torch
from layers.linear import Linear
from model import EMBEDDING_DIM, HEAD_DIM, CONTEXT_SIZE


class ROPEAttnHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = Linear(EMBEDDING_DIM, HEAD_DIM)
        self.Wk = Linear(EMBEDDING_DIM, HEAD_DIM)
        self.Wv = Linear(EMBEDDING_DIM, HEAD_DIM)

        # 1/ theta = 10000 ** (2i/d)
        # Shape: (HEAD_DIM // 2)
        freqs = 1 / (10000 ** torch.arange(0, HEAD_DIM, 2) / HEAD_DIM)

        # Shape: (CONTEXT_SIZE)
        positions = torch.arange(CONTEXT_SIZE)
        # (CONTEXT_SIZE, HEAD_DIM // 2) = (CONTEXT_SIZE, 1) * (1, HEAD_DIM // 2)
        angles = positions[:, None] * freqs[None, :]

        # Shape: (CONTEXT_SIZE, HEAD_DIM // 2)
        # Register buffers are automatically moved to another device when the model is moved to another device
        self.register_buffer("cos_table", torch.cos(angles))
        self.register_buffer("sin_table", torch.sin(angles))

    def rope_rotation(self, x):
        """
        RoPE injection into the standard attention mechanism

        Computational efficient realization of rotary matrix multiplication according the the original paper
        """
        b_size, seq_len, _ = x.shape

        # Shape: (BATCH_SIZE, CONTEXT_SIZE, HEAD_DIM)
        Q = self.Wq(x)
        K = self.Wk(x)

        # Reshaping Q and K to meet the shape of the cos and sin tables
        # Shape: (BATCH_SIZE, CONTEXT_SIZE, HEAD_DIM // 2, 2) - form pairs throughout HEAD_DIM
        Q_reshaped = Q.view(b_size, seq_len, HEAD_DIM // 2, 2)
        K_reshaped = K.view(b_size, seq_len, HEAD_DIM // 2, 2)

        # 90 degree rotation (x, y) -> (-y, x)
        #  -Q_reshaped[..., 1] = -y (second element of the last dimension)
        #  Q_reshaped[..., 0] = x (first element of the last dimension)
        Q_rotated = torch.stack(
            (
                -Q_reshaped[..., 1],
                Q_reshaped[..., 0],
            ),
            dim=-1,
        )
        K_rotated = torch.stack(
            (
                -K_reshaped[..., 1],
                K_reshaped[..., 0],
            ),
            dim=-1,
        )

        # RoPE implementation
        Q_RoPE = Q_reshaped * self.cos_table + Q_rotated * self.sin_table
        K_RoPE = K_reshaped * self.cos_table + K_rotated * self.sin_table

        # Reshape back to the original shape
        Q = Q_RoPE.view(b_size, seq_len, HEAD_DIM)
        K = K_RoPE.view(b_size, seq_len, HEAD_DIM)

        return Q, K

    def forward(self, x):
        """
        Forward pass with rotated Q and K
        """
        Q, K = self.rope_rotation(x)
        V = self.Wv(x)

        # Standard attention mechanism
        attn_scores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(HEAD_DIM))

        # create a mask with ones on an upper triangle and 0 on diagonal and below, then convert to bool
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
