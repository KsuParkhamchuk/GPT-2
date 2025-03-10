import torch
from torch import nn
import torch.nn.functional as F
from hyperparams import VOCABULARY_SIZE, EMBEDDING_DIM, CONTEXT_SIZE, PE_N


class TokenEmbeddings(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCABULARY_SIZE, EMBEDDING_DIM)

    def forward(self, x):
        return self.token_embedding(x)


class PositionalEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

        pe = torch.zeros(CONTEXT_SIZE, EMBEDDING_DIM)
        # CONTEXT_SIZE x 1
        positions = torch.arange(CONTEXT_SIZE).unsqueeze(1)
        # formula: e in power of -2i*ln(n)/EMBEDDING_DIMENTION
        # torch.arange created sequeance like [0, 2 , 4 ... ] = 2i
        # EMBEDDING_DIM/2 x 1
        div_term = torch.exp(
            torch.arange(0, EMBEDDING_DIM, 2)
            * -(torch.log(torch.tensor(PE_N)) / EMBEDDING_DIM)
        ).unsqueeze(0)
        # fill each even value
        # syntax like : - take all rows, 0::2 - take columns from 0 to
        pe[:, 0::2] = torch.sin(positions * div_term)
        # fill each add value
        pe[:, 1::2] = torch.cos(positions * div_term)

        # instead of self.pe = pe
        # tensor will not receive any gradients during backprop
        # positional encoding is a constant tensor, should be saved with a model
        self.register_buffer("pe", pe)

    def forward(self, x):
        # take all rows and all columns from 0 to x second dimention which is CONTEXT_SIZE
        return x + self.pe[: x.size(1), :]


class EmbeddingLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_emb = TokenEmbeddings()
        self.positional_emb = PositionalEmbedding()

    def forward(self, x):
        # works as a look up table, returns the needed token embeddings
        # CONTEXT_SIZE x EMBEDDING_DIM
        token_emb = self.token_emb(x)
        # works as a look up table, returns needed positions embeddings
        # CONTEXT_SIZE x EMBEDDING_DIM
        input_emb = self.positional_emb(token_emb)

        return input_emb
