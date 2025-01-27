import torch
from torch import nn
import torch.nn.functional as F
from hyperpearams import VOCABULARY_SIZE, EMBEDDING_DIM, CONTEXT_SIZE


class TokenEmbeddings(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCABULARY_SIZE, EMBEDDING_DIM)

    def forward(self, x):
        return self.token_embedding(x)


class PositionalEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.positional_embedding = nn.Embedding(CONTEXT_SIZE, EMBEDDING_DIM)

    def forward(self, x):
        return self.positional_embedding(x)


class EmbeddingLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_emb = TokenEmbeddings()
        self.positional_emb = PositionalEmbedding()

    def forward(self, x):
        # works as a look up table, returns the needed token embeddings
        # CONTEXT_SIZE x EMBEDDING_DIM
        token_emb = self.token_emb(x)
        # range from 0 til end [0,1, ... , end]
        positions = torch.arange(CONTEXT_SIZE)
        # works as a look up table, returns needed positions embeddings
        # CONTEXT_SIZE x EMBEDDING_DIM
        positional_emb = self.positional_emb(positions)

        return token_emb + positional_emb
