from torch import nn
from layers.embedding import EmbeddingLayer
from layers.transformer import DecoderBlock
from layers.normalization import NormalizationLayer
from model.hyperparams import (
    EMBEDDING_DIM,
    DECODER_BLOCKS,
    VOCABULARY_SIZE,
)
from layers.linear import Linear


class GPT2(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb_layer = EmbeddingLayer()
        self.layer_norm = NormalizationLayer()
        self.output_proj = Linear(EMBEDDING_DIM, VOCABULARY_SIZE, bias=False)
        self.decoder = nn.ModuleList([DecoderBlock() for _ in range(DECODER_BLOCKS)])

    def forward(self, tokens):
        # embeddings
        # [BATCH_SIZE x CONTEXT_SIZE x EMBEDDING_DIM]
        embeddings = self.emb_layer.forward(tokens)
        # transformer : multi-head attention, projections, MLP, attention residual
        x = embeddings
        for block in self.decoder:
            x = block(x)

        # [BATCH_SIZE x CONTEXT_SIZE x EMBEDDING_DIM]
        normalized_output = self.layer_norm(x)
        # [BATCH_SIZE x CONTEXT_SIZE x VOCAB_SIZE]
        logits = self.output_proj(normalized_output)
        return logits
