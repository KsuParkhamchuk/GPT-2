import torch
from torch import nn
from layers.embedding import TokenEmbeddings
from layers.normalization import LayerNormalization
from layers.transformer import DecoderBlock
from layers.hyperparams import (
    EMBEDDING_DIM,
    ATTENTION_HEADS,
    DECODER_BLOCKS,
    VOCABULARY_SIZE,
)


class GPT2(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb_layer = TokenEmbeddings()
        self.layer_norm = LayerNormalization()
        self.output_proj = nn.Linear(EMBEDDING_DIM, VOCABULARY_SIZE, bias=False)
        self.decoder = nn.ModuleList([decoder for decoder in range(DECODER_BLOCKS)])

    def forward(self, x):
        # tokenization
        # embeddings
        embeddings = self.emb_layer.forward(x)
        # transformer : multi-head attention, projections, MLP, attention residual
        transformer_output = self.decoder(embeddings)
        normalized_output = self.layer_norm.forward(transformer_output)
        logits = self.output_proj(normalized_output)
        return logits
