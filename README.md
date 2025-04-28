# GPT-2 Full rewrite

This repository contains a full rewrite of the GPT-2 model with many layers and concepts implemented from scratch.
It is a learning resource for understanding the inner workings of the model.

**Table of contents:**

- [Concepts](./concepts) - Jupyter notebooks with visualizations and explanations of the concepts

  - [Positional encoding](./concepts/positional_encoding.ipynb) +
  - [Backpropagation](./concepts/backprop.ipynb)
  - [Cross-entropy](./concepts/cross_entropy.ipynb)
  - [Optimizers](./concepts/optimizers.ipynb)

- [Layers](./layers) - Implementations of the layers used in the model
  - [Embedding](./layers/embedding.py)
  - [Transformer](./layers/transformer.py)
  - [Linear](./layers/linear.py)
  - [Normalization](./layers/normalization.py)
  - [Optimizations](./layers/optimizations)
    - [RoPE Implementation](./layers/optimizations/RoPE.py)
    - [KV Cache Implementation](./layers/optimizations/KVCache.py)
- [Model](./model) - The main model implementation
- [Train](./train) - Training loop implementation
- [Checkpoints](./checkpoints) - Checkpoint for the model
