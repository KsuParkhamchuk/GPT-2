## Transformer architecture

## Learning materials

[GPT 2 illustrated article](https://jalammar.github.io/illustrated-gpt2/)
[LLM visualizations](https://bbycroft.net/llm)

## Tokenizer

**BPE - Byte Pair Encoding**

Vocabulary:
Vocabulary is built iteratively during training.
Starts with all unique characters in training data + special tokens (<unk>, <pad>, <eos> ...).

input: raw text
output: sequence of tokens

Steps:

Training:

1. Split into words (to identify the word boundaries if needed)
   - grammatical meaning
   - semantic meaning
2. Pair counting
3. Merge most frequent pair
4. Update vocabulary
5. Mapping with indexes (= index of vocabulary array)
6. Repeat until vocabulary size is reached

Inference:

1. Identify max_length (context length), batch_size hyperparameters
2. Split into words (if needed)
3. Tokenize
4. Convert tokens to IDs
5. Split into sequences of max_length
6. Pad and truncate to max length (context length)
7. Feed to model (each sequence will be processed independently by transformer)

![tokenizer vizualization](img/tokenization.png)

## Embedding layer

Hyperparameters:

- embedding_dim
  GPT-2 small: 768
  GPT-2 medium 1024
  GPT-2 large 1280
  GPT-2 XL 1600

- context_length
  GPT-2 small: 768
  GPT-2 medium 1024
  GPT-2 large 1280
  GPT-2 XL 1600

**Training:**

1. Init positional encoding matrix with random values
2. Init token embedding matrix with random values

**Inference:**

Token embeddings + positional embeddings = input embeddings

1. Token embeddings lookup. Each index in the token embedding matrix encodes a corresponding token from tokenizer vocabulary.
2. Positional embeddings lookup.
3. Sum of token embeddings and positional embeddings = input embeddings
4. Truncate or pad input embeddings to context length

![Embedding layer vizualization](img/embedding.png)

## Q&A
