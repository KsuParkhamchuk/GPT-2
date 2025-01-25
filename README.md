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

Usage:

1. Identify max_length (context length), batch_size hyperparameters
2. Split into words (if needed)
3. Tokenize
4. Convert tokens to IDs
5. split into sequences of max_length
6. Pad and truncate to max length (context length)
7. Feed to model (each sequence will be processed independently by transformer)

![alt text](image.png)
