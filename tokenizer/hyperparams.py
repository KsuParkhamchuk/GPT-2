GPT4_SPLIT_PATTERN = rb"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
BASE_VOCABULARY_SIZE = 258
N_RAW_BYTES = 256
TARGET_VOCABULARY_SIZE = 500
