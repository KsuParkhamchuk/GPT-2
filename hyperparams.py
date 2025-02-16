# layers
VOCABULARY_SIZE = 50257  # 50 000 BPE merges + 256 bytes tokens + 1 <endoftext> token
EMBEDDING_DIM = 768
CONTEXT_SIZE = 1024
PE_N = 10000
DECODER_BLOCKS = 12
ATTENTION_HEADS = 12

# train
EPOCH_NUMBER = 10
# step during optimization
LEARNING_RATE = 0.001
BATCH_SIZE = 32
