# layers
VOCABULARY_SIZE = 8879
EMBEDDING_DIM = 768
CONTEXT_SIZE = 1024
PE_N = 8879
DECODER_BLOCKS = 12
ATTENTION_HEADS = 12

# train
EPOCH_NUMBER = 100
# step during optimization
# LR scheduler types:
# 1. OneCycleLR - includes warmup and cooldown, updates every batch, good for transformers
# Used when loss is unstable at the start or spikes randomly
# 2. ExponentialLR - aggressive increase
# Used when training is too slow
# 3. ReduceLROnPlateau - reduces lr when the progress stops
# 4. CosineAnnealingLR - popular in computer vision tasks, good default
# 5. StepLR - simple step decay
LEARNING_RATE = 0.001
BATCH_SIZE = 32
