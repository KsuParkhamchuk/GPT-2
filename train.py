from models.model import GPT2
from torch import nn
from torch.optim import AdamW
from wb.wandb_config import init_wandb, log_wandb
from tokenizer.BPETokenizer import BPETokenizer
from torch.utils.data import DataLoader
from hyperparams import EPOCH_NUMBER, LEARNING_RATE, BATCH_SIZE
from datasets import TextDataset


def train():
    init_wandb()
    model = GPT2()
    tokenizer = BPETokenizer()
    # parameters are automatically tracked by nn.Module, otherwise should be registered through nn.Parameter
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    dataset = TextDataset(tokenizer)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCH_NUMBER):

        for batch in train_dataloader:
            optimizer.zero_grad()
            # batch shape: [batch_size, context_size + 1]
            # all tokens except the last one
            inputs = batch[:, :-1]
            # all tokens except the first one
            target = batch[:, 1:]
            model.train()

            logits = model.forward(inputs)
            loss = criterion(logits)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()


tokenizer = BPETokenizer()
dataset = TextDataset(tokenizer)
