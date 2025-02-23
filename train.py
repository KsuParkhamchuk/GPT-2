import torch
from models.model import GPT2
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from wb.wandb_config import init_wandb, log_batch_loss, log_lr
from tokenizer import BPETokenizer
from torch.utils.data import DataLoader
from hyperparams import EPOCH_NUMBER, LEARNING_RATE, BATCH_SIZE
from datasets import TextDataset


class Trainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        val_loss = 0

        for batch in dataloader:
            batch.to(self.device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = self.model(inputs)
            logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            loss = self.criterion(logits, targets)
            val_loss += loss.item()

    def train_epoch(self, train_dataloader):
        epoch_loss = 0

        for batch in train_dataloader:
            self.optimizer.zero_grad()
            # batch shape: [batch_size, context_size + 1]
            # all tokens except the last one
            inputs = batch[:, :-1]
            # all tokens except the first one
            target = batch[:, 1:]
            self.model.train()

            logits = self.model.forward(inputs)
            logits = logits.view(
                -1, logits.size(-1)
            )  # reshape to [batch_size*sequence_length, vocab_size]
            target = target.view(-1)
            loss = self.criterion(logits, target)

            # loss is a final node of a graph
            # see details inside experiments/backprop.ipynb
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            log_batch_loss(loss.item())
            print(f"Batch loss={loss.item()}")

        return epoch_loss


def train():
    init_wandb()
    device = str = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2()
    tokenizer = BPETokenizer()
    # parameters are automatically tracked by nn.Module, otherwise should be registered through nn.Parameter
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=10)
    total_loss = 0

    dataset = TextDataset(tokenizer)
    train_dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    trainer = Trainer(model, optimizer, scheduler, device)

    try:
        for epoch in range(EPOCH_NUMBER):
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_loss = trainer.train_epoch(train_dataloader)
            total_loss += epoch_loss
            print(
                f"Epoch = {epoch}, epoch_loss = {epoch_loss}, total_loss = {total_loss}"
            )

            log_lr(current_lr)
            scheduler.step()
    finally:
        # Force cleanup of workers
        train_dataloader._iterator = None


if __name__ == "__main__":
    train()
