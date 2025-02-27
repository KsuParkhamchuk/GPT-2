import torch
from torch import nn
from wb.wandb_config import log_batch_loss, log_lr, log_validation_loss
from hyperparams import EPOCH_NUMBER
import traceback


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_dataloader, val_dataloader):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss = 0

        for batch in self.val_dataloader:
            batch.to(self.device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = self.model(inputs)
            logits.view(-1, logits.size(-1))
            targets = targets.view(-1)

            loss = self.criterion(logits, targets)
            val_loss += loss.item()
            log_validation_loss(val_loss)

    @torch.no_grad
    def evaluate(self, dataloader):
        self.model.eval()
        pass

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
            # after slicing target sequence is not contiguous anymore
            target = target.contiguous().view(-1)
            loss = self.criterion(logits, target)

            # loss is a final node of a graph
            # see details inside experiments/backprop.ipynb
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            log_batch_loss(loss.item())
            print(f"Batch loss={loss.item()}")

        return epoch_loss

    def train(self):
        total_loss = 0

        try:
            for epoch in range(EPOCH_NUMBER):
                current_lr = self.optimizer.param_groups[0]["lr"]

                epoch_loss = self.train_epoch(self.dataloader)
                total_loss += epoch_loss
                print(
                    f"Epoch = {epoch}, epoch_loss = {epoch_loss}, total_loss = {total_loss}"
                )

                log_lr(current_lr)
                self.scheduler.step()
                self.validate()
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
        finally:
            # Force cleanup of workers
            self.dataloader._iterator = None
