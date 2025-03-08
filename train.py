import torch
from torch import nn
from wb.wandb_config import log_param
from hyperparams import EPOCH_NUMBER
import traceback
from tracking_utils import (
    get_time,
    calculate_perplexity,
    get_gradient_metrics,
    calculate_accuracy,
)


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
            batch = batch.to(self.device)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = self.model(inputs)
            logits = logits.view(-1, logits.size(-1))
            targets = targets.contiguous().view(-1)

            loss = self.criterion(logits, targets)
            val_loss += loss.item()
            val_accuracy = calculate_accuracy(logits, targets)
            log_param("validation accuracy", val_accuracy)
            log_param("validation loss", val_loss)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        pass

    def train_epoch(self):
        start_time = get_time()
        epoch_loss = 0

        for batch in self.train_dataloader:
            batch_start_time = get_time()
            self.optimizer.zero_grad()
            # Move batch to device
            batch = batch.to(self.device)
            # all tokens except the last one
            inputs = batch[:, :-1]
            # all tokens except the first one
            target = batch[:, 1:]
            self.model.train()

            logits = self.model.forward(inputs)
            # reshape to [batch_size*sequence_length, vocab_size]
            logits = logits.view(-1, logits.size(-1))
            # after slicing target sequence is not contiguous anymore
            target = target.contiguous().view(-1)
            loss = self.criterion(logits, target)

            # loss is a final node of a graph
            # see details inside experiments/backprop.ipynb
            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()

            batch_finish_time = get_time()
            grad_metrics = get_gradient_metrics(self.model.parameters())
            accuracy = calculate_accuracy(logits, target)
            log_param("accuracy", accuracy)
            for m_name, m_value in grad_metrics.items():
                log_param(m_name, m_value)
            log_param("batch loss", loss.item())
            log_param("batch time", batch_finish_time - batch_start_time)
            log_param("perplexity", calculate_perplexity(loss.item()))
            print(
                f"Batch loss={loss.item()}, perplexity={calculate_perplexity(loss.item())}, batch time={batch_finish_time - batch_start_time}"
            )

        finish_time = get_time()
        log_param("epoch duration", finish_time - start_time)
        log_param("epoch loss", epoch_loss)

        return epoch_loss

    def train(self):
        total_loss = 0

        try:
            for epoch in range(EPOCH_NUMBER):
                current_lr = self.optimizer.param_groups[0]["lr"]

                epoch_loss = self.train_epoch()
                total_loss += epoch_loss
                print(
                    f"Epoch = {epoch}, epoch_loss = {epoch_loss}, total_loss = {total_loss}"
                )

                log_param("learning rate", current_lr)
                self.scheduler.step()
                self.validate()
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
