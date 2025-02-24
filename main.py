from datasets.dataset import TextDataset
from tokenizer import BPETokenizer
from wandb import init_wandb
from torch.utils.data import DataLoader
from hyperparams import BATCH_SIZE, LEARNING_RATE
from models import GPT2
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from train import Trainer

wikitext_file = "datasets/wikitext/train1.arrow"
wikitext_test = "datasets/wikitext/test.arrow"
wikitext_validate = "datasets/wikitext/validate.arrow"


def init_tokenizer():
    return BPETokenizer()


def init_datasets(tokenizer):
    train_dataset = TextDataset(tokenizer, wikitext_file)
    test_dataset = TextDataset(tokenizer, wikitext_file)
    validate_dataset = TextDataset(tokenizer, wikitext_validate)

    return train_dataset, test_dataset, validate_dataset


def init_dataloaders():
    train_dataset, test_dataset, validate_dataset = init_datasets()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )

    return train_dataloader, test_dataloader, validate_dataloader


def init_optimizer(model):
    # parameters are automatically tracked by nn.Module, otherwise should be registered through nn.Parameter
    return AdamW(model.parameters(), lr=LEARNING_RATE)


def init_scheduler(optimizer):
    return StepLR(optimizer, step_size=10)


def init_model():
    return GPT2()


def main():
    init_wandb()
    model = init_model()
    optimizer = init_optimizer(model)
    scheduler = init_scheduler(optimizer)
    train_dataloader, validate_dataloader, test_dataloader = init_dataloaders()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=validate_dataloader,
    )

    trainer.train()
    trainer.evaluate(test_dataloader)


if __name__ == "__main__":
    main()
