from datasets.dataset import TextDataset
from bpe_tokenizer import BPETokenizer
from wb import init_wandb
from torch.utils.data import DataLoader
from model import BATCH_SIZE, LEARNING_RATE
from model import GPT2
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from model import Trainer

wikitext_file = "datasets/wikitext/train1.arrow"
wikitext_test = "datasets/wikitext/test.arrow"
wikitext_validate = "datasets/wikitext/validate.arrow"


def init_tokenizer():
    return BPETokenizer("general")


def init_datasets(tokenizer):
    train_dataset = TextDataset(tokenizer, wikitext_file)
    test_dataset = TextDataset(tokenizer, wikitext_test)
    validate_dataset = TextDataset(tokenizer, wikitext_validate)

    return train_dataset, test_dataset, validate_dataset


def init_dataloaders(tokenizer):
    train_dataset, test_dataset, validate_dataset = init_datasets(tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return train_dataloader, test_dataloader, validate_dataloader


# Clean up dataloaders resources to avoid memory leak
def cleanup_dataloaders(dataloaders):

    for dataloader in dataloaders:
        # Close the workers to avoid memory leaks
        dataloader._iterator = None


def init_optimizer(model):
    # parameters are automatically tracked by nn.Module, otherwise should be registered through nn.Parameter
    return AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)


def init_scheduler(optimizer):
    return StepLR(optimizer, step_size=10)


def init_model():
    return GPT2()


def main():
    init_wandb()
    model = init_model()
    optimizer = init_optimizer(model)
    scheduler = init_scheduler(optimizer)
    tokenizer = init_tokenizer()
    train_dataloader, validate_dataloader, test_dataloader = init_dataloaders(tokenizer)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=validate_dataloader,
    )

    trainer.train()
    trainer.evaluate(test_dataloader)

    cleanup_dataloaders([train_dataloader, test_dataloader, validate_dataloader])


if __name__ == "__main__":
    main()
