import wandb


def init_wandb():
    wandb.init(
        project="gpt-2",
        config={
            "learning_rate": 0.01,
            "batch_size": 32,
            "number_of_epochs": 100,
        },
    )


def log_batch_loss(batch_loss):
    wandb.log({"batch_loss": batch_loss})


def log_lr(lr):
    wandb.log({"learning_rate": lr})


def log_validation_loss(val_loss):
    wandb.log({"validation_loss": val_loss})
