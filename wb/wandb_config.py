import wandb


def init_wandb():
    wandb.init(
        project="gpt-2",
        config={
            "learning_rate": 0.01,
            "batch_size": 32,
            "number_of_epochs": 10,
        },
    )


def log_wandb(batch_loss):
    wandb.log({"batch_loss": batch_loss})
