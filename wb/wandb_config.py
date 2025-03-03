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


def log_param(title, value):
    wandb.log({f"{title}": value})
