import torch


def save_checkpoint(model, optimizer, epoch, batch, loss):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "batch": batch,
    }

    torch.save(checkpoint, f"checkpoints/model_checkpoint_epoch_{epoch}.pt")
    print(f"Checkpoint saved at epoch = {epoch}, batch = {batch}")


def load_checkpoint(model, optimizer, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state-dict"])
        epoch = checkpoint["epoch"]
        batch = checkpoint["batch"]
        return model, optimizer, epoch, batch
    except Exception as e:
        print(f"Fail to load checkpoint {checkpoint_path}, exception: {e}")

    return None
