import torch


def save_checkpoint(state, filename):
    tmp_path = filename + ".tmp"
    torch.save(state, tmp_path)
    print(f"Checkpoint saved to {filename}")


def verify_checkpoint(state):
    required_keys = {"model", "optimizer", "step", "config"}
    return all(k in state for k in required_keys)


def load_checkpoint(path):
    try:
        state = torch.load(path, map_location="cpu")
        if verify_checkpoint(state):
            return state
    except Exception as e:
        print(f"Checkpoint load failed: {e}")
    return None
