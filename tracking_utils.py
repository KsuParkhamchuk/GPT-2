import torch
import time


def get_time():
    return time.time()


def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss))


def get_gradient_metrics(parameters):
    # Initialize variables
    total_squared_norm = 0.0
    max_grad = -float("inf")
    min_grad = float("inf")

    # Variables to track non-zero gradients
    non_zero_grads = []

    for param in parameters:
        # Skip parameters without gradients
        if param.grad is None:
            continue

        # Get the gradient data
        grad_data = param.grad.data

        # Update total norm (sum of squared L2 norms)
        param_norm = grad_data.norm(2).item()
        total_squared_norm += param_norm**2

        # Find max absolute gradient (bc the direction does not matter)
        max_abs_grad = grad_data.abs().max().item()
        max_grad = max(max_grad, max_abs_grad)

        flat_grad = grad_data.abs().flatten()
        non_zero_flat_grad = flat_grad[flat_grad > 0]
        if non_zero_flat_grad.numel() > 0:
            non_zero_grads.append(non_zero_flat_grad.min().item())

    # Calculate final L2 norm
    grad_norm = torch.sqrt(torch.tensor(total_squared_norm))

    # Get min non-zero gradient
    min_grad = min(non_zero_grads) if non_zero_grads else 0

    return {"grad_norm": grad_norm, "max_grad": max_grad, "min_grad": min_grad}


def calculate_accuracy(predictions, targets):
    _, predicted = torch.max(predictions, dim=-1)
    correct = (predicted == targets).float()
    return correct.mean().item()
