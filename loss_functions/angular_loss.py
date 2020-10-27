import torch
import math
import numpy as np


def angular_loss(output, target, reduction='sum'):
    if reduction is 'mean':
        reduce_fn = torch.mean
    elif reduction is 'sum':
        reduce_fn = torch.sum
    else:
        reduce_fn = torch.mean
        
    angular_losses = (180 / math.pi) * torch.acos(
        torch.clamp(
            reduce_fn(output * target / torch.norm(output) /
                      torch.norm(target), dim=1), min=-1, max=1))

    return torch.mean(angular_losses)