import torch
import math
import numpy as np


def angular_loss(output, target):
    angular_losses = (180.0 / math.pi) * torch.acos(
        torch.clamp(
            torch.sum(output * target, dim=1) /
            torch.norm(output, dim=1) /
            torch.norm(target, dim=1),
            min=-1, max=1
        )
    )
    return torch.mean(angular_losses)


def calc_AngErr(illu_1, illu_2):
    return np.degrees(np.arccos(np.sum(illu_1*illu_2)/(np.sqrt(np.sum(illu_1*illu_1))*np.sqrt(np.sum(illu_2*illu_2)))))
