import torch
from models.PFAN.recurrent_model import caculate_mask

print(caculate_mask(torch.full((1, 1, 10, 10), 1), 3, 3, 2))