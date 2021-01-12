from models.RCF import RCF
from constant import *
from PIL import Image, ImageFilter
import numpy as np
import torch

model = RCF()
model.to(DEVICE)
model.load_state_dict(
    torch.load(
        os.path.join(PROJECT_ROOT, 'pths/RCFcheckpoint_epoch12.pth'),
        map_location=DEVICE
    )['state_dict']
)


def get_edge_map(im, radius=20):
    # Convert a normal image to a tensor with a batch dimension.
    im = np.array(im, dtype=np.float32)
    im = torch.from_numpy(
        np.expand_dims(
            im.transpose(2, 0, 1),
            0
        )
    )
    edge_map = model(im)[-1][0] * 255
    edge_map = torch.cat((edge_map, edge_map, edge_map), 0) \
        .permute(1, 2, 0).to('cpu', torch.uint8).detach().numpy()
    edge_map = Image.fromarray(edge_map).filter(ImageFilter.GaussianBlur(radius=radius))

    return np.array(edge_map)
