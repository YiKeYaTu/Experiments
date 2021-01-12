import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.models
import math


def loc_loss(locs_list, target_locs_list):
    loss = []

    for idx, (locs) in enumerate(locs_list):
        loss.append(
            torch.nn.functional.smooth_l1_loss(locs, target_locs_list[idx])
        )

    loss = torch.sum(torch.stack(loss))
    return loss

def saliency_loss(outputs_list, targets, weight_0=1.0, weight_1=1.12, eps=1e-15):
    loss = []

    for outputs in outputs_list:
        loss.append(
            torch.mean(
                -weight_1 * targets * torch.log(outputs + eps) -
                weight_0 * (1 - targets) * torch.log(1 - outputs + eps),
            )
        )

    loss = torch.sum(torch.stack(loss))
    return loss


def rank_loss(outputs_list, locs_list, targets, margin=0.05):
    loss = []
    rank_pairs = []

    for idx, (outputs) in enumerate(outputs_list[0:-1]):
        masks = torch.zeros_like(outputs)
        locs = locs_list[idx]
        
        for i, (loc) in enumerate(locs):
            tx, ty, tl = loc
            masks[i, :, ty - tl:ty + tl + 1, tx - tl:tx + tl + 1] = 1

        rank_pairs.append((
            outputs_list[idx] * targets * masks,
            outputs_list[idx + 1] * targets * masks
        ))

    for rank_pair in rank_pairs:
        loss.append(
            torch.mean(
                torch.clamp(rank_pair[0] - rank_pair[1] + margin, min=0),
            )
        )

    loss = torch.sum(torch.stack(loss))
    return loss

if __name__ == '__main__':
    img = cv2.imread('/home/ncrc-super/data/DataSets/saliency_deteciton/DUTS-TR/DUTS-TR-Mask/sun_dzkggnowaqnfrorl.png', 0)
    print(img.shape)

    o = torch.sigmoid(torch.randn((1, 268, 400)))

    outputs_list = [
        torch.stack([o, o]),
        torch.stack([o, o]),
        torch.stack([o, o]),
    ]

    locs_list = [
        torch.tensor([
            [200, 100, 50],
            [200, 100, 50],
        ]),
        torch.tensor([
            [200, 100, 20],
            [200, 100, 20]
        ])
    ]

    targets = torch.stack([
        torch.from_numpy(img).unsqueeze(0),
        torch.from_numpy(img).unsqueeze(0)
    ])

    # print(outputs_list[0].shape, targets.shape)
    
    print(rank_loss(outputs_list, locs_list, targets))
    print(saliency_loss([outputs for outputs in outputs_list], targets / 255))