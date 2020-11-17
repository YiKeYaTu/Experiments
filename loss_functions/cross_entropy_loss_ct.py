import torch


def cross_entropy_loss_ct(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduction='sum')

    return torch.sum(cost)

def cross_entropy_loss(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask > 0).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask > 0] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)

    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduction='sum')

    return torch.sum(cost)
