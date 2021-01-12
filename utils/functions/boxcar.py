import torch

def h(x):
    return 1.0 / (1.0 + torch.exp(1e4 * - x))

def boxcar_mask(image, tx, ty, tl):
    mask = torch.zeros_like(image)
    coordinate_y = torch.zeros_like(image)
    coordinate_x = torch.zeros_like(image)

    coordinate_y[:, :, :, :] = torch.arange(0, image.shape[2])
    coordinate_x = coordinate_y.transpose(2, 3)

    mask[:, :, :, :] = (h(coordinate_x - tx + tl) - h(coordinate_x - tx - tl)) * (h(coordinate_y - ty + tl) - h(coordinate_y - ty - tl))

    return mask

if __name__ == '__main__':
    print(
        boxcar_mask(torch.full((1, 1, 10, 10), 1), torch.full((1, 1, 1, 1), 3), torch.full((1, 1, 1, 1), 3), torch.full((1, 1, 1, 1), 1))
    )
    # print(boxcar_mask(torch.full((1, 1, 10, 10), 1), 3, 3, 0))
    # print(boxcar_mask(torch.full((1, 1, 10, 10), 1), 4, 4, 1))
    # print(boxcar_mask(torch.full((1, 1, 10, 10), 1), 200, 200, 1))