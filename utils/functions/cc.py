def correct_image(image, illuminant):
    return image[:, :, :] = image[:, :, :] / illuminant[:]
