from torchvision.io import ImageReadMode, read_image
import torch

IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 11


def read_image_pytorch(x):
    return read_image(x, mode=ImageReadMode.RGB)


def to_float(x):
    return x.to(torch.float32)
