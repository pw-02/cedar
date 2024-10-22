import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.datapoints import BoundingBox, BoundingBoxFormat


def to_float(x):
    return x.to(torch.float32)


def to_tensor(x):
    x["image"] = v2.ToTensor()(x["image"])
    return x


def distort(x):
    x["image"] = v2.RandomPhotometricDistort(p=1)(x["image"])
    return x


def zoom_out(x):
    x["image"], x["boxes"] = v2.RandomZoomOut(fill=[123.0, 117.0, 104.0], p=1)(
        x["image"], x["boxes"]
    )
    return x


def crop(x):
    x["image"], x["boxes"] = v2.RandomIoUCrop()(x["image"], x["boxes"])
    return x


def read_image(x):
    img = Image.open(x["image"]).convert("RGB")
    x["image"] = img
    bboxes = BoundingBox(
        x["boxes"],
        format=BoundingBoxFormat.XYXY,
        spatial_size=(img.height, img.width),
    )
    x["boxes"] = bboxes

    return x
