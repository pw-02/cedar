import pathlib
import torch
import torchdata.datapipes as dp

from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from evaluation.torch_utils import TorchEvalSpec
from torch.utils.data import DataLoader

DATASET_LOC = "datasets/imagenette2"
IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 11


def to_float(x):
    return x.to(torch.float32)


def build_datapipe(root, spec: TorchEvalSpec):
    datapipe = dp.iter.FileLister(root=root, recursive=True)
    # TODO: Evaluate where is a fair place to put this...
    datapipe = datapipe.sharding_filter()
    datapipe = dp.iter.Mapper(
        datapipe, lambda x: read_image(x, mode=ImageReadMode.RGB)
    )
    datapipe = dp.iter.Mapper(datapipe, to_float)
    datapipe = dp.iter.Mapper(
        datapipe, transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH))
    )
    datapipe = dp.iter.Mapper(datapipe, transforms.RandomHorizontalFlip())
    datapipe = dp.iter.Mapper(
        datapipe, transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    )
    datapipe = dp.iter.Mapper(
        datapipe, transforms.Grayscale(num_output_channels=1)
    )
    datapipe = dp.iter.Mapper(
        datapipe, transforms.GaussianBlur(GAUSSIAN_BLUR_KERNEL_SIZE)
    )
    datapipe = dp.iter.Mapper(
        datapipe, transforms.Normalize((0.1307,), (0.3081,))
    )
    return datapipe


def get_dataset(spec: TorchEvalSpec):
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    train_filepath = pathlib.Path(data_dir) / pathlib.Path("imagenette2/train")

    datapipe = build_datapipe(str(train_filepath), spec)

    dataloader = DataLoader(
        datapipe, batch_size=spec.batch_size, num_workers=spec.num_workers
    )

    return dataloader


if __name__ == "__main__":
    dataset = get_dataset(TorchEvalSpec(8, 1))
    for x in dataset:
        print(x)
        print(x.size())
