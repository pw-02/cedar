import pathlib
import torch
import glob
from torchvision import transforms
import PIL
import ray
import time

DATASET_LOC = "datasets/imagenette2"
IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 11


class Timer:
    def __init__(self):
        self._start = None
        self._end = None

    def __enter__(self):
        self._start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter()

    def reset(self):
        self._start = time.perf_counter()

    def delta(self):
        if self._start is None or self._end is None:
            raise RuntimeError()
        return self._end - self._start


def read_img(x):
    return {"image": PIL.Image.open(x["item"])}


def transform_img(x):
    transform = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.Grayscale(num_output_channels=1),
            transforms.GaussianBlur(GAUSSIAN_BLUR_KERNEL_SIZE),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return {"image": transform(x["image"])}


def ret_img(x):
    return x


def build_ds(root):
    # Get list of dirs
    dir_list = []
    file_list = []
    for item in pathlib.Path(root).iterdir():
        if item.is_dir():
            dir_list.append(str(item))
            files = glob.glob(f"{str(item)}/*.JPEG")
            file_list.extend(files)
    ds = ray.data.from_items(file_list)
    ds = ds.map(read_img)
    ds = ds.map(transform_img)

    return ds


def get_dataset():
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    train_filepath = pathlib.Path(data_dir) / pathlib.Path("imagenette2/train")

    ds = build_ds(str(train_filepath))
    ray.data.DataContext.get_current().execution_options.locality_with_output = (
        True
    )

    return ds


if __name__ == "__main__":
    ds = get_dataset()
    epoch_times = []
    for _ in range(3):
        timer = Timer()
        with timer:
            for idx, row in enumerate(ds.iter_rows()):
                pass
        epoch_times.append(timer.delta())
        print(epoch_times[-1])

    print("Epoch times: {}".format(epoch_times))
