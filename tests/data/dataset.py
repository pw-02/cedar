import pathlib
import torch
import functools
import operator
from typing import (
    Any,
    BinaryIO,
    cast,
    Iterator,
    List,
    Tuple,
)
from torchvision import transforms


from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import (
    IterDataPipe,
    FileOpener,
    IterableWrapper,
)

from cedar.utils.profiler import ProfilerSpec
from cedar_benchmarks.mnist.utils import tensor_from_file


prod = functools.partial(functools.reduce, operator.mul)


def _to_float(x):
    return x.to(torch.float32)


def _reshape(x):
    bsize = len(x)
    tensor = torch.stack(x)
    return torch.reshape(tensor, (bsize, 1, 28, 28))


@functional_datapipe("read_mnist_file")
class MNISTFileReader(IterDataPipe[torch.Tensor]):
    """
    Adapted from https://github.com/pytorch/vision/blob/main/torchvision/prototype/datasets/_builtin/mnist.py  # noqa: E501
    """

    _DTYPE_MAP = {
        8: torch.uint8,
        9: torch.int8,
        11: torch.int16,
        12: torch.int32,
        13: torch.float32,
        14: torch.float64,
    }

    def __init__(self, datapipe: IterDataPipe[Tuple[Any, BinaryIO]]) -> None:
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[torch.Tensor]:
        for _, file in self.datapipe:
            try:
                read = functools.partial(
                    tensor_from_file, file, byte_order="big"
                )

                magic = int(read(dtype=torch.int32, count=1))
                dtype = self._DTYPE_MAP[magic // 256]
                ndim = magic % 256 - 1

                num_samples = int(read(dtype=torch.int32, count=1))
                shape = (
                    cast(
                        List[int], read(dtype=torch.int32, count=ndim).tolist()
                    )
                    if ndim
                    else []
                )
                count = prod(shape) if shape else 1

                for _ in range(num_samples):
                    yield read(dtype=dtype, count=count).reshape(shape)
            finally:
                file.close()


@functional_datapipe("lazy_zip")
class LazyZipperIterDataPipe(IterDataPipe):
    """
    Zips together elements into a tuple from input DataPipes.
    The output is yielded one by one - does not buffer
    the entire input iterable, only yields a single element
    when the output is requested.

    Args:
        *datapipes: Iterable DataPipes being aggregated

    """

    def __init__(self, *datapipes: IterDataPipe) -> None:
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("All inputs are required to be `IterDataPipe` ")

        super().__init__()
        self.datapipes = datapipes

    def __iter__(self) -> Iterator[Tuple]:
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        yield from zip(*iterators)


def get_dataset(spec: ProfilerSpec):
    data_dir = pathlib.Path(__file__).resolve().parents[0]
    train_imgs = pathlib.Path(data_dir) / "t10k-images-idx3-ubyte.gz"
    train_labels = pathlib.Path(data_dir) / "t10k-labels-idx1-ubyte.gz"

    img_dp = IterableWrapper((str(train_imgs),))
    img_dp = FileOpener(img_dp, mode="rb")
    label_dp = IterableWrapper((str(train_labels),))
    label_dp = FileOpener(label_dp, mode="rb")

    img_dp = (
        img_dp.decompress()
        .read_mnist_file()
        .map(_to_float)
        .batch(spec.batch_size)
        .map(_reshape)
        .map(transforms.Normalize((0.1307,), (0.3081,)))
    )

    label_dp = (
        label_dp.decompress()
        .read_mnist_file()
        .batch(spec.batch_size)
        .map(torch.stack)
    )

    dp = img_dp.lazy_zip(label_dp)
    return dp
