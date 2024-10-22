"""
Implements data pipes and helper functions necessary for SimCLRV2 pipelines.
"""

import h5py
import logging
import pathlib
import tarfile
import torch
import os
import numpy as np
import pandas as pd

from torchdata.datapipes.iter import IterDataPipe

from torchvision import transforms
from torchvision.io import read_image

from torchdata.datapipes import functional_datapipe

from typing import (
    BinaryIO,
    Iterator,
    List,
    Tuple,
)

DATASET_NAME = "imagenette2"
DATASET_LOC = "datasets/imagenette2"
DATASET_FILE = "imagenette2.tgz"
INTERMEDIATE_STORAGE_LOC = "datasets/imagenette2/intermediate"
DATASET_SOURCE = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 3


# HELPER FUNCTIONS


def to_float(x):
    return x.to(torch.float32)


# Recursively explores all the files in the directory specified as the
# source directory
def get_all_data_paths(data_source) -> List[str]:
    file_paths = []
    for root, dirs, files in os.walk(data_source):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


# DATA PIPES


@functional_datapipe("read_imagenette_imgs")
class ImagenetteFileReader(IterDataPipe[torch.Tensor]):
    """
    Implements the image reading for the Imagenette2 dataset.

    datapipe (IterDataPipe[Tuple[str, BinaryIO]): iterable datapipe containing
                                                tuples with the filepath as
                                                first elem
    """

    def __init__(self, datapipe: IterDataPipe[Tuple[str, BinaryIO]]) -> None:
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[torch.Tensor]:
        for path, file in self.datapipe:
            image_tensor = read_image(path)
            yield image_tensor


@functional_datapipe("simclrv2_transforms")
class SimCLRV2Transformer(IterDataPipe[torch.Tensor]):
    """
    Data Pipe that takes images and applies SimCLR_V2 transformations.
    When iterated over, yields tensors representing transformed images.

    datapipe (IterDataPipe[torch.Tensor]): iterable datapipe with source
                                            images loaded as tensors.
    """

    def __init__(self, datapipe: IterDataPipe[torch.Tensor]):
        super().__init__()
        self.width = IMG_WIDTH
        self.height = IMG_HEIGHT
        self.datapipe = datapipe
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop((self.height, self.width)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.Grayscale(num_output_channels=1),
                transforms.GaussianBlur(GAUSSIAN_BLUR_KERNEL_SIZE),
            ]
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        for tensor_image in self.datapipe:
            transformed_image = self.transform(tensor_image)
            yield transformed_image


@functional_datapipe("simclrv2_transforms_filter")
class SimCLRV2TransformerFilter(IterDataPipe[torch.Tensor]):
    """
    Data Pipe that takes images and applies SimCLR_V2 transformations.
    When iterated over, yields tensors representing transformed images.

    Differences to SimCLRV2Transformer:
    - Filters out images that have width < IMG_WIDTH or height < IMG_HEIGHT
    - sets num_output_channels to 3 for Grayscale

    datapipe (IterDataPipe[torch.Tensor]): iterable datapipe with source
                                            images loaded as tensors.
    """

    def __init__(self, datapipe: IterDataPipe[torch.Tensor]):
        super().__init__()
        self.width = IMG_WIDTH
        self.height = IMG_HEIGHT
        self.datapipe = datapipe
        self.transform = transforms.Compose(
            [
                transforms.RandomCrop((self.height, self.width)),
                transforms.Resize((self.height, self.width)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(),
                transforms.Grayscale(num_output_channels=3),
                transforms.GaussianBlur(GAUSSIAN_BLUR_KERNEL_SIZE),
            ]
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        for tensor_image in self.datapipe:
            channels, width, height = tensor_image.shape
            if width < IMG_WIDTH or height < IMG_HEIGHT:
                continue

            transformed_image = self.transform(tensor_image)
            yield transformed_image


@functional_datapipe("cache_data")
class DataCacher(IterDataPipe[torch.Tensor]):
    """
    Implements DataCacher, which reads data from cache if
    there, or fetches it from upstream datapipes otherwise.

    datapipe (IterDataPipe[str]): iterable datapipe containing image tensors.
    file_type (str): file type in which to store cached data. Default is .pt.
    """

    def __init__(
        self, datapipe: IterDataPipe[torch.Tensor], file_type: str = "pt"
    ) -> None:
        logging.info(f"Initializing cache with file type: {file_type}")
        self.datapipe = datapipe
        self.file_type = file_type
        self.save_func_dict = {
            "pt": self._save_pt,
            "npz": self._save_npz,
            "csv": self._save_csv,
            "hdf5": self._save_hdf5,
        }
        self.load_func_dict = {
            "pt": self._load_pt,
            "npz": self._load_npz,
            "csv": self._load_csv,
            "hdf5": self._load_hdf5,
        }
        self.save_function = self._get_save_function(file_type)
        self.load_function = self._get_load_function(file_type)
        self.other_info = {}

    def _get_save_function(self, file_type: str):
        return self.save_func_dict.get(file_type, self._invalid_type)

    def _get_load_function(self, file_type: str):
        return self.load_func_dict.get(file_type, self._invalid_type)

    # Functions to save to / load from different file types

    def _save_pt(self, tensor: torch.Tensor, file_name: str):
        torch.save(tensor, f"{file_name}.pt")

    def _load_pt(self, file_name: str):
        return torch.load(file_name)

    def _save_npz(self, tensor: torch.Tensor, file_name: str):
        np.savez(f"{file_name}.npz", tensor)

    def _load_npz(self, file_name: str):
        np_image = np.load(file_name)["arr_0"]
        return torch.from_numpy(np_image)

    def _save_hdf5(self, tensor: torch.Tensor, file_name: str) -> None:
        # TODO: Currently hard coded --> change
        storage_dir = (
            pathlib.Path(__file__)
            .resolve()
            .parents[1]
            .joinpath(INTERMEDIATE_STORAGE_LOC)
        )
        hdf5_storage_loc = pathlib.Path(storage_dir) / pathlib.Path(
            "cache_file.hdf5"
        )
        with h5py.File(str(hdf5_storage_loc), "a") as f:
            f.create_dataset(file_name, data=tensor.numpy())

    def _load_hdf5(self, file_name: str) -> torch.Tensor:
        storage_dir = (
            pathlib.Path(__file__)
            .resolve()
            .parents[1]
            .joinpath(INTERMEDIATE_STORAGE_LOC)
        )
        hdf5_storage_loc = pathlib.Path(storage_dir) / pathlib.Path(
            "cache_file.hdf5"
        )
        with h5py.File(str(hdf5_storage_loc), "r") as f:
            return torch.from_numpy(f[file_name])

    def _save_csv(self, tensor: torch.Tensor, file_name: str):
        tensor = tensor.numpy()
        # Save metadata in order to recover higher-dimensional arrays
        tensor_metadata = tensor.shape
        flattened_tensor = tensor.reshape(tensor_metadata[0], -1)
        tensor_table = pd.DataFrame(flattened_tensor)
        with open(f"{file_name}.csv", "w") as f:
            tensor_metadata = [str(elem) for elem in tensor_metadata]
            f.write(",".join(tensor_metadata) + "\n")
            tensor_table.to_csv(f, index=False)

    def _load_csv(self, file_name: str):
        with open(file_name, "r") as f:
            tensor_metadata = f.readline().strip().split(",")
            tensor_metadata = [int(elem) for elem in tensor_metadata]
            tensor_table = pd.read_csv(f)
        flattened_tensor = tensor_table.values
        tensor = flattened_tensor.reshape(tensor_metadata)
        return torch.from_numpy(tensor)

    def _invalid_type(self):
        raise ValueError(f"Invalid file type {self.file_type} specified.")

    # Saving / Loading

    def save(self, tensor: torch.Tensor, file_name: str):
        self.save_function(tensor, file_name)

    def load(self, file_name: str):
        return self.load_function(file_name)

    def __iter__(self) -> Iterator[torch.Tensor]:
        logging.info("Checking existence of intermediate storage dir")
        storage_dir = (
            pathlib.Path(__file__)
            .resolve()
            .parents[1]
            .joinpath(INTERMEDIATE_STORAGE_LOC)
        )
        exists = pathlib.Path.exists(storage_dir)
        logging.info(f"Intermediate storage dir exists? {exists}")

        if exists:
            logging.info("Found data in intermediate storage.")
            if self.file_type == "hdf5":
                hdf5_storage_loc = pathlib.Path(storage_dir) / pathlib.Path(
                    "cache_file.hdf5"
                )
                with h5py.File(str(hdf5_storage_loc), "r") as f:
                    logging.info(
                        f"Found HDF5 file with {len(list(f.keys()))} tensors."
                    )
                    for key in f.keys():
                        yield torch.from_numpy(f[key][()])
            else:
                data = get_all_data_paths(storage_dir)
                for path in data:
                    yield self.load(path)
        else:
            logging.info("Did not find data in intermediate storage.")
            logging.info(
                "Reading raw data and storing in intermediate storage."
            )
            storage_dir.mkdir(parents=True)
            counter = 0

            # TODO: Clean this
            # TODO: Make saving asynch
            for image in self.datapipe:
                file_name = f"tensor_{counter}"
                file_path = storage_dir / pathlib.Path(file_name)
                counter += 1
                if self.file_type == "hdf5":
                    self.save(image, str(file_name))
                else:
                    self.save(image, str(file_path))
                yield image
