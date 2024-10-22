from typing import Optional
from .registry import register_optimizer_pipe
from ..context import (
    PipeVariantType,
    InProcessPipeVariantContext,
)
from ..pipe import (
    Pipe,
)
from ..variant import (
    InProcessPipeVariant,
    PipeVariant,
)

from typing import List, Any

import copy
import glob
import logging
import os
import pathlib
import pickle
import tempfile
import torch


@register_optimizer_pipe("ObjectDiskCachePipe")
class ObjectDiskCachePipe(Pipe):
    """
    Given a pipe that yielding any type of object,
    saves the tensors to files in temporary directory on disk.

    Can save data to following format: .pt (tensors) and .pkl (any object).
    User should indicate the type of format in the file_type variable.
    """

    def __init__(
        self,
        input_pipe: Optional[Pipe] = None,
        file_type: str = "pkl",
        is_random: bool = False,
    ):
        if input_pipe:
            super().__init__(
                "ObjectDiskCachePipe", [input_pipe], is_random=is_random
            )
        else:
            super().__init__("ObjectDiskCachePipe", [], is_random=is_random)
        self.file_type = file_type

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        return InProcessObjectDiskCachePipeVariant(
            self.input_pipes[0].pipe_variant, file_type=self.file_type
        )


class InProcessObjectDiskCachePipeVariant(InProcessPipeVariant):
    def __init__(
        self,
        input_pipe_variant: PipeVariant,
        file_type: str = "pkl",
        max_samples_in_cache_file: int = 1000,
    ):
        super().__init__(input_pipe_variant)
        # NOTE: Could write method that check the data type of the input pipe
        # and chooses the file_type appropriately
        self.file_type = file_type
        self.save_funcs = {"pt": self._save_pt, "pkl": self._save_pkl}
        self.load_funcs = {"pt": self._load_pt, "pkl": self._load_pkl}
        self.max_samples_in_cache_file = max_samples_in_cache_file

        self._check_file_type()
        self.save_function = self.save_funcs[file_type]
        self.load_function = self.load_funcs[file_type]
        temp_dir = tempfile.gettempdir()
        pid = os.getpid()
        dir_name = "cedar_" + str(pid)
        self.cache_dir = pathlib.Path(temp_dir) / pathlib.Path(dir_name)

    def _check_file_type(self) -> None:
        if (
            self.file_type not in self.save_funcs
            or self.file_type not in self.load_funcs
        ):
            raise ValueError(
                f"Specified file type {self.file_type} is not supported for\
                    caching.Supported file types: {self.save_funcs.keys()}"
            )

    def _save_pt(self, tensors: List[torch.Tensor], file_name: str) -> None:
        torch.save(tensors, f"{file_name}.pt")

    def _load_pt(self, file_name: str) -> List[torch.Tensor]:
        return torch.load(file_name)

    def _save_pkl(self, data: List[Any], file_name: str) -> None:
        with open(f"{file_name}.pkl", "wb") as file:
            pickle.dump(data, file)

    def _load_pkl(self, file_name: str) -> List[Any]:
        with open(file_name, "rb") as file:
            loaded_data = pickle.load(file)
        return loaded_data

    def _save(self, data: List[Any], file_name: str) -> None:
        self.save_function(data, file_name)

    def _load(self, file_name: str) -> List[Any]:
        return self.load_function(file_name)

    def _save_given_count(self, data: List[Any], count: int) -> None:
        file_name = f"data_batch_{count}"
        file_path = self.cache_dir / pathlib.Path(file_name)
        self._save(data, file_path)

    def _iter_impl(self):
        # Check existance of caching directory
        pid = os.getpid()
        logging.info(f"Checking existence of caching directory for pid {pid}")
        cache_dir_exists = pathlib.Path.exists(self.cache_dir)
        logging.info(
            f"Caching directory exists for pid {pid}? {cache_dir_exists}"
        )

        # Read from caching directory if it exists
        if cache_dir_exists:
            all_files = list(glob.glob(str(self.cache_dir) + "/*"))
            for file in all_files:
                items = self._load(file)
                for item in items:
                    # NOTE: Cached item should be datasample
                    item.read_from_cache = True
                    yield item
        else:
            self.cache_dir.mkdir(parents=True)
            file_count = 0
            item_buffer = []
            try:
                for item in self.input_pipe_variant:
                    item_cache_copy = copy.deepcopy(item)
                    item_buffer.append(item_cache_copy)
                    if len(item_buffer) == self.max_samples_in_cache_file:
                        self._save_given_count(item_buffer, file_count)
                        item_buffer = []
                        file_count += 1
                    yield item
            finally:
                self._save_given_count(item_buffer, file_count)
