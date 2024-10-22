import numpy as np
from PIL import Image
import glob
import pathlib
import pickle
import shutil
import tempfile
import torch
from typing import Any, List
from cedar.config import CedarContext
from cedar.compose import Feature
from cedar.client import DataSet
from cedar.pipes import NoopPipe, Pipe, MapperPipe
from cedar.sources import IterSource
from cedar.pipes.optimize.io import ObjectDiskCachePipe
from pathlib import Path


# FOR DISK CACHE PIPE TESTS WITH ONE PROCESS
class CacheTestHelper:
    def __init__(self, file_type: str, data: List[int]):
        all_file_types = ["pkl", "pt"]
        assert file_type in all_file_types

        self.file_type = file_type
        self.data = data

    def setup(self, regular) -> None:
        file_type = self.file_type

        class TestFeature(Feature):
            def _compose(self, source_pipes: List[Pipe]):
                ft = source_pipes[0]
                ft = NoopPipe(ft)
                ft = ObjectDiskCachePipe(ft, file_type)
                ft = NoopPipe(ft)
                return ft

        class TestMapperFeature(Feature):
            def _add_thousand(self, x: int) -> int:
                return x + 1000

            def _compose(self, source_pipes: List[Pipe]):
                ft = source_pipes[0]
                ft = NoopPipe(ft)
                ft = ObjectDiskCachePipe(ft, file_type)
                ft = MapperPipe(ft, self._add_thousand)
                ft = NoopPipe(ft)
                return ft

        source = IterSource(self.data)

        if regular:
            test_feature = TestFeature()
        else:
            test_feature = TestMapperFeature()

        test_feature.apply(source)

        ctx = CedarContext()

        self.dataset = DataSet(
            ctx, {"feature": test_feature}, enable_controller=False
        )

    def _read_files(self, file_names: str) -> List[int]:
        saved_data = []

        for file in file_names:
            if self.file_type == "pkl":
                with open(file, "rb") as f:
                    loaded_data = pickle.load(f)
            elif self.file_type == "pt":
                loaded_data = torch.load(file)
            else:
                raise Exception("No file type specified")
            saved_data.extend(loaded_data)

        return saved_data

    def _check_directory_and_files(self, cache_dir: pathlib.Path) -> List[str]:
        # Check whether directory was created
        assert cache_dir.exists()

        # Check whether directory not empty
        file_names = glob.glob(str(cache_dir) + "/*")
        assert len(file_names) > 0

        # Check whether all cached files have same format
        file_suffix = "pkl"
        if self.file_type == "pt":
            file_suffix = "pt"

        suffix_file_names = glob.glob(str(cache_dir) + "/*." + file_suffix)
        assert len(suffix_file_names) == len(file_names)

        return file_names

    def _get_cache_dirs(self) -> List[Path]:
        temp_dir = pathlib.Path(tempfile.gettempdir())
        prefix = "cedar_"
        return [
            d
            for d in temp_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ]
        return [
            d
            for d in temp_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ]

    def _remove_cache_dirs(self) -> None:
        cache_dirs = self._get_cache_dirs()
        if len(cache_dirs) > 0:
            for directory in cache_dirs:
                shutil.rmtree(str(directory))

    def _check_read_vs_output(self) -> None:
        cache_dirs = self._get_cache_dirs()

        assert len(cache_dirs) == 1

        file_names = self._check_directory_and_files(cache_dirs[0])

        # Check whether files in directory contain the same dataset
        saved_data = self._read_files(file_names)

        # Reconstruct samples
        saved_data_unpacked = [x.data for x in saved_data]

        assert sorted(saved_data_unpacked) == sorted(self.data)

        # Remove cached files
        shutil.rmtree(cache_dirs[0])

    # Should be called before executing check_read
    def cache(self) -> None:
        self._remove_cache_dirs()

        # Check outputs
        out = []
        for x in self.dataset:
            out.append(x)
        assert out == self.data

        cache_dirs_after = self._get_cache_dirs()

        assert len(cache_dirs_after) > 0

    def check_regular(self) -> None:
        self._remove_cache_dirs()

        # Check outputs
        out = []
        for x in self.dataset:
            out.append(x)
        assert out == self.data

        self._check_read_vs_output()

    def check_non_regular(self, expected_cached, expected_out) -> None:
        self._remove_cache_dirs()

        # Check outputs
        out = []
        for x in self.dataset:
            out.append(x)
        assert out == expected_out

        self._check_read_vs_output()

    def check_break(self) -> None:
        # NOTE: ASSUMES 1000 samples per saved file

        self._remove_cache_dirs()

        # Check outputs
        out = []
        half_size = len(self.data) // 2
        counter = 0
        for x in self.dataset:
            out.append(x)
            counter += 1
            if counter == half_size:
                break

        # Delete dataset to trigger 'finally' statement
        # Should now write the last batch to disk
        # NOTE: This does not work as envisioned
        self.dataset._del_iter()

        # If we have more than 1000 samples, then number of written samples
        # should be half of total samples. If number of total samples is less
        # than 1000, then all samples should be written to disk.
        num_out = half_size
        if len(self.data) < 1000:
            num_out = len(self.data)

        assert out == self.data[:half_size]

        cache_dirs = self._get_cache_dirs()

        assert len(cache_dirs) == 1

        file_names = self._check_directory_and_files(cache_dirs[0])

        saved_data = self._read_files(file_names)

        # Reconstruct samples
        saved_data_unpacked = [x.data for x in saved_data]

        assert sorted(saved_data_unpacked) == sorted(self.data[:num_out])

        # Remove cached files
        shutil.rmtree(str(cache_dirs[0]))

    # Should be called as follows: setup, cache, setup, check_read
    def check_read(self) -> None:
        cache_dirs = self._get_cache_dirs()

        assert len(cache_dirs) == 1

        # Check outputs
        out = []
        for x in self.dataset:
            out.append(x)
        assert sorted(out) == sorted(self.data)

        # Remove cached files
        shutil.rmtree(str(cache_dirs[0]))


# FOR DISK CACHE PIPE TEST MULTIPROCESS
class CacheTestHelperMultiprocess:
    def __init__(self, file_type: str, data: List[int], num_processes: int):
        all_file_types = ["pkl", "pt"]
        assert file_type in all_file_types

        self.file_type = file_type
        self.data = data
        self.num_processes = num_processes
        # NOTE: The amount of features should correspond
        # to the amount of processes

    def setup(self, regular) -> None:
        file_type = self.file_type

        class TestFeature(Feature):
            def _compose(self, source_pipes: List[Pipe]):
                ft = source_pipes[0]
                ft = NoopPipe(ft)
                ft = ObjectDiskCachePipe(ft, file_type)
                ft = NoopPipe(ft)
                return ft

        class TestMapperFeature(Feature):
            def _add_thousand(self, x: int) -> int:
                return x + 1000

            def _compose(self, source_pipes: List[Pipe]):
                ft = source_pipes[0]
                ft = NoopPipe(ft)
                ft = ObjectDiskCachePipe(ft, file_type)
                ft = MapperPipe(ft, self._add_thousand)
                ft = NoopPipe(ft)
                return ft

        feats = {}
        for i in range(self.num_processes):
            if regular:
                test_feature = TestFeature()
            else:
                test_feature = TestMapperFeature()

            source = IterSource(self.data, rank_spec=(self.num_processes, i))

            test_feature.apply(source)
            feats[str(i)] = test_feature

        print(feats)
        ctx = CedarContext()

        self.dataset = DataSet(
            ctx,
            feats,
            enable_controller=False,
            iter_mode="mp",
            prefetch=False,
            enable_optimizer=False,
        )

    def _get_cache_dirs(self) -> List[Path]:
        temp_dir = pathlib.Path(tempfile.gettempdir())
        prefix = "cedar_"
        return [
            d
            for d in temp_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ]
        return [
            d
            for d in temp_dir.iterdir()
            if d.is_dir() and d.name.startswith(prefix)
        ]

    def _remove_cache_dirs(self) -> None:
        cache_dirs = self._get_cache_dirs()
        if len(cache_dirs) > 0:
            for directory in cache_dirs:
                shutil.rmtree(str(directory))

    def _read_files(self, file_names: str) -> List[int]:
        saved_data = []

        for file in file_names:
            if self.file_type == "pkl":
                with open(file, "rb") as f:
                    loaded_data = pickle.load(f)
            elif self.file_type == "pt":
                loaded_data = torch.load(file)
            else:
                raise Exception("No file type specified")
            saved_data.extend(loaded_data)

        return saved_data

    """
    Cache the data, check for expected outputs,
    and for directories being created.
    """

    def cache(self, clean: bool = True, expected_data: List[int] = None):
        self._remove_cache_dirs()

        # Check outputs
        out = []
        for x in self.dataset:
            out.append(x)
        if expected_data:
            assert sorted(out) == sorted(expected_data)
        else:
            assert sorted(out) == sorted(self.data)

        cache_dirs_after = self._get_cache_dirs()

        assert len(cache_dirs_after) == self.num_processes

        if clean:
            self._remove_cache_dirs()

    """
    Check the data written to disk. Should be called after a call to cache.
    """

    def check_write(self, clean: bool = True, expected_data: List[int] = None):
        cache_dirs = self._get_cache_dirs()
        assert len(cache_dirs) == self.num_processes

        all_data = []
        for cache_dir in cache_dirs:
            file_names = glob.glob(str(cache_dir) + "/*")
            all_data.extend(self._read_files(file_names))

        saved_data_unpacked = [x.data for x in all_data]

        if expected_data:
            assert sorted(saved_data_unpacked) == sorted(expected_data)
        else:
            assert sorted(saved_data_unpacked) == sorted(self.data)

        if clean:
            self._remove_cache_dirs()

    """
    Check whether the data gets read from disk correctly.
    Should be called after a call to cache and then check_read.
    """

    def check_read(
        self, clean: bool = True, expected_data: List[int] = None
    ) -> None:
        cache_dirs = self._get_cache_dirs()
        print(cache_dirs)
        assert len(cache_dirs) == self.num_processes

        out = []
        for x in self.dataset:
            out.append(x)

        if expected_data:
            assert sorted(out) == sorted(expected_data)
        else:
            assert sorted(out) == sorted(self.data)

        if clean:
            self._remove_cache_dirs()


# FOR WEB READER PIPE TESTS
class WebReaderHelper:
    def __init__(self):
        self.all_urls = [
            "https://farm4.staticflickr.com/3376/3298850144_463e6f4917_z.jpg",
            "https://farm5.staticflickr.com/4102/4888234256_538b8dee56_z.jpg",
            "https://farm9.staticflickr.com/8316/7976335813_975d3fd4a7_z.jpg",
            "https://farm4.staticflickr.com/3132/2632337218_12b348974b_z.jpg",
            "https://farm3.staticflickr.com/2770/4268231981_d3579025d4_z.jpg",
            "https://farm4.staticflickr.com/3749/9491571342_f288b24d57_z.jpg",
            "https://farm8.staticflickr.com/7091/7082396979_9df0137e22_z.jpg",
            "https://farm7.staticflickr.com/6019/5913927953_292b0761e5_z.jpg",
            "https://farm1.staticflickr.com/187/368235957_b5d42708d1_z.jpg",
            "https://farm8.staticflickr.com/7214/6861986830_34473988c3_z.jpg",
            "https://farm3.staticflickr.com/2840/10078057985_984381cbf4_z.jpg",
        ]
        self.images = []

    def get_urls(self):
        return self.all_urls

    def read_images(self):
        test_dir = pathlib.Path(__file__).resolve().parents[0]
        file_names = glob.glob(str(test_dir) + "/data/images/*")
        for image_file in file_names:
            self.images.append(Image.open(image_file))

    def _sort_image_list(self, images: List[Any]):
        # sort by top left pixel RGB values, width and height
        return sorted(
            images,
            key=lambda x: (sum(x.getpixel((0, 0))), x.size[0], x.size[1]),
        )

    def compare(self, out_imgs: List[Any]):
        assert len(self.images) == len(out_imgs)
        out_imgs = self._sort_image_list(out_imgs)
        self.images = self._sort_image_list(self.images)
        for i in range(len(self.images)):
            out_img = np.array(out_imgs[i])
            real_img = np.array(self.images[i])
            assert out_img.shape == real_img.shape
            assert np.array_equal(out_img, real_img)


def check_fault_tolerance_funcs(
    pipe: Pipe,
    expected_samples_in_partition: int = 1000,
    expected_num_in_flight_partitions: int = 0,
    expected_num_fully_sent_partitions: int = 0,
    expected_starting_sample_id: int = 0,
    expected_end_sample_id: int = 0,
) -> None:
    """
    Helper function to test individual fault tolerance functions.
    The function checks whether the pipe variant of the pipe correctly
    outputs the expected samples in a partition, the expected number
    of in-flight partitions after the amount of iterations done over
    the pipe. The function also checks for the expected starting sample id.
    The last partition is then sealed and then we check for the expected
    end sample id.

    Expected state at the start of the function:
    - Should be invoked after we have iterated through the pipe and
    after the in-flight partitions have not yet been marked as sealed.
    - Number of sealed partitions should be 0

    Args:
    - pipe (Pipe): SourcePipe object.
    - expected_samples_in_partition (int): number of samples per partition
        in this pipe's pipe variant.
    - expected_num_in_flight_partitions (int): number of in flight partitions
        of pipe.pipe_variant when this function is invoked.
    - expected_num_fully_sent_partitions (int): number of fully sent partitions
        of pipe.pipe_variant when this function is invoked.
    - expected_starting_sample_id (int): first sample ID of all sample IDs
        generated by this pipe.pipe_variant.
    - expected_end_sample_id (int): last sample ID of all sample IDs generated
        by this pipe variant (equivalent to the last sample ID of the last
        partition).
    """

    # Test pipe variant state after iteration
    assert (
        pipe.pipe_variant.get_num_samples_in_partition()
        == expected_samples_in_partition
    )

    # Make sure that the only the last partition is in flight
    assert expected_num_in_flight_partitions <= 1
    in_flight_partitions = pipe.pipe_variant.get_in_flight_partitions()
    assert len(in_flight_partitions) == expected_num_in_flight_partitions

    fully_sent_partitions = pipe.pipe_variant.get_fully_sent_partitions()
    assert len(fully_sent_partitions) == expected_num_fully_sent_partitions

    sealed_partitions = pipe.pipe_variant.get_sealed_partitions()
    assert len(sealed_partitions) == 0

    # Test sealing all partitions except for last partition
    for i in range(expected_num_fully_sent_partitions):
        pipe.pipe_variant.seal_partition(i)
    assert len(pipe.pipe_variant.get_fully_sent_partitions()) == 0
    assert (
        len(pipe.pipe_variant.get_sealed_partitions())
        == expected_num_fully_sent_partitions
    )

    # Test sealing last partition if the last partition has not been processed
    if len(pipe.pipe_variant.get_in_flight_partitions()) > 0:
        pipe.pipe_variant.seal_last_partition()

    sealed_partitions = pipe.pipe_variant.get_sealed_partitions()
    assert (
        len(sealed_partitions)
        == expected_num_fully_sent_partitions
        + expected_num_in_flight_partitions
    )
    counter = 0
    for key in sorted(sealed_partitions.keys()):
        assert key == counter
        counter += 1

    if len(sealed_partitions) > 0:
        assert (
            sealed_partitions[0].starting_sample_id
            == expected_starting_sample_id
        )
        assert (
            sealed_partitions[len(sealed_partitions) - 1].end_sample_id
            == expected_end_sample_id
        )


def get_checkpoint_dir():
    """
    Returns the cedar checkpoint dir as pathlib.Path.
    """
    temp_dir = tempfile.gettempdir()
    checkpoint_dir = pathlib.Path(temp_dir) / pathlib.Path(
        "cedar_checkpointing"
    )
    return checkpoint_dir


def get_checkpoint_file():
    """
    Returns the cedar checkpoint file as pathlib.Path.
    """
    checkpoint_dir = get_checkpoint_dir()
    checkpoint_file = checkpoint_dir / pathlib.Path("cedar_checkpoint.pkl")
    return checkpoint_file


def read_data_from_checkpoint_file():
    """
    Reads data from the expected checkpoint file.
    If file does not exists, fails with assertion error.
    """
    checkpoint_file = get_checkpoint_file()

    assert pathlib.Path.exists(checkpoint_file)

    with open(str(checkpoint_file), "rb") as file:
        loaded_data = pickle.load(file)

    return loaded_data


def remove_checkpoint_dir():
    """
    Removes checkpoint dir.
    """
    checkpoint_dir = get_checkpoint_dir()
    shutil.rmtree(str(checkpoint_dir))


def check_sealed_partition_dict_correctness(
    partitions, num_samples_per_partition, num_total_samples
):
    """
    Checks whether partition dict is formatted correctly.
    Expectation: {partition_index : Partition}, where partition_index
    starts at 0 and includes all necessary partitions generate for
    num_samples_per_partition. Checks for each partition containing
    the correct start and end sample id.

    Args:
    - partitions (Dict[int, Partition]): Map from partition index to Partition.
    - num_samples_per_partition (int): Number of samples for each partition.
    - num_total_samples (int): Number of total samples that should be present.
    """
    num_partitions = num_total_samples // num_samples_per_partition
    if num_total_samples % num_samples_per_partition != 0:
        num_partitions += 1

    # Check for all partition indexes being included
    partitions = partitions["sealed"]
    expected_partitions = {x for x in range(num_partitions)}
    for partition_id in sorted(partitions.keys()):
        assert partition_id in expected_partitions
        expected_partitions.remove(partition_id)

    assert len(expected_partitions) == 0

    # Check for partition ranges
    curr_start = 0
    curr_end = num_samples_per_partition - 1
    if curr_end + 1 > num_total_samples:
        curr_end == num_total_samples - 1
    for partition_id in sorted(partitions.keys()):
        curr_partition = partitions[partition_id]
        assert partition_id == curr_partition.partition_id
        assert curr_partition.starting_sample_id == curr_start
        assert curr_partition.end_sample_id == curr_end
        curr_start += num_samples_per_partition
        curr_end += num_samples_per_partition
        # Handle last sealed case
        if curr_end + 1 >= num_total_samples:
            curr_end = num_total_samples - 1


def check_full_dict_correctness(
    partitions,
    num_total_samples,
    num_samples_per_partition,
    num_in_flight_partitions,
    num_fully_sent_partitions,
    num_sealed_partitions,
):
    in_flight_partitions = partitions["in-flight"]
    fully_sent_partitions = partitions["fully-sent"]
    sealed_partitions = partitions["sealed"]

    assert len(in_flight_partitions) == num_in_flight_partitions
    assert len(fully_sent_partitions) == num_fully_sent_partitions
    assert len(sealed_partitions) == num_sealed_partitions


def check_replay_files_image_folder(
    outputs: List[str],
    before_replay: List[str],
    num_samples_in_partition: int = 2,
    only_in_flight_lost: bool = False,
    expect_no_duplicated=False,
) -> None:
    test_dir = pathlib.Path(__file__).resolve().parents[0]
    path_to_files = pathlib.Path(test_dir) / "data/images"
    expected_outputs = list(glob.glob(str(path_to_files) + "/*"))

    for output in outputs:
        assert output in expected_outputs

    dedepublicated_list = sorted(list(set(outputs)))
    assert dedepublicated_list == sorted(expected_outputs)

    if not expect_no_duplicated:
        if only_in_flight_lost:
            final_list = expected_outputs
            num_elems_from_back = len(before_replay) % num_samples_in_partition
            if num_elems_from_back:
                final_list.extend(before_replay[-num_elems_from_back:])
        else:
            non_dup_list = [
                item for item in expected_outputs if item not in before_replay
            ]
            final_list = before_replay
            final_list.extend(before_replay)
            final_list.extend(non_dup_list)
    else:
        final_list = expected_outputs

    assert sorted(final_list) == sorted(outputs)


def check_line_source_replay_output(
    replay_line_start: int, replay_line_end: int, output: List[str]
) -> None:
    """
    Checks that the post-replay output is the same as expected
    given the first replayed line and the last.
    """
    test_dir = pathlib.Path(__file__).resolve().parents[0]
    path_to_files = pathlib.Path(test_dir) / "data/test_text_2.txt"
    full_output = []
    replayed_output = []
    with open(path_to_files, mode="rb") as file:
        counter = 0
        for line in file:
            line = line.decode("utf-8")
            line = line.strip("\r\n")
            full_output.append(line)
            if replay_line_start is not None:
                if counter >= replay_line_start and counter <= replay_line_end:
                    replayed_output.append(line)
            counter += 1

    if replay_line_start is not None:
        expected_output = (
            full_output[: replay_line_end + 1]
            + replayed_output
            + full_output[replay_line_end + 1 :]  # noqa: E203
        )
    else:
        expected_output = full_output

    assert expected_output == output
