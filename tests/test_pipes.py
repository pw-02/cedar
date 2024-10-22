import functools
import numpy as np
import pathlib
from typing import List
from PIL import Image
import pytest
from io import BytesIO

from cedar.config import CedarContext
from cedar.compose import Feature
from cedar.client import DataSet
from cedar.pipes import (
    BatcherPipe,
    FileOpenerPipe,
    LineReaderPipe,
    MapperPipe,
    NoopPipe,
    Pipe,
    PipeVariantType,
    ImageReaderPipe,
    WebReaderPipe,
    MultithreadedPipeVariantContext,
)
from cedar.sources import LocalFSSource, IterSource

from .utils import WebReaderHelper


def test_map_pipe():
    def add(x, y):
        return x + y

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            add_two = functools.partial(add, y=2)
            ft = source_pipes[0]
            ft = MapperPipe(ft, add_two)
            ft = MapperPipe(ft, add_two)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    dataset = DataSet(
        CedarContext(), {"feature": test_feature}, enable_controller=False
    )
    out = []

    for x in dataset:
        out.append(x)

    assert out == [5, 6, 7]


def test_file_opener_pipe():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]) -> Pipe:
            fp = source_pipes[0]
            fp = FileOpenerPipe(fp)
            fp = LineReaderPipe(fp)
            return fp

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    test_file = pathlib.Path(test_dir) / "data/test_text.txt"
    pipe = TestFeature()
    pipe.apply(LocalFSSource(str(test_file)))
    res = pipe.load(CedarContext())

    data = []
    for x in res:
        data.append(x.data)

    assert data == ["hello", "world"]


def test_batcher_pipe():
    data = [1, 2, 3, 4, 5, 6, 7]
    source = IterSource(data)
    ctx = CedarContext()

    source_pipe = source.to_pipe()
    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)

    pipe = BatcherPipe(source_pipe, 3, False)
    pipe.mutate(ctx, PipeVariantType.INPROCESS)

    it = pipe.get_variant()

    out = []
    for x in it:
        out.append(x.data)

    assert out == [[1, 2, 3], [4, 5, 6], [7]]


def test_image_reader_pipe():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]) -> Pipe:
            fp = source_pipes[0]
            fp = ImageReaderPipe(fp)
            return fp

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    test_file = pathlib.Path(test_dir) / "data/example_image.jpeg"
    pipe = TestFeature()
    pipe.apply(LocalFSSource(str(test_file)))
    res = pipe.load(CedarContext())
    out = []
    for x in res:
        out.append(x.data)

    assert np.asarray(out[0]).shape == (3, 4, 4)
    # TODO: fix actual check, flaky across packages
    # expected_output = [
    #     [
    #         [151, 145, 107, 83],
    #         [86, 158, 176, 147],
    #         [55, 157, 189, 139],
    #         [22, 104, 137, 115],
    #     ],
    #     [
    #         [182, 150, 115, 138],
    #         [90, 140, 163, 183],
    #         [39, 123, 165, 171],
    #         [14, 80, 131, 176],
    #     ],
    #     [
    #         [177, 127, 64, 73],
    #         [102, 130, 121, 121],
    #         [84, 140, 137, 106],
    #         [99, 130, 119, 98],
    #     ],
    # ]
    # expected_output = np.asarray(expected_output, dtype=np.uint8)
    # real_output = np.array(out[0], dtype=np.uint8)
    # print(real_output)
    # comparison = real_output == expected_output
    # assert comparison.all()


def test_image_reader_pipe_threaded():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]) -> Pipe:
            fp = source_pipes[0]
            fp = ImageReaderPipe(fp)
            fp = NoopPipe(fp)
            return fp

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    test_file = pathlib.Path(test_dir) / "data/example_image.jpeg"
    pipe = TestFeature()
    pipe.apply(LocalFSSource(str(test_file)))
    _ = pipe.load(CedarContext())

    print(pipe.physical_pipes[1].get_logical_name())
    image_reader_pipe = pipe.physical_pipes[1]

    image_reader_pipe.dynamic_mutate(
        PipeVariantType.MULTITHREADED, pipe.physical_pipes[0]
    )

    res = image_reader_pipe.pipe_variant
    assert image_reader_pipe.pipe_variant_type == PipeVariantType.MULTITHREADED

    out = []
    for x in res:
        out.append(x.data)

    assert np.asarray(out[0]).shape == (3, 4, 4)
    # expected_output = [
    #     [
    #         [151, 145, 107, 83],
    #         [86, 158, 176, 147],
    #         [55, 157, 189, 139],
    #         [22, 104, 137, 115],
    #     ],
    #     [
    #         [182, 150, 115, 138],
    #         [90, 140, 163, 183],
    #         [39, 123, 165, 171],
    #         [14, 80, 131, 176],
    #     ],
    #     [
    #         [177, 127, 64, 73],
    #         [102, 130, 121, 121],
    #         [84, 140, 137, 106],
    #         [99, 130, 119, 98],
    #     ],
    # ]
    # expected_output = np.asarray(expected_output, dtype=np.uint8)
    # real_output = np.array(out[0], dtype=np.uint8)
    # comparison = real_output == expected_output
    # assert comparison.all()


@pytest.mark.parametrize("mutation_type", ["inprocess", "multithreaded"])
def test_web_reader_pipe(mutation_type: str, mocked_responses):
    tester = WebReaderHelper()

    # Set up requests mocking
    urls = tester.all_urls
    data_dir = pathlib.Path(__file__).resolve().parents[0]
    data_dir = data_dir / pathlib.Path("data/images")
    for idx, url in enumerate(urls):
        image_dir = data_dir / pathlib.Path(f"image_{idx+1}.jpg")
        with open(str(image_dir), "rb") as image:
            image_bytes = image.read()
            mocked_responses.get(url=url, body=image_bytes)

    tester.read_images()
    urls = tester.get_urls()
    source = IterSource(urls)

    ctx = CedarContext()
    if mutation_type == "multithreaded":
        variant_ctx = MultithreadedPipeVariantContext(5)
    elif mutation_type == "inprocess":
        pass
    else:
        raise NotImplementedError(
            f"Specified wrong test param for WebReaderPipe.\
                Got: {mutation_type}"
        )

    source_pipe = source.to_pipe()
    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)

    pipe = WebReaderPipe(source_pipe, 5)
    if mutation_type == "inprocess":
        pipe.mutate(ctx, PipeVariantType.INPROCESS)
    elif mutation_type == "multithreaded":
        pipe.mutate(ctx, PipeVariantType.MULTITHREADED, variant_ctx)
    else:
        raise NotImplementedError(
            f"Specified wrong test param for WebReaderPipe.\
                Got: {mutation_type}"
        )

    it = pipe.get_variant()

    # Collect iterable output
    out = []
    for x in it:
        # check for URL returning
        assert x.data[0] in urls
        out.append(x.data[1])

    # Compare images using numpy
    out_imgs = []
    for bytestream in out:
        out_imgs.append(Image.open(BytesIO(bytestream)))

    tester.compare(out_imgs)
