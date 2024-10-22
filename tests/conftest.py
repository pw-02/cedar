import pytest
import ray
import logging
import pathlib
import responses
import time
from unittest.mock import patch, MagicMock
from cedar.config import CedarContext
from cedar.compose import Feature
from cedar.client import DataSet
from cedar.client.controller import ControllerThread
from cedar.client.profiler import FeatureProfiler
from cedar.pipes import (
    NoopPipe,
    BatcherPipe,
    MapperPipe,
    DataSample,
    PipeVariantType,
)
from cedar.pipes.optimize.prefetch import PrefetcherPipe
from cedar.sources import IterSource, LocalLineSource, LocalFSSource, source
from .utils import CacheTestHelper, CacheTestHelperMultiprocess
from cedar.compose import constants


@pytest.fixture(autouse=True)
def setup_logging():
    logging.basicConfig(level=logging.INFO)


@pytest.fixture
def setup_cache_helper_class(request):
    file_type, input, regular = request.param
    tester = CacheTestHelper(file_type, input)
    tester.setup(regular)
    return tester


@pytest.fixture
def setup_cache_helper_class_mp(request):
    file_type, input, regular, num_processes = request.param
    tester = CacheTestHelperMultiprocess(file_type, input, num_processes)
    tester.setup(regular)
    return tester


# NOTE: Have to install responses through pip (not conda)
@pytest.fixture
def mocked_responses():
    with responses.RequestsMock() as rsps:
        yield rsps


@pytest.fixture
def basic_noop_dataset_prefetch():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    return DataSet(ctx, {"feature": test_feature}, enable_controller=False)


@pytest.fixture
def basic_noop_dataset():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    return DataSet(
        ctx, {"feature": test_feature}, prefetch=False, enable_controller=False
    )


def _add_one(x):
    return x + 1


@pytest.fixture
def basic_map_dataset_long():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = MapperPipe(ft, _add_one)
            ft = MapperPipe(ft, _add_one)
            ft = NoopPipe(ft)

            return ft

    data = range(1000)
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    return DataSet(
        ctx, {"feature": test_feature}, prefetch=False, enable_controller=False
    )


@pytest.fixture
def basic_3map_dataset_long():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = MapperPipe(ft, _add_one)
            ft = MapperPipe(ft, _add_one)
            ft = MapperPipe(ft, _add_one)
            ft = NoopPipe(ft)

            return ft

    data = range(1000)
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    return DataSet(
        ctx, {"feature": test_feature}, prefetch=False, enable_controller=False
    )


@pytest.fixture
def basic_noop_batch_dataset():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = BatcherPipe(ft, batch_size=3)

            return ft

    data = [1, 2, 3, 4, 5]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    return DataSet(
        ctx, {"feature": test_feature}, prefetch=False, enable_controller=False
    )


@pytest.fixture
def basic_noop_batch_dataset_prefetch():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = BatcherPipe(ft, batch_size=3)

            return ft

    data = [1, 2, 3, 4, 5]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    return DataSet(ctx, {"feature": test_feature}, enable_controller=False)


@pytest.fixture
def basic_noop_feature():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    return test_feature


def _sleep(x):
    time.sleep(0.5)
    return x


@pytest.fixture
def noop_sleep_dataset():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            fp = source_pipes[0]
            fp = NoopPipe(fp)
            fp = MapperPipe(fp, _sleep)
            return fp

    data = range(5)
    source = IterSource(data)

    feature = TestFeature()
    feature.apply(source)
    ctx = CedarContext()

    dataset = DataSet(
        ctx, {"feature": feature}, prefetch=False, enable_controller=False
    )

    return dataset


@pytest.fixture
def prefetch_dataset():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            fp = source_pipes[0]
            fp = PrefetcherPipe(fp)
            fp = MapperPipe(fp, _sleep)
            return fp

    data = range(5)
    source = IterSource(data)

    feature = TestFeature()
    feature.apply(source)
    ctx = CedarContext()

    dataset = DataSet(ctx, {"feature": feature}, enable_controller=False)

    return dataset


@pytest.fixture
def noop_profiled_feature():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)
    test_feature = TestFeature()
    test_feature.apply(source)
    test_feature.load(CedarContext())

    profiler = FeatureProfiler(test_feature)

    ds = DataSample([])

    # source = 10s -> noop = 10s -> noop = 20s -> noop(size=2) = 5s
    ds.trace_dict = {
        -1: 10,
        3: 20,
        2: 30,
        1: 50,
        0: 60,
    }
    ds.trace_order = [-1, 3, 2, 1, 0]
    ds.size_dict = {
        -1: 1,
        3: 1,
        2: 1,
        1: 1,
        0: 2,
    }
    ds.do_trace = True
    ds.buffer_size_dict = {}

    profiler.update_ds(ds)

    return (test_feature, profiler)


@pytest.fixture(autouse=True)
def override_control_loop_period(monkeypatch):
    monkeypatch.setattr("cedar.client.controller.CONTROLLER_PERIOD_SEC", 0.01)


@pytest.fixture
def fully_iterated_over_iterable_source_pipe(request):
    """
    Creates an iterable source pipe that has been fully
    iterated over. request is a tuple containing the
    number of samples that should be considered and the
    number of samples per partition.
    """
    num_samples, num_samples_per_partition = request.param
    data = [i for i in range(num_samples)]
    source = IterSource(data)
    pipe = source.to_pipe()

    pipe.mutate(CedarContext(), PipeVariantType.INPROCESS)
    assert pipe.pipe_variant is not None
    pipe.pipe_variant.num_samples_in_partition = num_samples_per_partition
    assert (
        pipe.pipe_variant.get_num_samples_in_partition()
        == num_samples_per_partition
    )

    out = []

    # Test results
    for x in pipe.pipe_variant:
        out.append(x.data)
    assert out == data
    assert pipe.pipe_variant.num_yielded == len(data)

    first_sampled_id = 0
    last_sample_id = num_samples - 1
    total_in_flight_partitions = (
        1 if num_samples % num_samples_per_partition != 0 else 0
    )
    total_fully_sent_partitions = num_samples // num_samples_per_partition

    return (
        pipe,
        num_samples,
        num_samples_per_partition,
        total_in_flight_partitions,
        total_fully_sent_partitions,
        first_sampled_id,
        last_sample_id,
    )


@pytest.fixture
def noop_dataset_with_variable_input(request):
    """
    request contains an int describing the amount of data
    that should be ingested into the source.
    """

    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    num_samples = request.param
    data = [i for i in range(num_samples)]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    return DataSet(
        ctx, {"feature": test_feature}, prefetch=False, enable_controller=False
    )


@pytest.fixture
def noop_dataset_with_line_reader_source(request):
    """
    request contains an int describing the amount of data
    that should be ingested into the source.
    """

    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    path_to_file = pathlib.Path(test_dir) / "data/test_text_2.txt"
    source = LocalLineSource(str(path_to_file))

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    dataset = DataSet(
        ctx, {"feature": test_feature}, prefetch=False, enable_controller=False
    )
    dataset._create_dataset_iter()
    source_pipes = dataset._get_source_pipes()
    feature_names = dataset._get_feature_names()
    variant = source_pipes[feature_names[0]][0].pipe_variant
    variant.num_samples_in_partition = 2

    return (dataset, variant)


@pytest.fixture
def noop_dataset_with_file_source(request):
    """
    request contains an int describing the amount of data
    that should be ingested into the source.
    """

    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    path_to_files = pathlib.Path(test_dir) / "data/images/"
    source = LocalFSSource(str(path_to_files))

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    dataset = DataSet(
        ctx, {"feature": test_feature}, prefetch=False, enable_controller=False
    )

    dataset._create_dataset_iter()
    source_pipes = dataset._get_source_pipes()
    feature_names = dataset._get_feature_names()
    variant = source_pipes[feature_names[0]][0].pipe_variant
    variant.num_samples_in_partition = 2

    return (dataset, variant)


def bottleneck_generator():
    print("Yiedl 1")
    yield []
    print("Yiedl 1")
    yield [0]
    print("Yiedl 1")
    yield [0]
    print("Yiedl 1")
    while True:
        yield []


@pytest.fixture
def mocked_controller_dataset():
    mock_bottleneck = MagicMock(side_effect=bottleneck_generator())
    with patch.object(
        ControllerThread, "_calculate_bottleneck", mock_bottleneck
    ) as mock_method:

        class TestFeature(Feature):
            def _compose(self, source_pipes):
                ft = source_pipes[0]
                ft = NoopPipe(ft)
                return ft

        data = list(range(1000))
        source = IterSource(data)
        test_feature = TestFeature()
        test_feature.apply(source)
        test_feature.logical_pipes[0].pipe_spec.mutable_variants = [
            PipeVariantType.INPROCESS,
            PipeVariantType.MULTITHREADED,
            PipeVariantType.SMP,
        ]

        dataset = DataSet(
            CedarContext(),
            {"feature": test_feature},
            enable_controller=True,
            test_mode=True,
        )
        yield dataset
        mock_method.stop()


@pytest.fixture(scope="session")
def setup_ray():
    if not ray.is_initialized():
        ray.init(num_cpus=16)
    yield None
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def trace_every_sample():
    original_value = source.TRACE_FREQUENCY_SEC
    source.TRACE_FREQUENCY_SEC = 0
    yield
    source.TRACE_FREQUENCY_SEC = original_value


@pytest.fixture
def set_ray_parallelism():
    original_value = constants.RAY_AVAILABLE_PARALLELISM
    constants.RAY_AVAILABLE_PARALLELISM = 4
    yield
    constants.RAY_AVAILABLE_PARALLELISM = original_value
