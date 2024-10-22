import pathlib
import filecmp
import pytest
from typing import List
from cedar.config import CedarContext
from cedar.compose import Feature
from cedar.client import DataSet
from cedar.pipes import NoopPipe, Pipe, PipeVariantType, MapperPipe
from cedar.sources import IterSource


def test_basic_feature(basic_noop_dataset_prefetch):
    out = []
    for x in basic_noop_dataset_prefetch:
        out.append(x)

    assert out == [1, 2, 3]

    assert (
        len(basic_noop_dataset_prefetch.features["feature"].logical_pipes) == 4
    )


@pytest.mark.skip()  # TODO: FIXME
def test_viz(basic_noop_dataset_prefetch):
    test_dir = pathlib.Path(__file__).resolve().parents[0]

    basic_noop_dataset_prefetch.viz_logical_plan(str(test_dir))
    basic_noop_dataset_prefetch.viz_physical_plan(str(test_dir))

    out = []

    for x in basic_noop_dataset_prefetch:
        out.append(x)

    assert out == [1, 2, 3]


def test_to_yaml(basic_noop_dataset):
    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_config_file = pathlib.Path(test_dir) / "data/config_ref.yml"
    test_config_file = pathlib.Path(test_dir) / "data/config_output.yml"

    basic_noop_dataset.features["feature"].to_yaml(str(test_config_file))

    assert filecmp.cmp(str(ref_config_file), str(test_config_file))


def test_from_yaml():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_config_file = pathlib.Path(test_dir) / "data/config_ref.yml"

    ctx = CedarContext()
    test_feature.load_from_yaml(ctx, str(ref_config_file))

    assert len(test_feature.physical_adj_list) == 4

    assert (
        test_feature.physical_pipes[0].pipe_variant_type
        == PipeVariantType.INPROCESS
    )
    assert (
        test_feature.physical_pipes[1].pipe_variant_type
        == PipeVariantType.INPROCESS
    )
    assert (
        test_feature.physical_pipes[2].pipe_variant_type
        == PipeVariantType.INPROCESS
    )
    assert (
        test_feature.physical_pipes[3].pipe_variant_type
        == PipeVariantType.INPROCESS
    )


def test_from_yaml_variant():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_config_file = pathlib.Path(test_dir) / "data/config_ref_variant.yml"

    ctx = CedarContext()
    test_feature.load_from_yaml(ctx, str(ref_config_file))

    assert len(test_feature.physical_adj_list) == 4

    assert (
        test_feature.physical_pipes[0].pipe_variant_type
        == PipeVariantType.INPROCESS
    )
    assert (
        test_feature.physical_pipes[1].pipe_variant_type == PipeVariantType.SMP
    )
    assert (
        len(
            test_feature.physical_pipes[
                1
            ].pipe_variant.variant_ctx.service.procs
        )
        == 10
    )
    assert (
        test_feature.physical_pipes[2].pipe_variant_type
        == PipeVariantType.INPROCESS
    )
    assert (
        test_feature.physical_pipes[3].pipe_variant_type
        == PipeVariantType.INPROCESS
    )


def _add_one(x):
    return x + 1


def _double(x):
    return x * 2


def test_reordering():
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = MapperPipe(ft, _add_one)
            ft = MapperPipe(ft, _double)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    dataset = DataSet(
        ctx, {"feature": test_feature}, prefetch=False, enable_controller=False
    )
    out = []

    for x in dataset:
        out.append(x)

    assert out == [4, 6, 8]

    # Flip order
    _, phys_plan = dataset.get_plan()["feature"]

    # original: source (5) -> noop (4) -> add (3) -> double (2) -> noop (1)
    #       -> noop (0)
    # new:  source (5) -> noop (4) -> double (2) -> noop (0) -> add (3)
    #       -> noop (1)

    phys_plan["graph"][4] = "2"
    phys_plan["graph"][2] = "0"
    phys_plan["graph"][0] = "3"
    phys_plan["graph"][3] = "1"
    phys_plan["graph"][1] = ""

    dataset.reset_feature("feature")
    dataset.load_feature_from_dict("feature", phys_plan)

    # ID 1 should now be the output
    assert (
        dataset.features["feature"].output_pipe.get_logical_uname()
        == "NoopPipe_1"
    )

    out = []

    for x in dataset:
        out.append(x)

    assert out == [3, 5, 7]


def test_reordering_cycles():
    # Make sure we can catch cycles when given a bad plan
    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = MapperPipe(ft, _add_one)
            ft = MapperPipe(ft, _double)
            ft = NoopPipe(ft)
            ft = NoopPipe(ft)

            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()

    dataset = DataSet(
        ctx, {"feature": test_feature}, prefetch=False, enable_controller=False
    )

    # Flip order
    _, phys_plan = dataset.get_plan()["feature"]

    # original: source (5) -> noop (4) -> add (3) -> double (2) -> noop (1)
    #       -> noop (0)

    phys_plan["graph"][0] = "4"

    dataset.reset_feature("feature")
    with pytest.raises(RuntimeError):
        dataset.load_feature_from_dict("feature", phys_plan)


def test_tracing(basic_noop_dataset):
    basic_noop_dataset._return_datasample = True

    it = iter(basic_noop_dataset)

    ds = next(it)
    assert ds.trace_dict is not None
    assert len(ds.trace_dict) == 5  # includes dummy source pipe
    assert (
        ds.trace_dict[0] > ds.trace_dict[1]
        and ds.trace_dict[1] > ds.trace_dict[2]
        and ds.trace_dict[2] > ds.trace_dict[3]
        and ds.trace_dict[3] > ds.trace_dict[-1]
    )
    ds = next(it)
    assert ds.trace_dict is None
    ds = next(it)
    assert ds.trace_dict is None


def test_tracing_batch(basic_noop_batch_dataset):
    basic_noop_batch_dataset._return_datasample = True

    it = iter(basic_noop_batch_dataset)

    ds = next(it)
    assert ds.trace_dict is not None
    assert len(ds.trace_dict) == 6
    assert (
        ds.trace_dict[0] > ds.trace_dict[1]
        and ds.trace_dict[1] > ds.trace_dict[2]
        and ds.trace_dict[2] > ds.trace_dict[3]
        and ds.trace_dict[3] > ds.trace_dict[4]
        and ds.trace_dict[4] > ds.trace_dict[-1]
    )
    assert ds.data == [1, 2, 3]
    assert ds.size_dict == {-1: 1, 4: 1, 3: 1, 2: 1, 1: 1, 0: 3}
    assert ds.trace_order == [-1, 4, 3, 2, 1, 0]

    ds = next(it)
    assert ds.data == [4, 5]


def test_get_batch_size(basic_noop_batch_dataset):
    assert basic_noop_batch_dataset.features["feature"].get_batch_size() == 3


def test_trace_profile(basic_noop_feature):
    ctx = CedarContext()
    it = iter(basic_noop_feature.profile(ctx))

    ds = next(it)
    assert ds.data_size_dict is not None
    assert len(ds.data_size_dict) == 4
    for k, v in ds.data_size_dict.items():
        assert v == 32
    assert ds.data == 1

    ds = next(it)
    assert ds.data == 2

    ds = next(it)
    assert ds.data == 3

    with pytest.raises(StopIteration):
        ds = next(it)


def test_sizeof():
    import torch
    import numpy as np
    from cedar.pipes.common import get_sizeof_data

    # N
    assert get_sizeof_data(None) == 0
    assert get_sizeof_data(torch.rand(100, 100)) == 40000
    assert get_sizeof_data(np.random.rand(100, 100)) >= 80000

    l_tensor = [torch.rand(5, 5, 5), torch.rand(2, 3, 3), torch.rand(3, 3, 3)]
    assert get_sizeof_data(l_tensor) == 680


def test_optimize_reorder():
    import yaml
    from cedar.compose.utils import calculate_reorderings

    class TestFeature(Feature):
        def _compose(self, source_pipes: List[Pipe]):
            ft = source_pipes[0]  # 3
            ft = NoopPipe(ft)  # 2
            ft = NoopPipe(ft)  # 1
            ft = NoopPipe(ft)  # 0
            return ft

    data = [1, 2, 3]
    source = IterSource(data)

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_config_file = pathlib.Path(test_dir) / "data/test_profile_stats.yml"

    test_feature = TestFeature()
    test_feature.apply(source)

    optimizer = test_feature.optimizer
    with ref_config_file.open("r") as f:
        optimizer.profiled_stats = yaml.safe_load(f)
    optimizer._init_stats()

    candidate_plans = calculate_reorderings(
        optimizer.logical_pipes, optimizer.physical_plan.graph
    )
    assert len(candidate_plans) == 6

    # Pipes have equal latencies. Base cost of 100
    # Pipe 2 grows size by 10x, pipe 1 constant, pipe 0 shrinks by 10x
    # Ideal ordering is 3,0,1,2
    optimal_plan, cost = optimizer._find_optimal_reordering(candidate_plans)

    assert optimal_plan == {
        3: {0},
        0: {1},
        1: {2},
        2: set(),
    }
    assert cost == 30.25
