from cedar.config import CedarContext
import pathlib
from cedar.pipes import (
    MapperPipe,
    PipeVariantType,
    NoopPipe,
    RayPipeVariantContext,
    TFRayPipeVariantContext,
    TFOutputHint,
    TFTensorDontCare,
)
from cedar.compose import Feature
from cedar.sources import IterSource, TFLocalLineSource
from cedar.pipes.optimize import FusedOptimizerPipe
import tensorflow as tf
import pytest


def _add_one(x):
    return x + 1


def test_map_pipe():
    data = [1, 2, 3, 4, 5, 6, 7]
    source = IterSource(data)
    ctx = CedarContext()

    source_pipe = source.to_pipe()
    source_pipe.id = 0
    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)

    map_pipe = MapperPipe(
        source_pipe,
        _add_one,
        input_tf_spec=tf.TensorSpec(shape=(), dtype=tf.int32),
        output_tf_hint=TFOutputHint(TFTensorDontCare, TFTensorDontCare),
    )
    map_pipe.id = 1
    map_pipe.mutate(ctx, PipeVariantType.TF)

    it = map_pipe.get_variant()

    out = []
    for x in it:
        try:
            out.append(x.data.numpy())
        except AttributeError:
            out.append(x.data)
    assert out == [x + 1 for x in data]

    out = []
    for x in it:
        try:
            out.append(x.data.numpy())
        except AttributeError:
            out.append(x.data)
    assert out == [x + 1 for x in data]


def test_sizeof_tf():
    from cedar.pipes.common import get_sizeof_data

    assert get_sizeof_data(tf.random.uniform((100, 100))) == 40000

    l_tensor = [
        tf.random.uniform((5, 5, 5)),
        tf.random.uniform((2, 3, 3)),
        tf.random.uniform((3, 3, 3)),
    ]
    assert get_sizeof_data(l_tensor) == 680


def test_trace_profile():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            ft = source_pipes[0]
            ft = NoopPipe(ft)
            ft = MapperPipe(
                ft,
                _add_one,
                input_tf_spec=tf.TensorSpec(shape=(), dtype=tf.int32),
                output_tf_hint=TFOutputHint(
                    TFTensorDontCare, TFTensorDontCare
                ),
            )
            ft = NoopPipe(ft)

            return ft

    ctx = CedarContext()

    data = [1, 2, 3]
    source = IterSource(data)
    feature = TestFeature()
    feature.apply(source)

    it = iter(feature.profile(ctx))

    ds = next(it)

    assert feature.logical_pipes[1].get_variant_type() == PipeVariantType.TF
    assert ds.data_size_dict is not None
    assert len(ds.data_size_dict) == 4

    ref_size = {3: 32, 2: 32, 1: 4, 0: 4}
    assert ds.data_size_dict == ref_size
    assert ds.data.numpy() == 2

    ds = next(it)
    assert ds.data.numpy() == 3

    ds = next(it)
    assert ds.data.numpy() == 4

    with pytest.raises(StopIteration):
        ds = next(it)


def test_ray(setup_ray):
    data = range(100)
    source = IterSource(data)

    ctx = CedarContext()
    noop_variant_ctx = RayPipeVariantContext(n_actors=1)
    map_variant_ctx = TFRayPipeVariantContext(n_actors=1, submit_batch_size=10)

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    map1 = MapperPipe(
        noop1,
        _add_one,
        input_tf_spec=tf.TensorSpec(shape=(), dtype=tf.int32),
        output_tf_hint=TFOutputHint(TFTensorDontCare, TFTensorDontCare),
    )
    noop2 = NoopPipe(map1)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.RAY, noop_variant_ctx)
    map1.mutate(ctx, PipeVariantType.TF_RAY, map_variant_ctx)
    noop2.mutate(ctx, PipeVariantType.INPROCESS)
    out = noop2.pipe_variant

    result = []
    for x in out:
        result.append(x.data.numpy())
    noop_variant_ctx.service.shutdown()
    map_variant_ctx.service.shutdown()
    assert result == list(range(1, 101))


def test_fuse_basic():
    ctx = CedarContext()
    data = range(100)
    source = IterSource(data)

    source_pipe = source.to_pipe()
    source_pipe.id = 0
    source_pipe.set_output_tf_spec(tf.TensorSpec(shape=(), dtype=tf.int32))
    map1 = MapperPipe(
        source_pipe,
        _add_one,
        output_tf_hint=TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
    )
    map1.id = 1
    map2 = MapperPipe(
        map1,
        _add_one,
        output_tf_hint=TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
    )
    map2.id = 2
    map3 = MapperPipe(
        map2,
        _add_one,
        output_tf_hint=TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
    )
    map3.id = 3

    fused_pipe = FusedOptimizerPipe([map1, map2, map3])

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    fused_pipe.mutate(ctx, PipeVariantType.TF)

    out = fused_pipe.pipe_variant
    res = []

    for x in out:
        res.append(x.data.numpy())

    assert res == list(range(3, 103))


def test_fuse_ray(setup_ray):
    ctx = CedarContext()
    data = range(100)
    source = IterSource(data)

    source_pipe = source.to_pipe()
    source_pipe.set_output_tf_spec(tf.TensorSpec(shape=(), dtype=tf.int32))
    map1 = MapperPipe(
        source_pipe,
        _add_one,
        output_tf_hint=TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
    )
    map2 = MapperPipe(
        map1,
        _add_one,
        output_tf_hint=TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
    )
    map3 = MapperPipe(
        map2,
        _add_one,
        output_tf_hint=TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
    )

    source_pipe.id = 0
    map1.id = 1
    map2.id = 2
    map3.id = 3

    map_variant_ctx = TFRayPipeVariantContext(n_actors=1, submit_batch_size=10)

    fused_pipe = FusedOptimizerPipe([map1, map2, map3])

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    fused_pipe.mutate(ctx, PipeVariantType.TF_RAY, map_variant_ctx)

    out = fused_pipe.pipe_variant
    res = []

    for x in out:
        res.append(x.data.numpy())

    assert res == list(range(3, 103))


def _fill_tensor(x):
    return tf.fill([10], x)


def _cast(x):
    return tf.cast(x, tf.float32)


def test_spec():
    ctx = CedarContext()
    data = range(100)
    source = IterSource(data)

    source_pipe = source.to_pipe()
    source_pipe.set_output_tf_spec(tf.TensorSpec(shape=(), dtype=tf.int32))
    map1 = MapperPipe(
        source_pipe,
        _add_one,
        output_tf_hint=TFOutputHint(TFTensorDontCare(), TFTensorDontCare()),
    )
    map2 = MapperPipe(
        map1,
        _fill_tensor,
        output_tf_hint=TFOutputHint([10], TFTensorDontCare()),
    )
    map3 = MapperPipe(
        map2,
        _cast,
        output_tf_hint=TFOutputHint(TFTensorDontCare(), tf.float32),
    )

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    map1.mutate(ctx, PipeVariantType.TF)
    map2.mutate(ctx, PipeVariantType.TF)
    map3.mutate(ctx, PipeVariantType.TF)

    out = map3.pipe_variant
    for i, x in enumerate(out):
        tensor = x.data
        assert tensor.shape.as_list() == [10]
        assert tensor.dtype == tf.float32
        tensor_l = tensor.numpy()
        assert [int(v) for v in tensor_l] == [i + 1] * 10


def test_reorder_spec():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            source_pipe = source_pipes[0]
            source_pipe.set_output_tf_spec(
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
            ft = source_pipe
            ft = MapperPipe(
                ft,
                _add_one,
                output_tf_hint=TFOutputHint(
                    TFTensorDontCare(), TFTensorDontCare()
                ),
            )
            ft = MapperPipe(
                ft,
                _fill_tensor,
                output_tf_hint=TFOutputHint([10], TFTensorDontCare()),
            )
            ft = MapperPipe(
                ft,
                _cast,
                output_tf_hint=TFOutputHint(TFTensorDontCare(), tf.float32),
            )
            ft = MapperPipe(
                ft,
                _add_one,
                output_tf_hint=TFOutputHint(
                    TFTensorDontCare(), TFTensorDontCare()
                ),
            )
            return ft

    data = range(100)
    source = IterSource(data)

    test_feature = TestFeature()
    test_feature.apply(source)

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    ref_config_file = (
        pathlib.Path(test_dir) / "data/config_fuse_reorder_tf.yml"
    )

    ctx = CedarContext()
    ds = test_feature.load_from_yaml(ctx, str(ref_config_file))

    for i, x in enumerate(ds):
        tensor = x.data
        assert tensor.shape.as_list() == [10]
        assert tensor.dtype == tf.float32
        tensor_l = tensor.numpy()
        assert [int(v) for v in tensor_l] == [i + 2] * 10


def test_fuse_inplace():
    class TestFeature(Feature):
        def _compose(self, source_pipes):
            source_pipe = source_pipes[0]
            source_pipe.set_output_tf_spec(
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
            ft = source_pipe
            ft = MapperPipe(
                ft,
                tf.strings.lower,
                output_tf_hint=TFOutputHint(
                    TFTensorDontCare(), TFTensorDontCare()
                ),
            )
            return ft

    test_dir = pathlib.Path(__file__).resolve().parents[0]
    source_file = test_dir / "data/test_tf_string.txt"

    source = TFLocalLineSource(str(source_file))
    test_feature = TestFeature()
    test_feature.apply(source)

    ctx = CedarContext()
    ref_config_file = pathlib.Path(test_dir) / "data/config_tf_string.yml"

    ds = test_feature.load_from_yaml(ctx, str(ref_config_file))

    res = []
    for x in ds:
        s = x.data.numpy()

        res.append(s)

    assert res == [
        b"hello",
        b"world",
        b"hello",
        b"world",
        b"hello",
        b"world",
        b"hello",
        b"world",
    ]
