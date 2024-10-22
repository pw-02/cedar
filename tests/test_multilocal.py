import functools
import pytest

from cedar.config import CedarContext
from cedar.sources import IterSource
from cedar.pipes import (
    PipeVariantType,
    NoopPipe,
    MapperPipe,
    MultiprocessPipeVariantContext,
    MultithreadedPipeVariantContext,
)


def test_basic_multiprocess():
    data = [1, 2, 3]
    source = IterSource(data)

    ctx = CedarContext()
    variant_ctx = MultiprocessPipeVariantContext(n_procs=2)

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.MULTIPROCESS, variant_ctx)
    noop2.mutate(ctx, PipeVariantType.INPROCESS)
    out = noop2.pipe_variant

    assert noop1.pipe_variant.variant_ctx.n_procs == 2

    result = []
    for x in out:
        result.append(x.data)
    variant_ctx.service.shutdown()
    assert result == data


def test_long_multiprocess():
    data = range(500)
    source = IterSource(data)

    ctx = CedarContext()
    variant_ctx = MultiprocessPipeVariantContext(n_procs=4)

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.MULTIPROCESS, variant_ctx)
    noop2.mutate(ctx, PipeVariantType.INPROCESS)

    out = noop2.pipe_variant

    result = []
    for x in out:
        result.append(x.data)

    assert noop1.pipe_variant.issued_tasks == 500
    assert noop1.pipe_variant.completed_tasks == 500

    variant_ctx.service.shutdown()
    assert result == list(data)


def _add(x, y):
    return x + y


def test_map():
    add_one = functools.partial(_add, y=1)

    data = range(500)
    source = IterSource(data)

    ctx = CedarContext()
    variant_ctx1 = MultiprocessPipeVariantContext(4)
    variant_ctx2 = MultiprocessPipeVariantContext(4)

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop = NoopPipe(source_pipe)
    map = MapperPipe(noop, add_one)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop.mutate(ctx, PipeVariantType.MULTIPROCESS, variant_ctx1)
    map.mutate(ctx, PipeVariantType.MULTIPROCESS, variant_ctx2)

    out = map.pipe_variant

    result = []
    for x in out:
        result.append(x.data)

    assert noop.pipe_variant.issued_tasks == 500
    assert noop.pipe_variant.completed_tasks == 500
    assert map.pipe_variant.issued_tasks == 500
    assert map.pipe_variant.completed_tasks == 500

    variant_ctx1.service.shutdown()
    variant_ctx2.service.shutdown()
    assert result == list(range(1, 501))


@pytest.mark.parametrize("use_threads", [False, True])
def test_basic_multithread(use_threads):
    data = [1, 2, 3]
    source = IterSource(data)

    ctx = CedarContext()
    variant_ctx = MultithreadedPipeVariantContext(
        n_threads=2, use_threads=use_threads
    )

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(ctx, PipeVariantType.MULTITHREADED, variant_ctx)
    noop2.mutate(ctx, PipeVariantType.INPROCESS)
    out = noop2.pipe_variant

    result = []
    for x in out:
        result.append(x.data)
    variant_ctx.service.shutdown()
    assert result == data


@pytest.mark.parametrize("use_threads", [False, True])
def test_map_multithreaded(use_threads):
    add_one = functools.partial(_add, y=1)

    data = range(500)
    source = IterSource(data)

    ctx = CedarContext()
    variant_ctx1 = MultithreadedPipeVariantContext(
        n_threads=4, use_threads=use_threads
    )
    variant_ctx2 = MultithreadedPipeVariantContext(
        n_threads=4, use_threads=use_threads
    )

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop = NoopPipe(source_pipe)
    map = MapperPipe(noop, add_one)

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop.mutate(ctx, PipeVariantType.MULTITHREADED, variant_ctx1)
    map.mutate(ctx, PipeVariantType.MULTITHREADED, variant_ctx2)

    out = map.pipe_variant

    result = []
    for x in out:
        result.append(x.data)

    assert noop.pipe_variant.issued_tasks == 500
    assert noop.pipe_variant.completed_tasks == 500
    assert map.pipe_variant.issued_tasks == 500
    assert map.pipe_variant.completed_tasks == 500

    variant_ctx1.service.shutdown()
    variant_ctx2.service.shutdown()
    assert result == list(range(1, 501))


@pytest.mark.parametrize("use_threads", [False, True])
def test_scale(use_threads):
    add_one = functools.partial(_add, y=1)

    data = range(500)
    source = IterSource(data)

    ctx = CedarContext()

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop = NoopPipe(source_pipe)
    map = MapperPipe(noop, add_one)

    source_pipe.id = 0
    noop.id = 1
    map.id = 2

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop.mutate(
        ctx,
        PipeVariantType.MULTITHREADED,
        MultithreadedPipeVariantContext(2, use_threads=use_threads),
    )
    map.mutate(
        ctx,
        PipeVariantType.MULTITHREADED,
        MultithreadedPipeVariantContext(2, use_threads=use_threads),
    )

    it = iter(map.pipe_variant)

    result = []

    for _ in range(100):
        result.append(next(it).data)

    map.pipe_variant.set_scale(4)
    assert map.pipe_variant.variant_ctx.service.n_threads == 4
    assert map.pipe_variant.get_scale() == 4

    for _ in range(200):
        result.append(next(it).data)

    map.pipe_variant.set_scale(1)
    assert map.pipe_variant.variant_ctx.service.n_threads == 1
    assert map.pipe_variant.get_scale() == 1

    for _ in range(200):
        result.append(next(it).data)

    with pytest.raises(StopIteration):
        next(it)

    assert noop.pipe_variant.issued_tasks == 500
    assert noop.pipe_variant.completed_tasks == 500
    assert map.pipe_variant.issued_tasks == 500
    assert map.pipe_variant.completed_tasks == 500

    noop.pipe_variant.variant_ctx.service.shutdown()
    map.pipe_variant.variant_ctx.service.shutdown()
    assert set(result) == set((range(1, 501)))  # ordering not guaranteed
