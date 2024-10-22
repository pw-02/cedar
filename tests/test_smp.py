import functools
import pytest
from cedar.config import CedarContext
from cedar.sources import IterSource
from cedar.pipes import (
    PipeVariantType,
    NoopPipe,
    MapperPipe,
    SMPPipeVariantContext,
)


@pytest.mark.parametrize("use_threads", [False, True])
def test_basic_smp(use_threads):
    data = [1, 2, 3]
    source = IterSource(data)

    ctx = CedarContext()

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)

    noop1.id = 0  # manually assign

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(
        ctx,
        PipeVariantType.SMP,
        SMPPipeVariantContext(
            3, use_threads=use_threads, disable_torch_parallelism=False
        ),
    )
    noop2.mutate(ctx, PipeVariantType.INPROCESS)
    out = noop2.pipe_variant

    result = []
    for x in out:
        result.append(x.data)
    noop1.pipe_variant.variant_ctx.service.shutdown()
    assert set(result) == set(data)


@pytest.mark.parametrize("use_threads", [False, True])
def test_long_smp(use_threads):
    data = range(500)
    source = IterSource(data)

    ctx = CedarContext()

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop1 = NoopPipe(source_pipe)
    noop2 = NoopPipe(noop1)

    noop1.id = 0
    noop2.id = 1

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop1.mutate(
        ctx,
        PipeVariantType.SMP,
        SMPPipeVariantContext(
            3, use_threads=use_threads, disable_torch_parallelism=False
        ),
    )
    noop2.mutate(
        ctx,
        PipeVariantType.SMP,
        SMPPipeVariantContext(
            3, use_threads=use_threads, disable_torch_parallelism=False
        ),
    )

    out = noop2.pipe_variant

    result = []
    for x in out:
        result.append(x.data)

    assert noop1.pipe_variant.issued_tasks == 500
    assert noop1.pipe_variant.completed_tasks == 500

    assert set(result) == set(data)
    noop1.pipe_variant.variant_ctx.service.shutdown()
    noop2.pipe_variant.variant_ctx.service.shutdown()


def _add(x, y):
    return x + y


@pytest.mark.parametrize("use_threads", [False, True])
def test_map(use_threads):
    add_one = functools.partial(_add, y=1)

    data = range(500)
    source = IterSource(data)

    ctx = CedarContext()

    # source -> noop -> noop
    source_pipe = source.to_pipe()
    noop = NoopPipe(source_pipe)
    map = MapperPipe(noop, add_one)

    noop.id = 0
    map.id = 1

    source_pipe.mutate(ctx, PipeVariantType.INPROCESS)
    noop.mutate(
        ctx,
        PipeVariantType.SMP,
        SMPPipeVariantContext(
            4, use_threads=use_threads, disable_torch_parallelism=False
        ),
    )
    map.mutate(
        ctx,
        PipeVariantType.SMP,
        SMPPipeVariantContext(
            4, use_threads=use_threads, disable_torch_parallelism=False
        ),
    )

    out = map.pipe_variant

    result = []
    for x in out:
        result.append(x.data)

    assert noop.pipe_variant.issued_tasks == 500
    assert noop.pipe_variant.completed_tasks == 500
    assert map.pipe_variant.issued_tasks == 500
    assert map.pipe_variant.completed_tasks == 500

    noop.pipe_variant.variant_ctx.service.shutdown()
    map.pipe_variant.variant_ctx.service.shutdown()

    assert set(result) == set((range(1, 501)))  # ordering not guaranteed


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
        PipeVariantType.SMP,
        SMPPipeVariantContext(
            2, 10, use_threads=use_threads, disable_torch_parallelism=False
        ),
    )
    map.mutate(
        ctx,
        PipeVariantType.SMP,
        SMPPipeVariantContext(
            2, 10, use_threads=use_threads, disable_torch_parallelism=False
        ),
    )

    it = iter(map.pipe_variant)

    result = []

    for _ in range(100):
        result.append(next(it).data)

    map.pipe_variant.set_scale(4)
    assert len(map.pipe_variant.variant_ctx.service.procs) == 4
    assert map.pipe_variant.get_scale() == 4

    for _ in range(200):
        result.append(next(it).data)

    map.pipe_variant.set_scale(1)
    assert len(map.pipe_variant.variant_ctx.service.procs) == 1
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
