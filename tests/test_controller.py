import pytest
import time
from cedar.client.controller import FeatureController
from cedar.pipes import (
    PipeVariantType,
)


# Override controller fixture
@pytest.fixture(autouse=True)
def override_control_loop_period(monkeypatch):
    monkeypatch.setattr("cedar.client.controller.CONTROLLER_PERIOD_SEC", 2)


@pytest.mark.skip()  # TODO: FIXME
def test_controller(noop_profiled_feature):
    feature, profiler = noop_profiled_feature
    controller = FeatureController(profiler, feature, test_mode=True)

    bottleneck = controller._controller_thread._calculate_bottleneck()

    assert bottleneck == [1, 2, 0]

    controller.close()


def test_controller_epoch(mocked_controller_dataset):
    # ensure fixed num cores
    controller = mocked_controller_dataset.controllers["feature"]
    controller._controller_thread._num_cores = 7
    out = []
    it = iter(mocked_controller_dataset)

    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant_type()
        == PipeVariantType.INPROCESS
    )
    for _ in range(300):
        out.append(next(it))

    controller._controller_thread._step_iteration.set()
    time.sleep(1)
    for _ in range(300):
        out.append(next(it))

    controller._controller_thread._step_iteration.set()
    time.sleep(1)
    for _ in range(300):
        out.append(next(it))

    controller._controller_thread._step_iteration.set()
    time.sleep(1)
    for _ in range(100):
        out.append(next(it))

    assert set(out) == set(range(1000))

    it = iter(mocked_controller_dataset)
    out = []
    for _ in range(1000):
        out.append(next(it))
    assert set(out) == set(range(1000))

    controller._controller_thread.stop()
    controller._controller_thread._step_iteration.set()


@pytest.mark.skip()  # TODO: FIXME
def test_controller_mutate(mocked_controller_dataset):
    controller = mocked_controller_dataset.controllers["feature"]
    controller._disable_scaling()

    out = []
    it = iter(mocked_controller_dataset)

    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant_type()
        == PipeVariantType.INPROCESS
    )
    for _ in range(300):
        out.append(next(it))

    # Wait for the bottleneck pipe to mutate to multithreaded
    controller._controller_thread._step_iteration.set()
    time.sleep(1)
    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant_type()
        == PipeVariantType.MULTITHREADED
    )
    for _ in range(300):
        out.append(next(it))

    # Wait for the bottleneck pipe to mutate to SMP
    # Need to drain the prefetch buffer to mutate
    controller._controller_thread._step_iteration.set()
    time.sleep(1)

    for _ in range(300):
        out.append(next(it))
    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant_type()
        == PipeVariantType.SMP
    )

    controller._controller_thread._step_iteration.set()
    time.sleep(1)
    for _ in range(100):
        out.append(next(it))
    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant_type()
        == PipeVariantType.SMP
    )

    assert set(out) == set(range(1000))

    controller._controller_thread.stop()
    controller._controller_thread._step_iteration.set()


@pytest.mark.skip()  # TODO: FIXME
def test_controller_scale(mocked_controller_dataset):
    # ensure fixed num cores
    controller = mocked_controller_dataset.controllers["feature"]
    controller._controller_thread._num_cores = 7
    out = []
    it = iter(mocked_controller_dataset)

    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant_type()
        == PipeVariantType.INPROCESS
    )
    for _ in range(300):
        out.append(next(it))

    # Wait for the bottleneck pipe to mutate to multithreaded
    controller._controller_thread._step_iteration.set()
    time.sleep(1)
    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant_type()
        == PipeVariantType.MULTITHREADED
    )
    for _ in range(300):
        out.append(next(it))

    # The controller should scale up the threads to 2

    controller._controller_thread._step_iteration.set()
    time.sleep(1)
    for _ in range(300):
        out.append(next(it))
    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant_type()
        == PipeVariantType.MULTITHREADED
    )
    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant()
        .get_scale()
        == 2
    )

    controller._controller_thread._step_iteration.set()
    time.sleep(1)
    for _ in range(100):
        out.append(next(it))
    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant()
        .get_scale()
        == 2
    )

    assert set(out) == set(range(1000))
    controller._controller_thread.stop()
    controller._controller_thread._step_iteration.set()


@pytest.mark.skip()  # TODO: FIXME
def test_controller_scale_epoch(mocked_controller_dataset):
    # ensure fixed num cores
    controller = mocked_controller_dataset.controllers["feature"]
    controller._controller_thread._num_cores = 7
    out = []
    it = iter(mocked_controller_dataset)

    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant_type()
        == PipeVariantType.INPROCESS
    )
    for _ in range(300):
        out.append(next(it))

    # Wait for the bottleneck pipe to mutate to multithreaded
    controller._controller_thread._step_iteration.set()
    time.sleep(1)
    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant_type()
        == PipeVariantType.MULTITHREADED
    )
    for _ in range(300):
        out.append(next(it))

    # The controller should scale up the threads to 2

    controller._controller_thread._step_iteration.set()
    time.sleep(1)
    for _ in range(300):
        out.append(next(it))
    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant_type()
        == PipeVariantType.MULTITHREADED
    )
    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant()
        .get_scale()
        == 2
    )

    controller._controller_thread._step_iteration.set()
    time.sleep(1)
    for _ in range(100):
        out.append(next(it))
    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant()
        .get_scale()
        == 2
    )

    assert set(out) == set(range(1000))

    it = iter(mocked_controller_dataset)
    out = []
    for _ in range(1000):
        out.append(next(it))
    assert set(out) == set(range(1000))

    assert (
        mocked_controller_dataset.features["feature"]
        .physical_pipes[0]
        .get_variant()
        .get_scale()
        == 2
    )

    controller._controller_thread.stop()
    controller._controller_thread._step_iteration.set()
