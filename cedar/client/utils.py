import torch
import multiprocessing as mp
import logging
from typing import Optional, Union, Dict

from cedar.config import CedarContext
from cedar.compose import Feature, PhysicalPlan
from cedar.pipes import PipeVariantType, DataSample

from .profiler import FeatureProfiler
from .controller import FeatureController
from .logger import DataSetLogger

logger = logging.getLogger(__name__)


class Sentinel:
    def __init__(self, idx):
        self.idx = idx


def multiprocess_worker_loop(
    idx: int,
    ctx: CedarContext,
    queue: mp.Queue,
    feature: Feature,
    feature_name: str,
    feature_plan: Optional[PhysicalPlan],
    done: mp.Event,
    epoch_start: mp.Event,
    enable_controller: bool,
    available_scale: Dict[PipeVariantType, int],
):
    logger.info(f"Starting multiprocess worker {idx}...")
    torch.set_num_threads(1)

    if ctx.ray_config is not None:
        logger.info(f"Initializing Ray at worker {idx}")
        ctx.init_ray()

    if feature_plan is not None:
        logger.info(f"Loading feature {feature_name} from plan.")
        feat = feature.load_from_plan(ctx, feature_plan)
    else:
        feat = feature.load(ctx, False)

    if enable_controller:
        path = f"/tmp/cedar_{feature_name}_log.txt"
        with open(path, "w") as _:
            pass
        ds_logger = DataSetLogger(path)
        profiler = FeatureProfiler(feature, ds_logger)
        controller = FeatureController(  # noqa: F841
            profiler=profiler,
            feature=feature,
            logger=ds_logger,
            test_mode=False,
            available_scale=available_scale,
        )

    while True:
        # Wait for dataset to signal start
        epoch_start.wait()
        epoch_start.clear()
        logger.info(f"MP worker {idx} starting epoch.")

        # For torch tensors, background process needs to be alive while
        # main process reads the queue. Keep this process alive until
        # signaled by the main process.
        if done.is_set():
            break

        for x in feat:
            if isinstance(x, DataSample):
                if x.dummy:
                    continue
                if enable_controller:
                    profiler.update_ds(x)
                queue.put(x.data)
            else:
                queue.put(x)

        logger.info(f"MP worker {idx} finished epoch.")
        queue.put(Sentinel(idx))

    logger.info(f"Terminating worker {idx}")


def unpack_feature_map(
    feature_name: str,
    feature_map: Optional[
        Union[
            str,
            Dict[
                str,
                str,
            ],
        ]
    ],
) -> Optional[str]:
    if isinstance(feature_map, dict) and feature_name in feature_map:
        map = feature_map[feature_name]
    elif isinstance(feature_map, str):
        map = feature_map
    else:
        map = None
    return map
