import os
import time
import logging
import torch

from typing import Dict, Optional, Union, Iterable

from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from cedar.client import DataSet

from .timer import Timer
from cedar.config import RayConfig

logger = logging.getLogger(__name__)

PROGRESS_UPDATE_TIME_SEC = 10  # 10 seconds
NUM_WARMUP_SAMPLES = 800


class ProfilerSpec:
    def __init__(
        self,
        batch_size: int,
        num_total_samples: Optional[int],
        num_epochs: int,
        num_samples_per_epoch: Optional[int],
        config: Optional[Union[str, Dict[str, str]]] = None,
        kwargs: Dict[str, str] = None,
        use_ray: bool = False,
        ray_ip: str = "",
        iteration_time: Optional[float] = None,
        profiled_stats: str = "",
        run_profiling: bool = False,
    ):
        self.batch_size = batch_size
        self.num_total_samples = num_total_samples
        self.num_epochs = num_epochs
        self.num_samples_per_epoch = num_samples_per_epoch
        self.config = config
        self.kwargs = kwargs
        self.use_ray = use_ray
        self.ray_ip = ray_ip
        self.iteration_time = iteration_time
        self.profiled_stats = profiled_stats
        self.run_profiling = run_profiling

    def to_ray_config(self) -> Optional[RayConfig]:
        """
        Returns a Ray spec for the CedarContext, if specified by
        the profiler spec
        """
        if not self.use_ray:
            return None

        return RayConfig(self.ray_ip)


class Profiler:
    """
    Profiler used for measuring data loading performance.

    Currently expects a torchdata IterDataPipe. If the dataset
    is not already an IterDataPipe, wrap it with an IterableWrapper

    TODO (myzhao): once we implement a client, have the
    profiler use the client.

    Args:
        dataset: Iterable to load data from
        spec: ProfilerSpec to use for this run
    """

    def __init__(
        self,
        dataset: Union[IterDataPipe, Iterable],
        spec: ProfilerSpec,
        limit_torch_parallelism: bool = True,
    ) -> None:
        self.load_time = 0
        self.epoch_run_times = []
        self.epoch_num_samples = []
        # clear the page cache to get cleanest results
        os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')

        # Disable intra-op parallelism in torch
        if limit_torch_parallelism:
            logger.warning("Setting torch threads to 1")
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)

        if isinstance(dataset, IterDataPipe):
            self.dataset = dataset
        elif isinstance(dataset, DataSet):
            print("Profiling using an cedar DataSet.")
            self.dataset = dataset

            # result_dir = pathlib.Path.cwd()
            # print(f"Saving plan viz/config to {str(result_dir)}")
            # self.dataset.viz_logical_plan(str(result_dir))
            # self.dataset.viz_physical_plan(str(result_dir))
            # self.dataset.save_config(str(result_dir))
        else:
            self.dataset = IterableWrapper(dataset)
        self.spec = spec

    def init(self) -> None:
        """
        Initializes the dataset and loads data sources
        """
        timer = Timer()
        with timer:
            # Just use the iterable itself for now
            # TODO: replace with a proper dataloader
            self.loader = self.dataset
        self.load_time = timer.delta()

    def write_output(self, file: str):
        """
        Saves up to num_total_samples (or the entire epoch) to a file.
        """
        with open(file, "a") as f:
            curr_total_samples = 0
            for _ in range(self.spec.num_epochs):
                curr_epoch_samples = 0
                for x in self.loader:
                    curr_total_samples += self.spec.batch_size
                    curr_epoch_samples += self.spec.batch_size
                    if (
                        self.spec.num_samples_per_epoch
                        and curr_epoch_samples
                        >= self.spec.num_samples_per_epoch
                    ) or (
                        self.spec.num_total_samples
                        and curr_total_samples >= self.spec.num_total_samples
                    ):
                        break
                    f.write(str(x))
                    f.write("\n")
                    if torch.is_tensor(x):
                        f.write(f"Tensor size: {str(x.size())}\n")

    def run(self) -> None:
        """
        Run the profiler.

        NOTE: The reported number of samples may be inaccurate for
            tail batches (e.g., incomplete last batch in the epoch)
        """
        curr_total_samples = 0
        print_time = time.time() + PROGRESS_UPDATE_TIME_SEC

        for epoch in range(self.spec.num_epochs):
            if (
                self.spec.num_total_samples
                and curr_total_samples >= self.spec.num_total_samples
            ):
                break
            curr_epoch_samples = 0
            num_batches = 0
            warmup_done = False

            timer = Timer()
            with timer:
                for i, x in enumerate(self.loader):
                    # Reset the timer after warming up each epoch
                    # NOTE: for benchmarking only
                    curr_total_samples += self.spec.batch_size
                    curr_epoch_samples += self.spec.batch_size

                    # Reset the timer after warming up each epoch
                    # NOTE: for benchmarking only
                    if (
                        not warmup_done
                        and curr_epoch_samples >= NUM_WARMUP_SAMPLES
                    ):
                        timer.reset()
                        warmup_done = True

                    if (
                        self.spec.num_samples_per_epoch
                        and curr_epoch_samples
                        >= self.spec.num_samples_per_epoch
                    ) or (
                        self.spec.num_total_samples
                        and curr_total_samples >= self.spec.num_total_samples
                    ):
                        break

                    num_batches += 1

                    curr_time = time.time()
                    if curr_time >= print_time:
                        self._print_status(
                            epoch,
                            curr_total_samples,
                            curr_epoch_samples,
                        )
                        print_time = curr_time + PROGRESS_UPDATE_TIME_SEC

                    if self.spec.iteration_time is not None:
                        time.sleep(self.spec.iteration_time)

            epoch_time = timer.delta()
            self._print_status(
                epoch,
                curr_total_samples,
                curr_epoch_samples,
                epoch_time * 1e6,
            )

            self.epoch_run_times.append(timer.delta())
            self.epoch_num_samples.append(curr_epoch_samples)

    def print_results(self):
        total_run_time = sum(self.epoch_run_times)
        total_samples = sum(self.epoch_num_samples)

        print("======Epoch time breakdowns======")
        for idx, epoch in enumerate(self.epoch_run_times):
            print(
                "Epoch [{}/{}]: {:4.3f}us".format(
                    idx, len(self.epoch_run_times) - 1, epoch * 1e6
                )
            )

        print("======Benchmark Summary======")
        print(
            (
                "Total time: {:4.3f}us\n"
                + "\tcreate time: {:4.3f}us\n"
                + "\trun time: {:4.3f}us\n"
                + "\tnum samples: {}\n"
                + "\ttime per sample: {:4.3f}us"
            ).format(
                (self.load_time + total_run_time) * 1e6,
                (self.load_time) * 1e6,
                (total_run_time * 1e6),
                (total_samples),
                (total_run_time / total_samples) * 1e6,
            )
        )

    def get_results(self):
        """
        Returns a dict with profiled results.
        """
        return {
            "num_epochs": self.spec.num_epochs,
            "batch_size": self.spec.batch_size,
            "num_total_samples": self.spec.num_total_samples,
            "num_samples_per_epoch": self.spec.num_samples_per_epoch,
            "epoch_run_times": self.epoch_run_times,
            "epoch_num_samples": self.epoch_num_samples,
            "load_time": self.load_time,
        }

    def _print_status(
        self,
        epoch: int,
        curr_total_samples: int,
        curr_epoch_samples: int,
        epoch_time: Optional[int] = None,
    ) -> None:
        print(
            "Epoch [{}/{}]: [{}/{}] total samples."
            " [{}/{}] epoch samples".format(
                epoch,
                self.spec.num_epochs - 1,
                curr_total_samples,
                (
                    self.spec.num_total_samples
                    if self.spec.num_total_samples
                    else "???"
                ),
                curr_epoch_samples,
                (
                    self.spec.num_samples_per_epoch
                    if self.spec.num_samples_per_epoch
                    else "???"
                ),
            )
        )
        if epoch_time:
            print("     Epoch time: {:4.3f}us".format(epoch_time))
