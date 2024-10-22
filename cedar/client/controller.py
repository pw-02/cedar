import threading
import time
import logging
import random
from typing import Optional, List, Dict, Any, Set, Tuple
from cedar.compose import Feature
from cedar.pipes import PipeVariantType, PipeVariantContext
from .profiler import FeatureProfiler
from .logger import DataSetLogger
from .constants import (
    CONTROLLER_PERIOD_SEC,
    EMPTY_BUFFER_THRESHOLD,
    SCALE_ATTEMPTS,
    THROUGHPUT_THRESHOLD,
    CONTROLLER_SCALE_DOWN_COUNTER,
)

logger = logging.getLogger(__name__)


class ControllerExit(Exception):
    pass


class ControllerThread(threading.Thread):
    def __init__(
        self,
        profiler: FeatureProfiler,
        feature: Feature,
        ds_logger: Optional[DataSetLogger],
        test_mode: bool = False,
        available_scale: Dict[PipeVariantType, int] = {},
    ) -> None:
        super().__init__(daemon=True)
        self.shutdown_event = threading.Event()
        self._profiler = profiler
        self._feature = feature
        self._logger = ds_logger

        # To track throughput per loop
        self._prev_tput_time = 0
        self._prev_tput_sample_count = 0
        self._curr_tput = 0
        self._prev_tput = 0
        self._prev_control_time = 0

        self._scale_down_counter = CONTROLLER_SCALE_DOWN_COUNTER

        self.available_scale = available_scale
        if len(self.available_scale) == 0 and not test_mode:
            raise RuntimeError("No scalable variants provided")

        # For testing
        self._test_mode = test_mode
        self._enable_scaling = True
        self._step_iteration = threading.Event()

        self._do_not_fuse = set()
        self._active_fused_pipes = set()
        self._active_fused_pipes_map = {}

        self._best_scale = {}  # Dict from pipe id to tuple(scale, tput)

    def run(self) -> None:
        # Warmup step
        for _ in range(10):
            self._loop_step()
        while True:
            # uncomment for throughput testing
            # avg_tput = self._get_average_throughput(10)
            # self._log("avg tput: {}".format(avg_tput))
            # continue
            self._log("====Controller Step=====")
            try:
                self._scale_pipes()
            except ControllerExit:
                break
        logger.info("Exiting controller...")

    def _scale_pipes(self):
        """
        Makes a single change to the scale of each bottleneck pipe
        """
        is_bottleneck, bottleneck_pipes = self._calculate_bottleneck()
        logger.info(bottleneck_pipes)

        if len(bottleneck_pipes) == 0:
            self._loop_step()
            return

        if is_bottleneck:
            # Get the lowest pipe
            sorted_pipes = sorted(bottleneck_pipes, key=bottleneck_pipes.get)
            p_id = sorted_pipes[0]
            # Can we increase the scale?
            variant_type = self._feature.get_pipe(p_id).get_variant_type()
            curr_aggregate_scale = self._get_current_scale(variant_type)
            self._log(
                f"Curr scale for {variant_type.name} at {curr_aggregate_scale}"
            )
            curr_scale = self._get_pipe_scale(p_id)
            self._log(f"Current Scale for {p_id}: {curr_scale}")

            if curr_aggregate_scale < self.available_scale[variant_type]:
                self._scale_up_pipe(p_id)
            else:
                self._log(f"{variant_type.name} at max scale...")
                # otherwise, pick the pipe w/ longest buffer
                # that matches the current variant and scale it down
                for p_id in sorted_pipes[:0:-1]:
                    cmp_var_type = self._feature.get_pipe(
                        p_id
                    ).get_variant_type()
                    if cmp_var_type == variant_type:
                        self._scale_down_pipe(p_id)
                        break
        else:
            # Not a bottleneck, pick random pipe and scale it down
            # Scale down slower than we scale up
            if self._scale_down_counter <= 0:
                p_id = random.choice(list(bottleneck_pipes.keys()))
                self._scale_down_pipe(p_id)
                self._scale_down_counter = CONTROLLER_SCALE_DOWN_COUNTER
            else:
                self._scale_down_counter -= 1

        for p_id in bottleneck_pipes:
            self._log(f"P{p_id} has scale {self._get_pipe_scale(p_id)}.")
        self._loop_step()

    def _scale_down_pipe(self, p_id: int):
        self._log("======Scaling Down pipe ID {}======".format(p_id))
        curr_scale = self._get_pipe_scale(p_id)

        if curr_scale > 1:
            curr_scale = curr_scale - 1
            self._log(f"Decreasing scale for P{p_id} to {curr_scale}")
            self._set_pipe_scale(p_id, curr_scale)
        else:
            self._log(f"Mutating P{p_id} to local")
            pipe = self._feature.get_pipe(p_id)
            pipe.wait_until_mutate_ready()
            self._feature.dynamic_mutate(p_id, PipeVariantType.INPROCESS)

        # spin a bit so we don't mutate too quickly
        self._get_average_throughput(10)

    def _scale_up_pipe(self, p_id: int):
        self._log("======Scaling Up pipe ID {}======".format(p_id))
        # Get a snapshot of current performance
        checkpoint_throughput = self._get_average_throughput(5)
        self._log(
            "\tThroughput checkpointed at {:4.2f}".format(
                checkpoint_throughput
            )
        )

        variant_type = self._feature.get_pipe(p_id).get_variant_type()
        curr_scale = self._get_pipe_scale(p_id)

        while True:
            curr_scale = curr_scale + 1
            self._set_pipe_scale(p_id, curr_scale)
            self._log(f"Scaled P{p_id} to: {curr_scale}")
            # Wait for warm up
            for _ in range(5):
                self._loop_step()
            scaled_throughput = self._get_average_throughput(5)
            self._log("Scaled throughput at {:4.2f}".format(scaled_throughput))

            if (
                scaled_throughput
                > checkpoint_throughput * THROUGHPUT_THRESHOLD
            ):
                checkpoint_throughput = scaled_throughput
                self._log(f"Throughput improved to {checkpoint_throughput}")
                if (
                    self._get_current_scale(variant_type)
                    >= self.available_scale[variant_type]
                ):
                    self._log("Reached max scale...")
                    break
            else:
                # Scaling doesn't help, so set it back to the original
                curr_scale = curr_scale - 1
                self._log(f"Decreasing scale for P{p_id} to {curr_scale}")
                self._set_pipe_scale(p_id, curr_scale)
                break
        # else:
        #     if curr_scale > 1:
        #         curr_scale = curr_scale - 1
        #         self._set_pipe_scale(p_id, curr_scale)
        #         self._log(f"Scaled P{p_id} to: {curr_scale - 1}")

        #         scaled_throughput = self._get_average_throughput(5)
        #         self._log(
        #             "Scaled throughput at {:4.2f}".format(scaled_throughput)
        #         )
        #         # If decreasing the scale of this pipe doesn't impact
        #         # throughput that much, then keep the scale.
        #         if (
        #             scaled_throughput * THROUGHPUT_THRESHOLD
        #             < checkpoint_throughput
        #         ):
        #             curr_scale = curr_scale + 1
        #             self._log(f"Increasing scale for P{p_id} to {curr_scale}")  # noqa: E501
        #             self._set_pipe_scale(p_id, curr_scale)

    def _get_pipe_scale(self, p_id: int) -> int:
        return self._feature.get_pipe(p_id).get_variant().get_scale()

    def _set_pipe_scale(self, p_id: int, scale: int) -> None:
        logger.info(f"Scaling pipe {p_id} parallelism to {scale}")
        self._feature.get_pipe(p_id).get_variant().set_scale(scale)

    def _tune_pipes(self, tuned_pipes: Set[int]):
        while True:
            bottleneck_pipes = self._calculate_bottleneck()
            # minus done pipes
            bottleneck_pipes = [
                p for p in bottleneck_pipes if p not in tuned_pipes
            ]

            if len(bottleneck_pipes) >= 1:
                # Do stuff
                bottleneck_p_id = bottleneck_pipes[0]
                self._tune_pipe(bottleneck_p_id)
                tuned_pipes.add(bottleneck_p_id)
            else:
                self._loop_step()
                return

    def _fuse_pipes(self):
        self._log("\tFusing pipes...")
        # Ok, get any pipes that are mutated
        active_pipe_variants = self._feature.get_active_pipe_variants()
        logger.info(active_pipe_variants)
        bottleneck_pipes = self._calculate_bottleneck()
        bottleneck_pipes = [
            p for p in bottleneck_pipes if p not in self._do_not_fuse
        ]
        self._log("\tBottleneck Pipes: {}".format(bottleneck_pipes))

        # Keep a memo of fused pipes
        fused_pipe_memo = {}

        if PipeVariantType.SMP in active_pipe_variants:
            for p_id in bottleneck_pipes:
                if p_id in active_pipe_variants[
                    PipeVariantType.SMP
                ] and self._is_fusable(p_id, PipeVariantType.SMP):
                    res = self._fuse_pipe(p_id, fused_pipe_memo)
                    if res:
                        fused_pipe_memo.clear()
        if PipeVariantType.MULTITHREADED in active_pipe_variants:
            for p_id in bottleneck_pipes:
                if p_id in active_pipe_variants[
                    PipeVariantType.MULTITHREADED
                ] and self._is_fusable(p_id, PipeVariantType.SMP):
                    res = self._fuse_pipe(p_id, fused_pipe_memo)
                    if res:
                        fused_pipe_memo.clear()
        if PipeVariantType.INPROCESS in active_pipe_variants:
            for p_id in bottleneck_pipes:
                if p_id in active_pipe_variants[
                    PipeVariantType.INPROCESS
                ] and self._is_fusable(p_id, PipeVariantType.SMP):
                    res = self._fuse_pipe(p_id, fused_pipe_memo)
                    if res:
                        fused_pipe_memo.clear()

        self._loop_step()

    def _fuse_pipe(self, p_id: int, fused_memo: Dict[Tuple[int], float]):
        self._log("\tFinding best fusion for pipe {}".format(p_id))
        checkpoint_throughput = self._get_max_throughput(5)
        self._log(
            "\tCheckpointed baseline throughput at {}".format(
                checkpoint_throughput
            )
        )
        best_throughput = 0
        best_fusion = []

        # Get the neighbors of this pipe
        (
            upstream_neighbor,
            downstream_neighbor,
        ) = self._feature.get_neighbor_ids(p_id)

        curr_fusion = [p_id]
        while True:
            _, downstream_neighbor = self._feature.get_neighbor_ids(
                curr_fusion[-1]
            )
            upstream_neighbor, _ = self._feature.get_neighbor_ids(
                curr_fusion[0]
            )
            left_fusion = [upstream_neighbor] + curr_fusion
            right_fusion = curr_fusion + [downstream_neighbor]
            fused_left = False
            fused_right = False
            left_tput = 0
            right_tput = 0

            if self._is_fusable(upstream_neighbor, PipeVariantType.SMP):
                left_tput = self._fuse_pipe_helper(left_fusion, fused_memo)

                # Wait a bit
                if left_tput > best_throughput:
                    self._log("\tImproved throughput to {}".format(left_tput))
                    best_fusion = left_fusion.copy()
                    best_throughput = left_tput
                # Reset the fusion
                fused_left = True

            if self._is_fusable(downstream_neighbor, PipeVariantType.SMP):
                right_tput = self._fuse_pipe_helper(right_fusion, fused_memo)
                if right_tput > best_throughput:
                    self._log("\tImproved throughput to {}".format(right_tput))
                    best_fusion = right_fusion.copy()
                    best_throughput = right_tput
                # Reset the fusion
                fused_right = True

            if fused_left and fused_right:
                if left_tput > right_tput:
                    curr_fusion = left_fusion.copy()
                else:
                    curr_fusion = right_fusion.copy()
            elif fused_left:
                curr_fusion = left_fusion.copy()
            elif fused_right:
                curr_fusion = right_fusion.copy()
            else:
                self._log("No possible fusion")
                break

            self._log("\tBest fusion option {}".format(curr_fusion))

        self._log("\tFound best fusion: {}".format(best_fusion))

        if len(best_fusion) >= 2:
            ctxs = self._feature.reset_pipes(best_fusion)
            fused_p_id = self._feature.dynamic_fusion(best_fusion)

            best_tput, best_variant_ctx = self._scale_pipe_deprecated(
                fused_p_id
            )

            if best_tput >= checkpoint_throughput * THROUGHPUT_THRESHOLD:
                self._log("\tKeeping fused pipe {}".format(fused_p_id))
                self._active_fused_pipes.update(best_fusion)
                self._active_fused_pipes_map[fused_p_id] = best_fusion
                return True
            else:
                self._log("\tReverting fused pipe {}".format(fused_p_id))
                self._feature.reset_pipes([fused_p_id])
                self._do_not_fuse.add(p_id)
                self._reset_to_original_ctx(ctxs)

                # Reset pipes to orignal context
                return False
        else:
            self._do_not_fuse.add(p_id)
            return False

    def _get_max_throughput(self, num_steps: int):
        checkpoint_throughput = 0
        for _ in range(num_steps):
            self._loop_step()
            checkpoint_throughput = max(checkpoint_throughput, self._curr_tput)
        return checkpoint_throughput

    def _get_average_throughput(self, num_steps: int):
        total_throughput = 0
        for _ in range(num_steps):
            self._loop_step()
            total_throughput += self._curr_tput
        return total_throughput / num_steps

    def stop(self) -> None:
        self.shutdown_event.set()

    def _loop_wait(self, start_time: float) -> None:
        if self._test_mode:
            self._step_iteration.wait(timeout=10)
            self._step_iteration.clear()
        else:
            self._sleep(start_time)

    def _loop_step(self) -> None:
        if self.shutdown_event.is_set():
            raise ControllerExit

        if self._test_mode:
            self._step_iteration.wait(timeout=10)
            self._step_iteration.clear()
        else:
            sleep_time = CONTROLLER_PERIOD_SEC - (
                time.time() - self._prev_control_time
            )
            if sleep_time > 0:
                time.sleep(sleep_time)

            self._prev_control_time = time.time()
            # Update the throughput in the last window
            self._curr_tput = self._calculate_throughput()
            self._log(
                "Curr Throughput (samples/s): {:6.3f}".format(self._curr_tput)
            )
            # self._log(
            #     "\tAvg samples/s since last step: {:6.3f}".format(
            #         self._curr_tput
            #     )
            # )
        # self._log("====Controller Step=====")

    def _log(self, msg: str) -> None:
        if self._logger:
            self._logger.log(msg)

    def _sleep(self, start_time: float) -> None:
        sleep_time = CONTROLLER_PERIOD_SEC - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    def _calculate_bottleneck(self) -> Tuple[bool, List[int]]:
        """
        Based on profiled measurements, return an ordered list
        designating bottleneck pipes, in priority order.
        Returns an empty list if no bottlenecks found.
        """
        bottlenecked = False

        pipe_latencies = self._profiler.calculate_avg_latency_per_sample()
        pipe_buffer_sizes = self._profiler.calculate_avg_buffer_size()

        logger.info(pipe_buffer_sizes)
        # Remove any pipes from consideration that don't currently use buffers
        no_buffer = set()
        prefetch_buffer_len = None
        for p_id in pipe_buffer_sizes:
            pipe = self._feature.get_pipe(p_id)
            if pipe.get_variant_type() == PipeVariantType.INPROCESS:
                no_buffer.add(p_id)
            logger.info(pipe.get_logical_name())
            if "PrefetcherPipe" in pipe.get_logical_name():
                prefetch_buffer_len = pipe_buffer_sizes[p_id]

        self._log(f"\tPrefetch buffer: {prefetch_buffer_len}")
        if prefetch_buffer_len is None:
            logger.warning("Could not find prefetch pipe!")
            return (True, {})

        if prefetch_buffer_len < 50:
            bottlenecked = True

        pipe_buffer_sizes = {
            k: v
            for k, v in pipe_buffer_sizes.items()
            if k not in no_buffer and self._valid_bottleneck_pipe(k)
        }

        self._log("\tAverage Pipe Latencies: {}".format(pipe_latencies))
        self._log("\tAverage Buffer Sizes: {}".format(pipe_buffer_sizes))

        for k, v in pipe_buffer_sizes.items():
            if v < EMPTY_BUFFER_THRESHOLD:
                bottlenecked = True

        return (bottlenecked, pipe_buffer_sizes)

        # # If no profiled results, do not proceed
        # if len(pipe_latencies) == 0 and len(pipe_buffer_sizes) == 0:
        #     # logger.warning("Unable to obtain profiled results.")
        #     return res

        # # If any pipe has a small-sized buffer, designate it the bottleneck
        # for k, v in pipe_buffer_sizes.items():
        #     if v < EMPTY_BUFFER_THRESHOLD and self._valid_bottleneck_pipe(k):
        #         res[k] = v

        # sorted_pipes = sorted(
        #     res, key=res.get, reverse=True
        # )

        # # res.extend(sorted_pipes)
        # return (bottlenecked, sorted_pipes)

    def _valid_bottleneck_pipe(self, p_id: int) -> bool:
        """
        Returns true if pipe ID p_id can be a bottleneck pipe.
        """
        # Filter out all immutable pipes
        if not self._feature.get_pipe(p_id).is_mutable():
            return False

        # Batcher pipes are invalid
        if "BatcherPipe" in self._feature.get_pipe(p_id).get_logical_name():
            return False

        return True

    def _mutate(self, p_id) -> bool:
        """
        Mutates the bottleneck pipe if possible. Returns True if action
        was taken.
        """
        # Just use the first pipe for now
        pipe = self._feature.get_pipe(p_id)
        pipe.wait_until_mutate_ready()
        # pipe.mark_mutate_in_progress()

        pipe_spec = pipe.get_spec()
        valid_pipe_variants = pipe_spec.mutable_variants

        # Get the valid set of variants to mutate to
        curr_variant_type = pipe.get_variant_type()

        try:
            curr_variant_idx = valid_pipe_variants.index(curr_variant_type)
        except ValueError:
            logger.error(
                "Unable to find variant type {}"
                " in variant list for pipe {}".format(
                    curr_variant_type, pipe.name
                )
            )
            return False

        if curr_variant_idx == len(valid_pipe_variants) - 1:
            # No other variants to mutate to
            return False

        target_variant_type = valid_pipe_variants[curr_variant_idx + 1]

        # Check if we have enough parallelism to do the mutation
        scale_dict = self._feature.get_aggregate_scale()
        curr_local_parallelism = self._calculate_local_parallelism(scale_dict)
        final_local_parallelism = curr_local_parallelism

        if self._is_local_variant(target_variant_type):
            final_local_parallelism = final_local_parallelism + 1
        if self._is_local_variant(curr_variant_type):
            final_local_parallelism = (
                final_local_parallelism - pipe.pipe_variant.get_scale()
            )
        if final_local_parallelism > self._num_cores:
            logger.info("Pipe {} not mutable due to parallelism".format(p_id))
            return False

        log_str = "\tMutating pipe {}:{} from {} to {}".format(
            p_id, pipe.name, curr_variant_type, target_variant_type
        )
        logger.info(log_str)
        self._log(log_str)

        res = self._feature.dynamic_mutate(p_id, target_variant_type)

        return res

    def _scale(self, p_id: int) -> bool:
        """
        Scales the bottleneck pipe if possible. Returns true if action was
        taken.
        """
        if not self._enable_scaling:
            return False

        logger.info(f"Attempting to scale pipe {p_id}")
        pipe = self._feature.get_pipe(p_id)
        pipe_variant_type = pipe.get_variant_type()

        if not pipe.pipe_variant.is_scalable():
            logger.info(f"Pipe {p_id} is not scalable.")
            return False

        scale_dict = self._feature.get_aggregate_scale()
        self._log("\tCurrent parallelism: {}".format(scale_dict))

        # Check if we have enough resources to scale
        if self._can_scale(pipe_variant_type, scale_dict):
            curr_scale = pipe.pipe_variant.get_scale()
            pipe.pipe_variant.set_scale(curr_scale + 1)

            log_str = "\tScaling pipe {}:{} from {} to {}".format(
                p_id, pipe.name, curr_scale, curr_scale + 1
            )
            logger.info(log_str)
            self._log(log_str)

            return True
        else:
            return False

    def _can_scale(
        self,
        variant_type: PipeVariantType,
        curr_scale: Dict[PipeVariantType, int],
    ) -> bool:
        """
        True if the given pipe variant type can be scaled given a dict
        of current resources
        """
        if self._is_local_variant(variant_type):
            curr_local_parallelism = self._calculate_local_parallelism(
                curr_scale
            )
            if curr_local_parallelism < self._num_cores:
                return True
            else:
                return False
        else:
            logger.error(
                "Invalid pipe variant type {} to scale".format(variant_type)
            )
            return False

    def _is_pipe_scalable(self, p_id: int) -> bool:
        pipe_variant_type = self._feature.get_pipe(p_id).get_variant_type()
        return pipe_variant_type in self.available_scale

    def disable_scaling(self):
        self._enable_scaling = False

    def _calculate_local_parallelism(
        self, curr_scale: Dict[PipeVariantType, int]
    ) -> int:
        return (
            curr_scale.get(PipeVariantType.MULTIPROCESS, 0)
            + curr_scale.get(PipeVariantType.SMP, 0)
            + curr_scale.get(PipeVariantType.MULTITHREADED, 0)
        )

    def _get_current_scale(self, variant_type: PipeVariantType):
        scale_dict = self._feature.get_aggregate_scale()
        return scale_dict.get(variant_type, 0)

    def _is_local_variant(self, variant_type: PipeVariantType):
        return (
            variant_type == PipeVariantType.MULTITHREADED
            or variant_type == PipeVariantType.SMP
            or variant_type == PipeVariantType.MULTIPROCESS
        )

    def _calculate_throughput(self) -> float:
        """
        Returns the throughput, in samples / second, processed within the last
        controller loop
        """
        curr_time = time.time()
        curr_sample_count = self._profiler.get_sample_count()
        batch_size = self._profiler.get_batch_size()

        delta_sample_count = curr_sample_count - self._prev_tput_sample_count
        tput = (batch_size * delta_sample_count) / (
            curr_time - self._prev_tput_time
        )
        self._prev_tput_time = curr_time
        self._prev_tput_sample_count = curr_sample_count

        return tput

    def _tune_pipe(self, p_id: int):
        """
        Scales and mutates an individual pipe until the pipe is not a
        bottleneck, or the search space is exhausted
        """
        self._log("\tTuning pipe ID {}".format(p_id))

        # Get a snapshot of the current performance
        pipe = self._feature.get_pipe(p_id)

        checkpoint_throughput = self._get_max_throughput(5)
        checkpoint_variant_ctx = pipe.get_variant().serialize()

        self._log(
            "\tThroughput checkpointed at {:4.2f}".format(
                checkpoint_throughput
            )
        )

        best_throughput = checkpoint_throughput
        best_variant_ctx = checkpoint_variant_ctx

        while True:
            scale_tput, scale_variant_ctx = self._scale_pipe_deprecated(p_id)
            if (
                scale_tput > best_throughput * THROUGHPUT_THRESHOLD
                and scale_variant_ctx is not None
            ):
                self._log(
                    "\tThroughput improved to {:4.2f}".format(scale_tput)
                )
                best_throughput = scale_tput
                best_variant_ctx = scale_variant_ctx

            # We can't scale the current variant any more, try to mutate
            self._log(f"\tReached scaling attempt limit for {p_id}")
            success = self._mutate(p_id)
            self._loop_step()
            if not success:
                # If we can't mutate anymore, nothing more we can do
                break

            if self._curr_tput > best_throughput:
                self._log(
                    "\tThroughput improved to {:4.2f}".format(self._curr_tput)
                )
                best_throughput = self._curr_tput
                best_variant_ctx = pipe.get_variant().serialize()

        # If we've reached the end, we've exhausted the search space.
        # Revert back to the best throughput we found
        self._log(
            "\tTuned pipe {}, best throughput {:4.2f}".format(
                p_id, best_throughput
            )
        )
        self._log("\t{}".format(best_variant_ctx))
        self._feature.mutate_to(p_id, best_variant_ctx)
        self._loop_step()

    def _is_bottleneck(self, p_id: int) -> bool:
        """
        Returns if p_id is the current bottleneck pipe
        """
        bottleneck_pipes = self._calculate_bottleneck()
        return len(bottleneck_pipes) > 0 and bottleneck_pipes[0] == p_id

    def _is_fusable(self, p_id: Optional[int]) -> bool:
        """
        Returns true if p_id is a fusable pipe
        """
        if p_id is None:
            return False
        if p_id in self._active_fused_pipes:
            return False
        return self._feature.is_fusable(p_id, PipeVariantType.SMP)

    def _scale_pipe_deprecated(
        self, p_id: int
    ) -> Tuple[float, Optional[PipeVariantContext]]:
        """
        Scales p_id until it stops improving throughput

        Returns a tuple of the best throughput achieved, and the variant_ctx
        """
        best_throughput = 0
        best_variant_ctx = None
        scale_attempts = SCALE_ATTEMPTS
        while scale_attempts > 0:
            prev_throughput = self._curr_tput
            success = self._scale(p_id)
            self._loop_step()
            if not success:
                self._log(f"\tCannot scale pipe {p_id}")
                break

            # Did we improve throughput? If not, keep trying to scale
            if self._curr_tput <= prev_throughput * THROUGHPUT_THRESHOLD:
                scale_attempts -= 1
                continue

            # Keep scaling if we're improving throughput
            if self._curr_tput > best_throughput:
                self._log(
                    "\tThroughput improved to {:4.2f}".format(self._curr_tput)
                )
                best_throughput = self._curr_tput
                best_variant_ctx = (
                    self._feature.get_pipe(p_id).get_variant().serialize()
                )
                scale_attempts = SCALE_ATTEMPTS

        tput = self._get_max_throughput(3)
        if tput > best_throughput:
            best_throughput = tput
            best_variant_ctx = (
                self._feature.get_pipe(p_id).get_variant().serialize()
            )

        return (best_throughput, best_variant_ctx)

    def _fuse_pipe_helper(self, pipe_ids, memo):
        k = tuple(pipe_ids)
        if k in memo:
            tput = memo[k]
        else:
            self._log("\tFusing {}".format(pipe_ids))
            ctxs = self._feature.reset_pipes(pipe_ids)
            # self._feature.wait_for_mutation(pipe_ids)

            fused_p_id = self._feature.dynamic_fusion(pipe_ids)

            # Wait a bit
            tput = self._get_max_throughput(5)
            memo[k] = tput

            # Reset the fusion, resets all to INPROCESS
            self._feature.reset_pipes([fused_p_id])
            # self._feature.wait_for_mutation([fused_p_id])
            # Are there any pipes that were not INPROCESS?
            self._reset_to_original_ctx(ctxs)

        return tput

    def _reset_to_original_ctx(self, original_ctxs: Dict[int, Dict[str, Any]]):
        # Are there any pipes that were not INPROCESS?
        ctxs = {
            k: v
            for k, v in original_ctxs.items()
            if v["variant_type"] != PipeVariantType.INPROCESS.name
        }
        ctxs_keys = list(ctxs.keys())
        if len(ctxs) > 0:
            logger.info("Resetting pipes {} to original variant".format(ctxs))
            # self._feature.wait_for_mutation(ctxs_keys)
            self._feature.reset_pipes(ctxs_keys, ctxs)
            # self._feature.wait_for_mutation(ctxs_keys)


class FeatureController:
    """
    Controller class, which dynamically manages the execution of a feature.
    """

    def __init__(
        self,
        profiler: FeatureProfiler,
        feature: Feature,
        logger: Optional[DataSetLogger] = None,
        test_mode: bool = False,
        available_scale: Dict[PipeVariantType, int] = {},
    ) -> None:
        self._profiler = profiler
        self._logger = logger
        self._feature = feature
        self._controller_thread = ControllerThread(
            profiler, feature, logger, test_mode, available_scale
        )
        self._controller_thread.start()
        self._test_mode = test_mode

    def close(self) -> None:
        self._controller_thread.stop()
        self._controller_thread.join()

    def __del__(self) -> None:
        if self._controller_thread.is_alive():
            self.close()

    # For testing only
    def _disable_scaling(self):
        self._controller_thread.disable_scaling()
