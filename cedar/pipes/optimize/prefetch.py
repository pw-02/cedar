import logging
import threading
import traceback
from collections import deque

from typing import Optional
from .registry import register_optimizer_pipe
from ..context import InProcessPipeVariantContext, PipeVariantType
from ..pipe import (
    Pipe,
)
from ..variant import (
    InProcessPipeVariant,
    PipeVariant,
)
from ..common import cedar_pipe, CedarPipeSpec


logger = logging.getLogger(__name__)


@register_optimizer_pipe("PrefetcherPipe")
@cedar_pipe(
    CedarPipeSpec(
        is_mutable=False,
        mutable_variants=[
            PipeVariantType.INPROCESS,
        ],
    )
)
class PrefetcherPipe(Pipe):
    """
    A prefetcher optimizer pipe, which prefetches a set amount of items
    from the preceding upstream pipe(s).

    Args:
        buf_size (int): Number of items to prefetch
    """

    def __init__(
        self,
        input_pipe: Optional[Pipe] = None,
        buffer_size: int = 100,
        is_random: bool = False,
    ):
        if input_pipe:
            super().__init__(
                "PrefetcherPipe", [input_pipe], is_random=is_random
            )
        else:
            super().__init__("PrefetcherPipe", [], is_random=is_random)

        self.buffer_size = buffer_size

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        variant = InProcessPrefetcherPipeVariant(
            self.input_pipes[0].pipe_variant, self.buffer_size
        )
        return variant

    def _check_mutation(self) -> None:
        super()._check_mutation()

        if len(self.input_pipes) != 1:
            raise RuntimeError("PrefetcherPipe only accepts one input.")


class InProcessPrefetcherPipeVariant(InProcessPipeVariant):
    def __init__(
        self, input_pipe_variant: Optional[PipeVariant], buffer_size: int
    ):
        super().__init__(input_pipe_variant)

        self._shutdown_event = threading.Event()
        self._exhausted_event = threading.Event()
        self._prefetch_thread = None
        self._buffer = deque()  # thread safe
        self._buffer_size = buffer_size

        self._buffer_lock = threading.Lock()
        self._buffer_not_full = threading.Condition(self._buffer_lock)
        self._buffer_not_empty = threading.Condition(self._buffer_lock)

    def _run(self) -> None:
        logger.info("Running prefetch thread")

        while not self._shutdown_event.is_set():
            try:
                data = next(self._input_iter)
            except StopIteration:
                # upstream data exhausted, stop thread
                logger.info("Upstream pipe exhausted.")
                with self._buffer_not_empty:
                    self._exhausted_event.set()
                    self._buffer_not_empty.notify()
                break
            except Exception as e:
                logger.error("Error prefetching data. Terminating.")
                logger.error(str(e))
                print(traceback.format_exc())
                break

            with self._buffer_not_full:
                # Wait if prefetch buffer full
                while len(self._buffer) >= self._buffer_size:
                    self._buffer_not_full.wait()
                self._buffer.append(data)
                self._buffer_not_empty.notify()

        logger.info("Terminating prefetch thread")
        self._exhausted_event.set()

    def _iter_impl(self):
        # Set up a separate thread to consume upstream pipes, to fill up
        # buffer. Yield from buffer when data is available.

        # In the event of a new epoch, clear all state
        self._reset_iter()

        self._prefetch_thread = threading.Thread(target=self._run, daemon=True)
        self._prefetch_thread.start()

        while (not self._exhausted_event.is_set()) or (len(self._buffer) > 0):
            with self._buffer_not_empty:
                while len(self._buffer) == 0:
                    if self._exhausted_event.is_set():
                        return
                    else:
                        self._buffer_not_empty.wait()
                data = self._buffer.popleft()
                self._buffer_not_full.notify()
            yield data

    def shutdown(self) -> None:
        self._shutdown_event.set()
        self._exhausted_event.set()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()

    def _reset_iter(self) -> None:
        # In the event of a new epoch, clear all state
        if self._prefetch_thread is not None:
            self._buffer.clear()
            self._shutdown_event.set()
            with self._buffer_not_full:
                # Notify the thread if waiting
                self._buffer_not_full.notify()
            self._prefetch_thread.join()
        self._exhausted_event.clear()
        self._shutdown_event.clear()
        self._buffer.clear()
        self._prefetch_thread = None

    def __del__(self):
        self.shutdown()

    def get_buffer_size(self) -> Optional[int]:
        return len(self._buffer)
