import logging
import threading

from concurrent.futures import ThreadPoolExecutor, Future

from .task import MultithreadedTask

logger = logging.getLogger(__name__)


class MultithreadedService:
    """
    A multithread service that executes preprocessing tasks using
    a thread pool.

    Compared to the MultiprocessService, using MultithreadedService
    is lighter weight, as tasks are executed in the same process.
    However, threads are subject to the GIL, so CPU-bound workloads
    may be better executed in the MultiprocessService

    Args:
        num_threads: Number of threads in the pool
    """

    def __init__(self, num_threads: int):
        if num_threads < 1:
            raise ValueError(
                "Cannot create a mutlithreaded "
                "service with {} threads.".format(num_threads)
            )

        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.n_threads = num_threads
        logger.info(f"Started MUltithread Service with {num_threads} threads.")

        # Lock for resizing executor
        self._lock = threading.Lock()

    def shutdown(self) -> None:
        self.executor.shutdown()

    def resize(self, num_threads: int) -> None:
        prev_num_threads = self.n_threads
        with self._lock:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = ThreadPoolExecutor(max_workers=num_threads)
            self.n_threads = num_threads
        logger.info(
            f"Resized Multithreaded Pool from {prev_num_threads}"
            f" to {num_threads} threads"
        )

    def submit(self, task: MultithreadedTask) -> Future:
        with self._lock:
            future = self.executor.submit(task.process)
        return future

    def __del__(self):
        self.shutdown()
