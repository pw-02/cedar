import logging

from concurrent.futures import ProcessPoolExecutor, Future

from .task import MultiprocessTask

logger = logging.getLogger(__name__)


class MultiprocessService:
    """
    A multiprocess service that executes preprocessing tasks
    using a pool of workers.

    Args:
        num_workers: Number of workers in the pool
    """

    def __init__(self, num_workers: int):
        if num_workers < 1:
            raise ValueError(
                "Cannot create a MultiprocessService with {} workers".format(
                    num_workers
                )
            )
        self.executor = ProcessPoolExecutor(max_workers=num_workers)
        logger.info(
            f"Started Multiprocess Service with {num_workers} workers."
        )

    def shutdown(self):
        self.executor.shutdown()

    def submit(self, task: MultiprocessTask) -> Future:
        future = self.executor.submit(task.process)
        return future

    def __del__(self):
        self.shutdown()
