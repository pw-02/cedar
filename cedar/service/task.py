import abc
from typing import Any


class Task(abc.ABC):
    """
    A Task represents a discrete unit of processing, meant to be offloaded
    to an executor.
    """

    @abc.abstractmethod
    def process(self) -> Any:
        pass


class MultiprocessTask(Task):
    def __init__(self, input_data: Any) -> None:
        self.input_data = input_data


class MultithreadedTask(Task):
    def __init__(self, input_data: Any) -> None:
        self.input_data = input_data
