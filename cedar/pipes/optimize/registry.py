import logging
from typing import Type, Callable


logger = logging.getLogger(__name__)


class OptimizerPipeRegistry:
    _registered_pipes = {}

    @classmethod
    def register_pipe(cls, name: str, pipe_cls: Type):
        if name in cls._registered_pipes:
            logger.warning(
                f"Optimizer Pipe {name} already registered. Overwriting."
            )
        logger.info(f"Registering Optimizer Pipe {name}.")
        cls._registered_pipes[name] = pipe_cls

    @classmethod
    def get_pipe(cls, name: str):
        if name not in cls._registered_pipes:
            raise ValueError(f"Pipe {name} not reigstered.")
        return cls._registered_pipes[name]


def register_optimizer_pipe(name: str) -> Callable[[Type], Type]:
    """
    Decorator to register an optimizer pipe.
    """

    def decorator(optimizer_cls: Type) -> Type:
        OptimizerPipeRegistry.register_pipe(name, optimizer_cls)
        return optimizer_cls

    return decorator
