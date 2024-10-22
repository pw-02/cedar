from typing import Optional
from .registry import register_optimizer_pipe
from ..context import InProcessPipeVariantContext
from ..pipe import (
    Pipe,
)
from ..variant import (
    InProcessPipeVariant,
    PipeVariant,
)


@register_optimizer_pipe("NoopOptimizerPipe")
class NoopOptimizerPipe(Pipe):
    """
    A noop pipe, that effectively just forwards the output of the input pipe.
    Intended to be used as an optimization, and not directly defined
    within the feature.

    Primarily intenteded for testing.
    """

    def __init__(
        self, input_pipe: Optional[Pipe] = None, is_random: bool = False
    ):
        if input_pipe:
            super().__init__(
                "NoopOptimizerPipe", [input_pipe], is_random=is_random
            )
        else:
            super().__init__("NoopOptimizerPipe", [], is_random=is_random)

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        variant = InProcessNoopOptimizerPipeVariant(
            self.input_pipes[0].pipe_variant
        )
        return variant

    def _check_mutation(self) -> None:
        super()._check_mutation()

        if len(self.input_pipes) != 1:
            raise RuntimeError("NoopOptimizerPipe only accepts one input.")


class InProcessNoopOptimizerPipeVariant(InProcessPipeVariant):
    def __init__(self, input_pipe_variant: Optional[PipeVariant]):
        super().__init__(input_pipe_variant)

    def _iter_impl(self):
        for x in self.input_pipe_variant:
            yield x
