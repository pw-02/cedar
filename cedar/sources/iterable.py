import logging
from typing import Any, Iterable, Tuple, Optional

from cedar.pipes import (
    InProcessPipeVariant,
    Pipe,
    InProcessPipeVariantContext,
    CedarPipeSpec,
    cedar_pipe,
    PipeVariantType,
)

from .source import Source, SourcePipeVariantMixin

logger = logging.getLogger(__name__)


class InProcessIterSourcePipeVariant(
    InProcessPipeVariant, SourcePipeVariantMixin
):
    def __init__(
        self, source: Iterable[Any], rank_spec: Tuple[int, int] = None
    ):
        InProcessPipeVariant.__init__(self, None)
        SourcePipeVariantMixin.__init__(self, rank_spec=rank_spec)
        self.source = source
        self.all_datasamples = None

    def _reset_source_iterator_for_epoch(self):
        it = iter(self.source)
        self.all_datasamples = iter(self.create_datasamples(it, size=1))

    def _iter_impl(self):
        # Reset for next epoch
        # TODO: Make sure this works properly for multiple epochs

        it = iter(self.source)
        all_datasamples = iter(self.create_datasamples(it, size=1))

        logger.info(f"{self.world_size}, {self.rank}")

        while True:
            # NOTE: Once you are done replaying,
            # you start with a completely new data sample

            # If we are not in replay mode
            if not self.replay:
                try:
                    # NOTE: Not a clean fix
                    if self.all_datasamples:
                        ds = next(self.all_datasamples)
                    else:
                        ds = next(all_datasamples)

                    self._update_partition_state(self.num_yielded)
                    ds.sample_id = self.num_yielded
                    self.num_yielded += 1
                    yield ds
                except StopIteration:
                    return

            else:  # We are in replay mode
                if not self.initialized_replay:
                    replay_it = iter(self.source)
                    replay_data = iter(
                        self.create_datasamples(replay_it, size=1)
                    )
                    self.initialized_replay = True
                    curr_ds_id = 0
                try:
                    if not self.partitions_to_replay:
                        self.disable_replay()
                        continue
                    ds = next(replay_data)
                    if not self._should_be_replayed(curr_ds_id):
                        curr_ds_id += 1
                        continue
                    else:
                        self._update_partition_state(curr_ds_id)
                        ds.sample_id = curr_ds_id
                        curr_ds_id += 1
                        yield ds
                except StopIteration:
                    self.disable_replay()
                    continue


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=False,
        mutable_variants=[
            PipeVariantType.INPROCESS,
        ],
        is_fusable=False,
        is_shardable=True,
    )
)
class IterSourcePipe(Pipe):
    """
    A pipe over an iterable source.
    """

    def __init__(
        self,
        source: Iterable[Any],
        rank_spec: Optional[Tuple[int, int]] = None,
        is_random: bool = False,
    ):
        super().__init__("IterSourcePipe", [], is_random=is_random)
        self.source = source
        self.variant_length = None
        self.rank_spec = rank_spec

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        variant = InProcessIterSourcePipeVariant(self.source, self.rank_spec)
        return variant


class IterSource(Source):
    """
    An IterSource represents a data source that
    is any Python Iterable.

    args:
        input (Iterable[Any]): An iterable input
    """

    def __init__(
        self,
        input_source: Iterable[Any],
        rank_spec: Optional[Tuple[int, int]] = None,
    ):
        self.input_source = input_source
        self.rank_spec = rank_spec

    def to_pipe(self) -> Pipe:
        pipe = IterSourcePipe(self.input_source, rank_spec=self.rank_spec)
        pipe.fix()
        return pipe
