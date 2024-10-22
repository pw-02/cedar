from typing import List, Tuple, Optional, Callable

from cedar.pipes import (
    Pipe,
    cedar_pipe,
    CedarPipeSpec,
    PipeVariantType,
    TFPipeVariantContext,
    TFPipeVariant,
)
from .source import Source, SourcePipeVariantMixin

import tensorflow as tf


class TFLocalLineSource(Source):
    def __init__(
        self, source: str, rank_spec: Optional[Tuple[int, int]] = None
    ):
        if not isinstance(source, str):
            raise TypeError("Invalid LocalLineSource Input")

        self.source = source
        self.rank_spec = rank_spec

    def to_pipe(self) -> Pipe:
        pipe = TFLocalLinePipe(self.source, rank_spec=self.rank_spec)
        pipe.fix()
        return pipe


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=False,
        mutable_variants=[
            PipeVariantType.TF,
        ],
        is_fusable=True,
        is_shardable=True,
    )
)
class TFLocalLinePipe(Pipe):
    def __init__(
        self,
        source: str,
        rank_spec: Optional[Tuple[int, int]] = None,
        is_random: bool = False,
    ):
        super().__init__("TFLocalLinePipe", [], is_random=is_random)
        self.source = source
        self.rank_spec = rank_spec
        self._is_tf = True

    def _to_tf(self, variant_ctx: TFPipeVariantContext) -> TFPipeVariant:
        if self._pipes_to_fuse is not None:
            fns = [p.get_fused_callable() for p in self._pipes_to_fuse]
            variant = TFLocalLinePipeVariant(
                self.source, variant_ctx, self.rank_spec, fns
            )
        else:
            variant = TFLocalLinePipeVariant(
                self.source, variant_ctx, self.rank_spec
            )
        return variant

    def fuse_into(self, pipes: List[Pipe]) -> None:
        self._pipes_to_fuse = pipes


class TFLocalLinePipeVariant(TFPipeVariant, SourcePipeVariantMixin):
    def __init__(
        self,
        source: str,
        variant_ctx: TFPipeVariantContext,
        rank_spec: Optional[Tuple[int, int]],
        fused_fns: Optional[List[Callable]] = None,
    ):
        TFPipeVariant.__init__(self, None)
        SourcePipeVariantMixin.__init__(self, rank_spec=rank_spec)
        self.source = source

        self.dataset = tf.data.TextLineDataset(source)
        if fused_fns is not None:
            for fn in fused_fns:
                self.dataset = self.dataset.map(
                    fn, variant_ctx.num_parallel_calls
                )
        self.dataset_iter = None

    def _iter_impl(self):
        self.dataset_iter = iter(self.dataset)
        datasamples = self.create_tf_datasamples(self.dataset_iter, size=1)
        while True:
            try:
                ds = next(datasamples)
                yield ds
            except StopIteration:
                return

    def _reset_source_iterator_for_epoch(self):
        pass
