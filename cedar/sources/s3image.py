from typing import List, Union, Iterator, Tuple, Optional, Callable
import glob
from urllib.parse import urlparse
import ray
import logging
from torch.utils.data import Dataset
from torchdata.datapipes.iter import FileLister
from torchvision.io import decode_image, ImageReadMode
import torch
import boto3


from cedar.pipes import (
    Pipe,
    InProcessPipeVariant,
    InProcessPipeVariantContext,
    PipeVariantType,
    CedarPipeSpec,
    RayDSPipeVariantContext,
    RayDSPipeVariant,
    cedar_pipe,
    DataSample,
)
from .source import Source, SourcePipeVariantMixin

logger = logging.getLogger(__name__)


class S3Url(object):
    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()


class S3ImageDataset(Dataset):
    def __init__(self, prefix, transform=None, region="us-west-2"):

        self.bucket = S3Url(prefix).bucket
        self.prefix = prefix
        self.transform = transform
        self.s3 = boto3.client('s3', region_name=region)
        # List all S3 keys (image files) under the prefix
        self.keys = self._list_keys()

    def _list_keys(self):
        paginator = self.s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
        keys = []
        for page in page_iterator:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # or other formats you expect
                    keys.append(key)
        return keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        img_bytes = obj['Body'].read()
        img_tensor = decode_image(torch.frombuffer(img_bytes, dtype=torch.uint8), mode=ImageReadMode.RGB)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor


class InProcessS3ListerPipeVariant(InProcessPipeVariant, SourcePipeVariantMixin):
    def __init__(
        self,
        s3_urls: Union[List[str], InProcessPipeVariant],
        rank_spec: Optional[Tuple[int, int]],
        region: str = "us-west-2",
    ) -> None:
        InProcessPipeVariant.__init__(self, None)
        SourcePipeVariantMixin.__init__(self, rank_spec=rank_spec)
        self.s3_urls = s3_urls
        self.region = region
        self.num_yielded = 0

        # Accepts list of s3://bucket/prefix strings
        self.dp = S3ImageDataset(s3_urls[0], region=self.region)

        # (optionally: Mapper to wrap S3 object data in your custom DataSample)
        # self.dp = Mapper(self.dp, your_datasample_wrapper)
        self.all_datasamples = None

    def _reset_source_iterator_for_epoch(self):
        it = iter(self.dp)
        self.all_datasamples = self.create_datasamples(it, size=1)

    def _iter_impl(self):
        # self.num_yielded = 0
        it = iter(self.dp)
        all_datasamples = self.create_datasamples(it, size=1)

        while True:
            # We are not in replay mode
            if not self.replay:
                try:
                    if self.all_datasamples:
                        ds = next(self.all_datasamples)
                    else:
                        ds = next(all_datasamples)

                    if isinstance(ds, DataSample):
                        self._update_partition_state(self.num_yielded)
                        ds.sample_id = self.num_yielded
                        self.num_yielded += 1
                    yield ds
                except StopIteration:
                    return

            else:  # We are in replay mode
                if self.sharded:
                    raise NotImplementedError("Can't shard with replay")
                if not self.initialized_replay:
                    new_dp = FileLister(
                        root=self.root, recursive=self.recursive
                    )
                    replay_it = iter(new_dp)
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
        is_fusable_source=True,
        fusable_source_variants=[PipeVariantType.RAY_DS],
    )
)


class S3ListerPipe(Pipe):
    """
    Pipe that iterates over files within an S3 bucket.

    Args:
        source: Sequence of S3 URLs, representing
        files or prefixes within a bucket.
    """

    source: Union[List[str], Pipe]

    def __init__(
        self,
        source: Union[List[str], Pipe],
        rank_spec: Optional[Tuple[int, int]],
        region: str = "us-west-2",
        is_random: bool = False,
    ) -> None:
        if isinstance(source, Pipe):
            super().__init__("S3ListerPipe", [source], is_random=is_random)
            self.source = []
        else:
            # empty inputs denotes source pipe
            super().__init__("S3ListerPipe", [])
            self.source = source
        self.region = region
        self.rank_spec = rank_spec

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        if self.is_source():
            variant = InProcessS3ListerPipeVariant(
                self.source, self.rank_spec, region=self.region
            )
        else:
            if (
                not self.input_pipes[0].pipe_variant_type
                != PipeVariantType.INPROCESS
            ):
                raise NotImplementedError
            variant = InProcessS3ListerPipeVariant(
                self.input_pipes[0].pipe_variant,
                self.rank_spec,
                region=self.region,
            )
        return variant

    def _to_ray_ds(
        self, variant_ctx: RayDSPipeVariantContext
    ) -> RayDSPipeVariant:
        if self._pipes_to_fuse is None:
            raise RuntimeError("Cannot use S3ListerPipe RayDS without Fuse")
        if not isinstance(self.source, list):
            raise NotImplementedError
        if not self.is_source():
            raise NotImplementedError

        fns = [p.get_fused_callable() for p in self._pipes_to_fuse]
        logger.info("Creating Ray DS Source!!!")
        return RayDSLocalFileListerPipeVariant(
            self.source, fns, self.rank_spec
        )

    def fuse_into(self, pipes: List[Pipe]) -> None:
        self._pipes_to_fuse = pipes


class RayDSLocalFileListerPipeVariant(
    RayDSPipeVariant, SourcePipeVariantMixin
):
    def __init__(
        self,
        source: List[str],
        fused_fns: List[Callable],
        rank_spec: Optional[Tuple[int, int]],
    ) -> None:
        RayDSPipeVariant.__init__(self, None)
        SourcePipeVariantMixin.__init__(self, rank_spec=rank_spec)
        self.source = source

        if len(source) != 1:
            raise NotImplementedError

        files = glob.glob(f"{source[0]}/*")

        self.dataset = ray.data.from_items(files)
        for fn in fused_fns:
            print(fn)
            self.dataset = self.dataset.map(fn)

    def _iter_impl(self):
        self.dataset_iter = iter(self.dataset.iter_rows())
        datasamples = self.create_datasamples(self.dataset_iter, size=1)
        while True:
            try:
                ds = next(datasamples)
                yield ds
            except StopIteration:
                return

    def _reset_source_iterator_for_epoch(self):
        pass
class S3ImageSource(Source):
    """
    A source that represents a collection of images stored in S3.
    """

    def __init__(
        self,
        source: Union[str, List[str]],
        rank_spec: Optional[Tuple[int, int]] = None,
    ) -> None:
        if isinstance(source, str):
            self.source = [
                source,
            ]
        elif isinstance(source, list):
            self.source = source
        else:
            raise TypeError("Invalid S3ImageSource input.")
        self.rank_spec = rank_spec

    def to_pipe(self) -> Pipe:
        pipe = S3ListerPipe(self.source, rank_spec=self.rank_spec)
        pipe.fix()
        return pipe

# @cedar_pipe(
#     CedarPipeSpec(
#         is_mutable=False,
#         mutable_variants=[
#             PipeVariantType.INPROCESS,
#         ],
#         is_fusable=False,
#         is_shardable=True,
#     )
# )