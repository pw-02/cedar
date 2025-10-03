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
from PIL import Image
from io import BytesIO

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
    def __init__(self, prefix, transform=None):

        self.bucket = S3Url(prefix).bucket
        self.prefix = S3Url(prefix).key
        self.transform = transform
        self.s3 = boto3.client('s3')
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
    
    def load_image(self, key):
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        img_bytes = obj['Body'].read()
        # img = Image.open(BytesIO(obj["Body"].read())).convert("RGB")
        img_tensor = decode_image(torch.frombuffer(img_bytes, dtype=torch.uint8), mode=ImageReadMode.RGB)
        return img_tensor

    def __getitem__(self, idx):
        key = self.keys[idx]
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        img_bytes = obj['Body'].read()
        # return Image.open(BytesIO(obj["Body"].read())).convert("RGB"), label
        img_tensor = decode_image(torch.frombuffer(img_bytes, dtype=torch.uint8), mode=ImageReadMode.RGB)
        # if self.transform:
        #     img_tensor = self.transform(img_tensor)
        return img_tensor


class InProcessS3SourcePipeVariant(InProcessPipeVariant, SourcePipeVariantMixin):
    def __init__(
        self,
        source: str,
        rank_spec: Optional[Tuple[int, int]],
    ) -> None:
        InProcessPipeVariant.__init__(self, None)
        SourcePipeVariantMixin.__init__(self, rank_spec=rank_spec)
        # self.s3_url = source
        self.source = source
        self.num_yielded = 0
        self.s3_dataset:S3ImageDataset = S3ImageDataset(self.source)
        self.all_datasamples = None

    def _reset_source_iterator_for_epoch(self):
        it = iter(self.s3_dataset.keys)
        self.all_datasamples = self.create_datasamples(it, size=1)
        pass

    def _iter_impl(self):
        # self.num_yielded = 0
        it = iter(self.s3_dataset.keys)
        self.all_datasamples = self.create_datasamples(it, size=1)

        while True:
            try:
                ds = next(self.all_datasamples)

                if isinstance(ds, DataSample):
                    img_path = ds.data
                else:
                    img_path = ds["file_name"]
                
                img = self.s3_dataset.load_image(img_path)

                if isinstance(ds, DataSample):
                    ds.data = img
                    yield ds
                else:
                    yield img
            except StopIteration:
                return


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=False,
        mutable_variants=[
            PipeVariantType.INPROCESS,
        ],
        is_fusable=False,
        is_shardable=True,
        is_fusable_source=False,
    )
)


class S3SourcePipe(Pipe):
    """
    Pipe that iterates over files within an S3 bucket.

    Args:
        source: Sequence of S3 URLs, representing
        files or prefixes within a bucket.
    """

    source: List[str]

    def __init__(
        self,
        source: str,
        rank_spec: Optional[Tuple[int, int]],
        is_random: bool = False,
    ) -> None:
        
        super().__init__("S3SourcePipe", [])  # empty inputs = source pipe
        self.source = source
        self.rank_spec = rank_spec

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        assert self.is_source()
        variant = InProcessS3SourcePipeVariant(self.source, self.rank_spec)
        return variant


class S3ImageSource(Source):
    """
    A source that represents a collection of images stored in S3.
    """

    def __init__(
        self,
        source: str,
        rank_spec: Optional[Tuple[int, int]] = None,
    ) -> None:
        
        self.source = source
        self.rank_spec = rank_spec

    def to_pipe(self) -> Pipe:
        pipe = S3SourcePipe(self.source, rank_spec=self.rank_spec)
        pipe.fix()
        return pipe