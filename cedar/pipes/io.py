from typing import Any, Optional, Iterator, Union
from cedar.service import MultithreadedTask

from torch.utils.data.datapipes.utils.common import StreamWrapper

from .pipe import (
    Pipe,
)
from .variant import (
    InProcessPipeVariant,
    PipeVariant,
    MultithreadedPipeVariant,
)
from .context import (
    PipeVariantType,
    InProcessPipeVariantContext,
    MultithreadedPipeVariantContext,
)
from .common import DataSample, cedar_pipe, CedarPipeSpec
from torchvision.io import read_image
from torchvision.io import ImageReadMode

import requests


class InProcessFileOpenerPipeVariant(InProcessPipeVariant):
    def __init__(self, input_pipe_variant: PipeVariant, mode: str):
        super().__init__(input_pipe_variant)
        self.mode = mode

    def _iter_impl(self):
        for x in self.input_pipe_variant:
            if isinstance(x, DataSample):
                if not x.dummy:
                    # iterate over paths, use torch StreamWrapper to wrap
                    # bytestream to ensure files are properly closed
                    x.data = StreamWrapper(open(x.data, self.mode))
                yield x
            else:
                yield StreamWrapper(open(x, self.mode))


class FileOpenerPipe(Pipe):
    """
    Given a file path, opens the file and yields
    a tuple containing the file path and the file stream
    """

    def __init__(
        self,
        input_pipe: Pipe,
        mode: str = "r",
        tag: Optional[str] = None,
        is_random: bool = False,
    ) -> None:
        super().__init__(
            "FileOpenerPipe", [input_pipe], tag=tag, is_random=is_random
        )
        self.mode = mode

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        if self.input_pipes[0].pipe_variant_type != PipeVariantType.INPROCESS:
            raise NotImplementedError
        return InProcessFileOpenerPipeVariant(
            self.input_pipes[0].pipe_variant, self.mode
        )


class LineReaderPipe(Pipe):
    """
    Given a FileOpener pipe, yield the next line from the file
    """

    def __init__(
        self,
        input_pipe: Pipe,
        tag: Optional[str] = None,
        is_random: bool = False,
    ):
        super().__init__(
            "LineReaderPipe", [input_pipe], tag=tag, is_random=is_random
        )

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        if self.input_pipes[0].pipe_variant_type != PipeVariantType.INPROCESS:
            raise NotImplementedError
        return InProcessLineReaderPipeVariant(self.input_pipes[0].pipe_variant)


class InProcessLineReaderPipeVariant(InProcessPipeVariant):
    def __init__(self, input_pipe_variant: PipeVariant):
        super().__init__(input_pipe_variant)

    def _iter_impl(self):
        for file in self.input_pipe_variant:
            stream = self._read_lines(file.data)
            stream = self._decode(stream)
            stream = self._strip(stream)

            # Create a new datasample for each line
            for line in stream:
                yield DataSample(line)

    @staticmethod
    def _read_lines(
        file: StreamWrapper,
    ) -> Union[Iterator[str], Iterator[bytes]]:
        try:
            yield from file
        finally:
            file.close()

    @staticmethod
    def _decode(
        stream: Union[Iterator[str], Iterator[bytes]]
    ) -> Iterator[str]:
        for line in stream:
            if isinstance(line, bytes):
                yield line.decode("utf-8")
            else:
                yield line

    @staticmethod
    def _strip(stream: Iterator[str]):
        for line in stream:
            yield line.strip("\r\n")


@cedar_pipe(
    CedarPipeSpec(
        is_mutable=True,
        mutable_variants=[
            PipeVariantType.INPROCESS,
            PipeVariantType.MULTITHREADED,
        ],
    )
)


class S3ImageReaderPipe(Pipe):
    """
    Given a pipe containing S3 URLs, downloads the contents of the image
    at that URL. Yields the image as a torch.Tensor.

    Args:
        request_timeout: Timeout in seconds for downloading an image from S3.
            Default is 5 seconds.
        mode: Mode to read in image
        tag: Tag to assign to this pipe.
        is_random: Whether this pipe is random.
    Output Format:
    Yields torch.Tensor of shape (image_channels, image_height, image_width).
    The values of the output tensor are dtype=uint8 in [0, 255].
    Ordering of channels is RGB.
    """

    def __init__(
        self,
        input_pipe: Pipe,
        request_timeout: int = 5,
        mode: Optional[ImageReadMode] = None,
        tag: Optional[str] = None,
        is_random: bool = False,
    ):
        super().__init__(
            "S3ImageReaderPipe", [input_pipe], tag=tag, is_random=is_random
        )

        if request_timeout < 1:
            raise RuntimeError(
                f"S3ImageReaderPipe cannot have request timeout of less than 1.\n\
                               Expected value >= 1, got {request_timeout}"
            )

        self.request_timeout = request_timeout
        if mode is None:
            self.mode = ImageReadMode.UNCHANGED
        else:
            self.mode = mode

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("S3ImageReader Pipe only accepts one input pipe.")

        variant = InProcessS3ImageReaderPipeVariant(
            self.input_pipes[0].pipe_variant,
            self.request_timeout,
            self.mode,
        )

        return variant

    def _to_multithreaded(
        self, variant_ctx: MultithreadedPipeVariantContext
    ) -> MultithreadedPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("S3ImageReader Pipe only accepts one input pipe.")

        variant = MultithreadedImageReaderTask(
            self.input_pipes[0].pipe_variant,
            self.request_timeout,
            self.mode,
        )
        return variant
    
class InProcessS3ImageReaderPipeVariant(InProcessPipeVariant):
    def __init__(
        self,
        input_pipe_variant: PipeVariant,
        request_timeout: int,
        mode: ImageReadMode,
    ):
        super().__init__(input_pipe_variant)
        self.request_timeout = request_timeout
        self.mode = mode

    def _iter_impl(self):
        while True:
            try:
                x = next(self._input_iter)
                if isinstance(x, DataSample):
                    if not x.dummy:
                        url = x.data
                        resp = requests.get(url, timeout=self.request_timeout)
                        img_bytes = resp.content
                        x.data = read_image(
                            img_bytes, mode=self.mode
                        )
                    yield x
                else:
                    url = x
                    resp = requests.get(url, timeout=self.request_timeout)
                    img_bytes = resp.content
                    yield read_image(img_bytes, mode=self.mode)
            except StopIteration:
                return
            
class MultithreadedS3ImageReaderTask(MultithreadedTask):
    def __init__(self, input_url: str, request_timeout: int, mode: ImageReadMode):
        super().__init__(input_url)
        self.request_timeout = request_timeout
        self.mode = mode

    def process(self) -> Any:
        resp = requests.get(self.input_data, timeout=self.request_timeout)
        img_bytes = resp.content
        return read_image(img_bytes, mode=self.mode)





@cedar_pipe(
    CedarPipeSpec(
        is_mutable=True,
        mutable_variants=[
            PipeVariantType.INPROCESS,
            PipeVariantType.MULTITHREADED,
        ],
    )
)

class ImageReaderPipe(Pipe):
    """
    Given a FileOpener pipe, yields the decoded tensor
    representation of the images specified in the FileOpener pipe.

    Args:
        mode: Mode to read in image

    Output Format:
    Yields torch.Tensor of shape (image_channels, image_height, image_width).
    The values of the output tensor are dtype=uint8 in [0, 255].
    Ordering of channels is RGB.
    """

    def __init__(
        self,
        input_pipe: Pipe,
        mode: Optional[ImageReadMode] = None,
        tag: Optional[str] = None,
        is_random: bool = False,
    ):
        super().__init__(
            "ImageReaderPipe", [input_pipe], tag=tag, is_random=is_random
        )
        if mode is None:
            self.mode = ImageReadMode.UNCHANGED
        else:
            self.mode = mode

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        if self.input_pipes[0].pipe_variant_type != PipeVariantType.INPROCESS:
            raise NotImplementedError
        return InProcessImageReaderPipeVariant(
            self.input_pipes[0].pipe_variant, self.mode
        )

    def _to_multithreaded(
        self, variant_ctx: MultithreadedPipeVariantContext
    ) -> MultithreadedPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError

        return MultithreadedImageReaderPipeVariant(
            self.input_pipes[0].pipe_variant, variant_ctx, self.mode
        )


class InProcessImageReaderPipeVariant(InProcessPipeVariant):
    def __init__(self, input_pipe_variant: PipeVariant, mode: ImageReadMode):
        super().__init__(input_pipe_variant)
        self.mode = mode

    def _iter_impl(self):
        while True:
            try:
                x = next(self._input_iter)
                if isinstance(x, DataSample):
                    if not x.dummy:
                        x.data = read_image(x.data, mode=self.mode)
                    yield x
                else:
                    yield read_image(x, mode=self.mode)
            except StopIteration:
                return


class MultithreadedImageReaderTask(MultithreadedTask):
    def __init__(self, input_data: Any, mode: ImageReadMode):
        super().__init__(input_data)
        self.mode = mode

    def process(self) -> Any:
        return read_image(self.input_data, mode=self.mode)


class MultithreadedImageReaderPipeVariant(MultithreadedPipeVariant):
    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        variant_ctx: MultithreadedPipeVariantContext,
        mode: ImageReadMode,
    ):
        super().__init__(input_pipe_variant, variant_ctx)
        self.mode = mode

    def _create_task(self, input_data: Any) -> MultithreadedTask:
        return MultithreadedImageReaderTask(input_data, self.mode)


class WebReaderPipe(Pipe):
    """
    Given a pipe containing URLs, downloads the contents of the web page
    at that URL. Yields the content as a bytestream.

    Output Format:
    Yields bytestream of format bytes with no particular specifications.
    The user is expected to convert the output into other objects for
    future use by using the BytesIO class from the io module.
    """

    def __init__(
        self,
        input_pipe: Pipe,
        request_timeout: int,
        tag: Optional[str] = None,
        is_random: bool = False,
    ):
        super().__init__(
            "WebReaderPipe", [input_pipe], tag=tag, is_random=is_random
        )

        if request_timeout < 1:
            raise RuntimeError(
                f"WebReaderPipe cannot have request timeout of less than 1.\n\
                               Expected value >= 1, got {request_timeout}"
            )

        self.request_timeout = request_timeout

    def _to_inprocess(
        self, variant_ctx: InProcessPipeVariantContext
    ) -> InProcessPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("WebReader Pipe only accepts one input pipe.")

        variant = InProcessWebReaderPipevariant(
            self.input_pipes[0].pipe_variant, self.request_timeout
        )

        return variant

    def _to_multithreaded(
        self, variant_ctx: MultithreadedPipeVariantContext
    ) -> MultithreadedPipeVariant:
        if len(self.input_pipes) != 1:
            raise RuntimeError("WebReader Pipe only accepts one input pipe.")

        variant = MultithreadedWebReaderPipeVariant(
            self.input_pipes[0].pipe_variant,
            self.request_timeout,
            variant_ctx,
        )
        return variant


class InProcessWebReaderPipevariant(InProcessPipeVariant):
    def __init__(self, input_pipe_variant: PipeVariant, request_timeout: int):
        super().__init__(input_pipe_variant)
        self.request_timeout = request_timeout

    def _load_url(self, url: str):
        # TODO: validate URL here
        resp = requests.get(url, timeout=self.request_timeout)
        return resp.content

    def _iter_impl(self):
        for url_ds in self.input_pipe_variant:
            if not url_ds.dummy:
                url = url_ds.data
                url_ds.data = (url, self._load_url(url))
            yield url_ds


class MultithreadedWebReaderTask(MultithreadedTask):
    def __init__(self, input_url: str, request_timeout: int):
        super().__init__(input_url)
        self.request_timeout = request_timeout

    def _load_url(self, url: str):
        resp = requests.get(url, timeout=self.request_timeout)
        return resp.content

    def process(self) -> Any:
        return (self.input_data, self._load_url(self.input_data))


class MultithreadedWebReaderPipeVariant(MultithreadedPipeVariant):
    def __init__(
        self,
        input_pipe_variant: Optional[PipeVariant],
        request_timeout: int,
        variant_ctx: MultithreadedPipeVariantContext,
    ):
        super().__init__(input_pipe_variant, variant_ctx=variant_ctx)
        self.request_timeout = request_timeout

    def _create_task(self, input_data: Any) -> MultithreadedTask:
        return MultithreadedWebReaderTask(input_data, self.request_timeout)
