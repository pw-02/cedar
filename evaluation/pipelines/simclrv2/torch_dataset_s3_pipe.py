import torch
from torchdata.datapipes.iter import IterableWrapper, S3FileLister, S3FileLoader, Mapper
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode
from torch.utils.data import DataLoader

IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 11

class TorchEvalSpec:
    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

def to_float(x):
    return x.to(torch.float32) / 255.0

def build_s3_datapipe(bucket: str, prefix: str, spec: TorchEvalSpec):
    # 1. List files in S3 bucket using a DataPipe of S3 URL prefixes
    s3_prefix = f"s3://{bucket}/{prefix}".rstrip("/")
    urls_dp = IterableWrapper([s3_prefix])
    datapipe = S3FileLister(urls_dp, region="us-west-2")
    datapipe = S3FileLoader(datapipe)
    
    # 2. Load image tensor from bytes
    def load_image(item):
        path, stream = item
        img_bytes = stream.read()
        img = decode_image(torch.frombuffer(img_bytes, dtype=torch.uint8), mode=ImageReadMode.RGB)
        return img

    datapipe = Mapper(datapipe, load_image)
    
    # 3. Compose transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Grayscale(num_output_channels=1),
        transforms.GaussianBlur(GAUSSIAN_BLUR_KERNEL_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    datapipe = Mapper(datapipe, transform)
    datapipe = Mapper(datapipe, to_float)
    return datapipe

def get_dataset(spec: TorchEvalSpec, bucket="sdl-cifar10", prefix="test/"):
    datapipe = build_s3_datapipe(bucket, prefix, spec)
    dataloader = DataLoader(
        datapipe, batch_size=spec.batch_size, num_workers=spec.num_workers
    )
    return dataloader

if __name__ == "__main__":
    spec = TorchEvalSpec(8, 0)
    dataloader = get_dataset(spec)
    for x in dataloader:
        print(x.shape)   # Should be [8, 1, 244, 244]
