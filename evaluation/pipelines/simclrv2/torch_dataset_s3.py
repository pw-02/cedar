import boto3
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode
from io import BytesIO

IMG_HEIGHT = 244
IMG_WIDTH = 244
GAUSSIAN_BLUR_KERNEL_SIZE = 11

class S3ImageDataset(Dataset):
    def __init__(self, bucket, prefix, transform=None, region="us-west-2"):
        self.bucket = bucket
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

class TorchEvalSpec:
    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers

def get_dataset(spec, bucket="sdl-cifar10", prefix="test/"):
    dataset = S3ImageDataset(bucket, prefix, transform=transform)
    dataloader = DataLoader(dataset, batch_size=spec.batch_size, num_workers=spec.num_workers)
    return dataloader

if __name__ == "__main__":
    spec = TorchEvalSpec(8, 0)
    dataloader = get_dataset(spec)
    for batch in dataloader:
        print(batch.shape)  # Should be [8, 1, 244, 244]
