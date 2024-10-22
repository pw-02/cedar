"""
Downloads the dataset for this pipeline
"""

import pathlib
import tarfile
import urllib.request

DATASET_NAME = "imagenette2"
DATASET_LOC = "datasets/imagenette2"
DATASET_FILE = "imagenette2.tgz"
DATASET_SOURCE = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"


def download_dataset():
    # Assume if tar file exists, dataset exists
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )

    extract_dir = data_dir / "imagenette2"
    if extract_dir.exists():
        print("Dataset already downloaded...")
        return

    dataset_file = data_dir / DATASET_FILE
    print(dataset_file)
    if dataset_file.is_file():
        return

    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset to {str(dataset_file)}...")
    urllib.request.urlretrieve(DATASET_SOURCE, str(dataset_file))

    tar_path = dataset_file.parent

    print("Extracting dataset...")
    with tarfile.open(dataset_file, "r:gz") as tar:
        tar.extractall(path=tar_path)

    dataset_file.unlink()


if __name__ == "__main__":
    download_dataset()
