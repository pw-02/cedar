"""
Download the WikiText103 dataset to local filesystem.
"""

import logging
import pathlib
import tarfile
from google.cloud import storage

DATASET_NAME = "cv-corpus-15.0-delta-2023-09-08"
DATASET_LOC = "datasets/commonvoice"
DATASET_FILE = "cv-corpus-15.0-delta-2023-09-08-en.tar"
BUCKET_NAME = "ember-data"
SOURCE_BLOB_NAME = "cv-corpus-15.0-delta-2023-09-08-en.tar"


logger = logging.getLogger(__name__)


def download_if_not_exists(path: pathlib.Path):
    if not path.is_file():
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(SOURCE_BLOB_NAME)
        blob.download_to_filename(str(path))
        print("Downloaded {}".format(str(path)))
    else:
        print("Path already exists: {}".format(str(path)))


def download_dataset() -> None:
    logger.info("Downloading Commonvoice Dataset")
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = data_dir / pathlib.Path(DATASET_FILE)
    zip_dir = dataset_file.parent

    if not (zip_dir / DATASET_NAME).exists():
        print(f"Downloading dataset to {str(dataset_file)}...")
        download_if_not_exists(dataset_file)

        with tarfile.open(dataset_file, "r") as tar:
            tar.extractall(path=str(zip_dir))

        dataset_file.unlink()


if __name__ == "__main__":
    download_dataset()
