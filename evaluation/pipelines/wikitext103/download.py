"""
Download the WikiText103 dataset to local filesystem.
"""

import logging
import pathlib
import urllib.request
import zipfile

DATASET_NAME = "wikitext103"
DATASET_LOC = "datasets/wikitext103"
DATASET_FILE = "wikitext-103-v1.zip"
DATASET_SOURCE = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"  # noqa: E501


logger = logging.getLogger(__name__)


def download_if_not_exists(url: str, path: pathlib.Path):
    if not path.is_file():
        urllib.request.urlretrieve(url, str(path))
        print("Downloaded {}".format(str(path)))
    else:
        print("Path already exists: {}".format(str(path)))


def download_dataset() -> None:
    logger.info("Downloading Wikitext103 Dataset")
    data_dir = (
        pathlib.Path(__file__).resolve().parents[2].joinpath(DATASET_LOC)
    )
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = data_dir / pathlib.Path(DATASET_FILE)
    zip_dir = dataset_file.parent

    print(zip_dir)
    if not (zip_dir / "wikitext-103").exists():
        print(f"Downloading dataset to {str(dataset_file)}...")
        urllib.request.urlretrieve(DATASET_SOURCE, str(dataset_file))
        logger.info("Extracting Wikitext103 data from zip file.")
        with zipfile.ZipFile(dataset_file, "r") as zip_ref:
            zip_ref.extractall(path=zip_dir)
        logger.info("Done extracting Wikitext103 data from zip file.")

        dataset_file.unlink()


if __name__ == "__main__":
    download_dataset()
