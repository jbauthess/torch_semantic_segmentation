"""jsrt 'Segmentation02' dataset as described here : http://imgcom.jsrt.or.jp/minijsrtdb/

Chest X-ray images (247 images) from the "Standard Digital Image Database" of
the Japan Society of Japan Radiological Technology
The label image (teacher image) is an image of 255 pixels in the lung area, 85 in the heart area,
170 in the lung field, and 0 in vitro, and is recorded in PNG file format.

There is no medical basis for the definition and determination of the lung area of the label data
because it has not been medically supervised.
"""

import os
import tempfile
import urllib.request
import zipfile
from logging import getLogger
from pathlib import Path

from src.dataset.custom_image_mask_dataset import CustomImageMaskDataset

DATASET_URL = r"http://imgcom.jsrt.or.jp/imgcom/wp-content/uploads/2019/07/segmentation02.zip"
DATASET_NAME = "segmentation02"
DATASET_BASE_FOLDER = "segmentation"
IMAGE_FOLDER_PREFIX = "org"
MASK_FOLDER_PREFIX = "label"

logger = getLogger()


def download_and_uncompress_zip_archive(url: str, output_folder_path: Path) -> None:
    """download zip archive and decompress it into 'output_folder_path'

    Args:
        url (str): url of the archive to download
        output_folder_path (Path): output folder that will contain
                                   uncompressed data
    """
    zip_path = Path(tempfile.gettempdir()).joinpath("archive.zip")

    # Download the ZIP file
    logger.info("download zip file at %s", url)
    urllib.request.urlretrieve(url, zip_path)

    # Extract it
    logger.info("uncompress files at %s", str(output_folder_path))
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(output_folder_path)

    # (Optional) remove the ZIP file after extraction
    os.remove(zip_path)


class JsrtSegmentation02Dataset(CustomImageMaskDataset):
    """JSRT Segmentation02 Dataset"""

    def __init__(self, root: Path, download: bool = True, image_set: str = "train"):
        """Initialization

        Args:
            root (Path): Root directory of dataset where directory jsrt Segmentation02
                        dataset exists or will be saved to if download is set to True.

            download (bool): If true, downloads the dataset from the internet and puts
                             it in root directory. If dataset is already downloaded,
                             it is not downloaded again.

            image_set (string, optional) – Select the image_set to use, "train", "train_s", "test"
                                           or "all" if you want to use all images.
        """
        # Downlaod the dataset if missing
        missing_dataset = not root.joinpath(DATASET_NAME).exists()
        if missing_dataset and not download:
            raise FileNotFoundError(
                f"jsrt dataset was not found unside {root} location. Activate download option ?"
            )

        if missing_dataset and download:
            download_and_uncompress_zip_archive(DATASET_URL, root)

        # define image and mask folder paths
        image_folder = mask_folder = None
        match image_set:
            case "all":
                image_folder = (
                    root.joinpath(DATASET_NAME)
                    .joinpath(DATASET_BASE_FOLDER)
                    .joinpath(IMAGE_FOLDER_PREFIX)
                )
                mask_folder = (
                    root.joinpath(DATASET_NAME)
                    .joinpath(DATASET_BASE_FOLDER)
                    .joinpath(MASK_FOLDER_PREFIX)
                )

            case "train" | "test" | "train_s":
                image_folder = (
                    root.joinpath(DATASET_NAME)
                    .joinpath(DATASET_BASE_FOLDER)
                    .joinpath(IMAGE_FOLDER_PREFIX + "_" + image_set)
                )
                mask_folder = (
                    root.joinpath(DATASET_NAME)
                    .joinpath(DATASET_BASE_FOLDER)
                    .joinpath(MASK_FOLDER_PREFIX + "_" + image_set)
                )

            case _:
                raise ValueError(
                    f"{image_set} is not a valid image set for jsrt segmentatiin02 dataset!"
                )

        # instanciqte the CustomImageMaskDataset
        super().__init__(image_folder, mask_folder)
