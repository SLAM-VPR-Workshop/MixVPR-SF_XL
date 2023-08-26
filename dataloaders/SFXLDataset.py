import os
import torch
import random
import logging
import numpy as np
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as T
from collections import defaultdict
from glob import glob


ImageFile.LOAD_TRUNCATED_IMAGES = True
BASE_PATH = "../data/datasets/SF_XL/processed/train"


def read_images_paths(dataset_folder, get_abs_path=False):
    """Find images within 'dataset_folder' and return their relative paths as a list.
    If there is a file 'dataset_folder'_images_paths.txt, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over large folders can be slow.

    Parameters
    ----------
    dataset_folder : str, folder containing JPEG images
    get_abs_path : bool, if True return absolute paths, otherwise remove
        dataset_folder from each path

    Returns
    -------
    images_paths : list[str], paths of JPEG images within dataset_folder
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        logging.debug(
            f"Reading paths of images within {dataset_folder} from {file_with_paths}"
        )
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [os.path.join(dataset_folder, path) for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(
                f"Image with path {images_paths[0]} "
                f"does not exist within {dataset_folder}. It is likely "
                f"that the content of {file_with_paths} is wrong."
            )
    else:
        logging.debug(f"Searching images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        if len(images_paths) == 0:
            raise FileNotFoundError(
                f"Directory {dataset_folder} does not contain any JPEG images"
            )

    if not get_abs_path:  # Remove dataset_folder from the path
        images_paths = [p[len(dataset_folder) + 1 :] for p in images_paths]

    return images_paths


class SFXLDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_folder: str = BASE_PATH,
        M: int = 10,
        alpha: int = 30,
        min_images_per_partition: int = 10,
        num_samples: int = 4,
    ):
        """
        Parameters (please check our paper for a clearer explanation of the parameters).
        ----------
        args : args for data augmentation
        dataset_folder : str, the path of the folder with the train images.
        M : int, the length of the side of each cell in meters.
        alpha : int, size of each partition in degrees.
        min_images_per_partition : int, minimum number of image in a partition.
        number_sample : int, the number of images sampled for descriptor space distribution in a partition.
        """
        super().__init__()
        self.M = M
        self.alpha = alpha
        self.dataset_folder = dataset_folder
        self.num_samples = num_samples

        # dataset_name should be either "processed", "small" or "raw", if you're using SF-XL
        dataset_name = os.path.basename(dataset_folder)
        filename = f"cache/{dataset_name}_M{M}_alpha{alpha}_mipc{min_images_per_partition}.torch"
        if not os.path.exists(filename):
            os.makedirs("cache", exist_ok=True)
            logging.info(
                f"Cached dataset {filename} does not exist, I'll create it now."
            )
            self.initialize(
                dataset_folder, M, alpha, min_images_per_partition, filename
            )

        self.images_per_partition = torch.load(filename)
        self.partition_ids = list(self.images_per_partition.keys())
        self.img_list = []
        for val in self.images_per_partition.values():
            self.img_list.extend(val)

    @staticmethod
    def open_image(path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int):
        # Pick num_samples random images from this partition.
        image_chosen = self.img_list[index]
        image_metadata = image_chosen.split("@")
        utmeast_utmnorth_heading = (image_metadata[1], image_metadata[2], image_metadata[9])
        partition_id = SFXLDataset.get__partition_id(*utmeast_utmnorth_heading, self.M, self.alpha)
        
        selected_image_paths = random.sample(
            self.images_per_partition[partition_id], self.num_samples
        )

        image_tensors = []

        for image_path in selected_image_paths:
            image_path = os.path.join(self.dataset_folder, image_path)
            try:
                pil_image = SFXLDataset.open_image(image_path)
            except Exception as e:
                logging.info(
                    f"ERROR image {image_path} couldn't be opened, it might be corrupted."
                )
                raise e

            tensor_image = T.functional.to_tensor(pil_image)
            assert tensor_image.shape == torch.Size(
                [3, 512, 512]
            ), f"Image {image_path} should have shape [3, 512, 512] but has {tensor_image.shape}."

            if self.augmentation_device == "cpu":
                tensor_image = self.transform(tensor_image)

            image_tensors.append(tensor_image)  # Increase the dimension by adding an extra axis

        # Stack all the image tensors along the new axis
        stacked_image_tensors = torch.stack(image_tensors)

        return stacked_image_tensors, torch.tensor(partition_id).repeat(self.num_samples)

    def get_images_num(self) -> int:
        """Return the number of images within the dataset."""
        return sum([len(val) for val in self.images_per_partition.values()])

    def __len__(self) -> int:
        """Return the number of partitions within the dataset."""
        return len(self.images_per_partition)

    @staticmethod
    def initialize(
        dataset_folder, M, alpha, min_images_per_partition, filename
    ) -> None:
        """
        The `initialize` function takes in a dataset folder, parameters for partitioning the dataset,
        and a filename, and performs various operations to initialize and save a dictionary of images
        grouped by partition.

        :param dataset_folder: The `dataset_folder` parameter is the path to the folder where the
        training images are located
        :param M: M is a parameter that determines the number of partitions to create. It is used in the
        function `TrainDataset.get__partition_id(*m, M, alpha)` to assign each image to a partition
        based on its UTM east, UTM north, and heading values
        :param alpha: The parameter `alpha` is a value used in determining the partition to which an
        image belongs. It is used in the calculation of the partition ID in the line
        `TrainDataset.get__partition_id(*m, M, alpha)`. The specific calculation of the partition ID is
        not shown in the code
        :param min_images_per_partition: The parameter "min_images_per_partition" is the minimum number
        of images required for a partition to be considered valid. If a partition has fewer images than
        this threshold, it will be excluded from the final set of partitions
        :param filename: The `filename` parameter is the name of the file where the
        `images_per_partition` dictionary will be saved using the `torch.save()` function. This file
        will store the mapping of partition IDs to the paths of images within each partition
        """
        logging.debug(f"Searching training images in {dataset_folder}")

        images_paths = read_images_paths(dataset_folder)
        logging.debug(f"Found {len(images_paths)} images")

        logging.debug(
            "For each image, get its UTM east, UTM north and heading from its path"
        )
        images_metadatas = [p.split("@") for p in images_paths]
        # field 1 is UTM east, field 2 is UTM north, field 9 is heading
        utmeast_utmnorth_heading = [(m[1], m[2], m[9]) for m in images_metadatas]
        utmeast_utmnorth_heading = np.array(utmeast_utmnorth_heading).astype(np.float64)

        logging.debug("For each image, get partition to which it belongs")
        partition_id = [
            SFXLDataset.get__partition_id(*m, M, alpha)
            for m in utmeast_utmnorth_heading
        ]

        logging.debug("Group together images belonging to the same partition")
        images_per_partition = defaultdict(list)
        for image_path, partition_id in zip(images_paths, partition_id):
            images_per_partition[partition_id].append(image_path)

        # Images_per_partition is a dict where the key is partition_id, and the value
        # is a list with the paths of images within that partition.
        images_per_partition = {
            k: v
            for k, v in images_per_partition.items()
            if len(v) >= min_images_per_partition
        }

        torch.save(images_per_partition, filename)

    @staticmethod
    def get__partition_id(utm_east, utm_north, heading, M, alpha) -> int:
        """Return partition_id and group_id for a given point.
        The partition_id is a triplet (tuple) of UTM_east, UTM_north and
        heading (e.g. (396520, 4983800,120)).
        The group_id represents the group to which the partition belongs
        (e.g. (0, 1, 0)), and it is between (0, 0, 0) and (N, N, L).
        """
        rounded_utm_east = int(
            utm_east // M * M
        )  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)
        rounded_heading = int(heading // alpha * alpha)

        partition_id = (rounded_utm_east, rounded_utm_north, rounded_heading)
        return partition_id
