import os
import glob

from numpy import ndarray
from torch import Tensor
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """MNIST Dataset

    Parameters
    ----------
    path: str
        Directory in which the images are stored
    """
    def __init__(self, path: str):
        # List of jpeg images in the folder
        self.imgs = glob.glob(os.path.join(path, "*.jpg"))

    def __len__(self):
        return len(self.imgs)

    def get_image(self, idx: int) -> ndarray:
        return np.array(Image.open(self.imgs[idx]))

    def __getitem__(self, idx: int) -> Tensor:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = np.array(Image.open(self.imgs[idx])) / 255.
        return torch.from_numpy(data).permute(2, 0, 1).float()


def test_mnist_dataset():
    # NOTE We assume the images are download and available
    mnist = ImageDataset("./data/dataset")

    # Making sure we have all the images
    assert len(mnist) == 42_000, f"MNIST has {len(mnist)} records, not 42,000"

    # Making sure we have the channel dimension and the height and width
    # checks out
    assert mnist[0].size() == torch.Size(
        [1, 28, 28]), f"Shape of MNIST image: {mnist[0].size()}"

    # Making sure the data is normalized
    assert mnist[0].max() == 1.
