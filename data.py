import os
import glob

from PIL import Image
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


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

        # We sort the image files based on their name
        # This is to make sure the order does not change on different platforms
        # and it always stays in sync with the embedding file
        self.imgs.sort()

        # Applying transforms on image for better generalization
        self.transform = transforms.Compose([
            transforms.RandomApply(
                [
                    transforms.RandomHorizontalFlip(p=0.85),
                    transforms.RandomPerspective()
                ],
                p=0.5,
            )
        ])

    def __len__(self):
        return len(self.imgs)

    def get_image(self, idx: int) -> ndarray:
        return np.array(Image.open(self.imgs[idx]))

    def __getitem__(self, idx: int) -> Tensor:
        data = np.array(Image.open(self.imgs[idx])) / 255.
        data = torch.from_numpy(data).permute(2, 0, 1).float()
        return self.transform(data)


def test_mnist_dataset():
    # NOTE We assume the images are download and available
    mnist = ImageDataset("./data/dataset")

    # Making sure we have all the images
    assert len(mnist) == 4738, f"MNIST has {len(mnist)} records, not 4738"

    # Making sure we have the channel dimension and the height and width
    # checks out
    assert mnist[0].size() == torch.Size(
        [3, 512, 512]), f"Shape of MNIST image: {mnist[0].size()}"

    # Making sure the data is normalized
    assert mnist[0].max() == 1.
