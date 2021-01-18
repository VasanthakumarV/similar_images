import os
import glob

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    """MNIST Dataset

    Parameters
    ----------
    path: str
        Directory in which the images are stored
    """
    def __init__(self, path: str):
        # List of jpeg images in the folder
        imgs = glob.glob(os.path.join(path, "*.jpg"))

        # Reading-in the jpeg files as numpy array
        # We also normalize them by dividing by 255
        data = np.array([np.array(Image.open(img)) for img in imgs]) / 255.

        # Converting numpy array to torch tensors
        # NOTE since the images are greyscale, we add the channel dimension
        self.mnist = torch.from_numpy(data).unsqueeze(1).float()

    def __len__(self):
        return self.mnist.size()[0]

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.mnist[idx]


def test_mnist_dataset():
    # NOTE We assume the images are download and available
    mnist = MnistDataset("./data/trainingSet")

    # Making sure we have all the images
    assert len(mnist) == 42_000, f"MNIST has {len(mnist)} records, not 42,000"

    # Making sure we have the channel dimension and the height and width
    # checks out
    assert mnist[0].size() == torch.Size(
        [1, 28, 28]), f"Shape of MNIST image: {mnist[0].size()}"

    # Making sure the data is normalized
    assert mnist[0].max() == 1.
