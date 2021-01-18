import glob

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    """MNIST Dataset

    Parameters
    ----------
    data_dir: str
        Directory in which the binary file is stored
    """
    def __init__(self, path: str):
        imgs = glob.glob(path)
        data = np.array([np.array(Image.open(img)) for img in imgs]) / 255.
        self.mnist = torch.from_numpy(data).unsqueeze(1).float()

    def __len__(self):
        return self.mnist.size()[0]

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.mnist[idx]


def test_mnist_dataset():
    # NOTE This test requires mnist binary data in ./data dir
    mnist = MnistDataset("./data/trainingSet/*.jpg")

    assert len(mnist) == 42_000, f"MNIST has {len(mnist)} records, not 42,000"
    assert mnist[0].size() == torch.Size(
        [1, 28, 28]), f"Shape of MNIST image: {mnist[0].size()}"
    assert mnist[0].max() == 1.
