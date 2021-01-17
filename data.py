import os
import struct
import array
import functools
import operator
from typing import IO

import numpy
import torch
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    """MNIST Dataset

    Parameters
    ----------
    data_dir: str
        Directory in which the binary file is stored
    """
    def __init__(self, data_dir: str):
        # Parsing and reading the binary file as torch tensor
        path = os.path.join(data_dir, "train-images-idx3-ubyte")
        with open(path, "rb") as f:
            mnist = parse_idx(f) / 255.
            self.mnist = torch.from_numpy(mnist).unsqueeze(1).float()

    def __len__(self):
        return self.mnist.size()[0]

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.mnist[idx]


def test_mnist_dataset():
    # NOTE This test requires mnist binary data in ./data dir
    mnist = MnistDataset("./data")

    assert len(mnist) == 60_000, f"MNIST has {len(mnist)} records, not 60,000"
    assert mnist[0].size() == torch.Size(
        [1, 28, 28]), f"Shape of MNIST image: {mnist[0].size()}"
    assert mnist[0].max() == 1.


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed"""
    pass


def parse_idx(fd: IO):
    """Parse an IDX file, and return it as a numpy array.

    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse
    endian : str
        Byte order of the IDX file. See [1] for available options

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    1. https://docs.python.org/3/library/struct.html
        #byte-order-size-and-alignment
    """
    DATA_TYPES = {
        0x08: 'B',  # unsigned byte
        0x09: 'b',  # signed byte
        0x0b: 'h',  # short (2 bytes)
        0x0c: 'i',  # int (4 bytes)
        0x0d: 'f',  # float (4 bytes)
        0x0e: 'd'
    }  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, '
                             'file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, '
                             'file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type '
                             '0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' %
                             (expected_items, len(data)))

    return numpy.array(data).reshape(dimension_sizes)
