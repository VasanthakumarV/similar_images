from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Convolutional Encoder

    Parameters
    ----------
    cin: int
        Input channel size
    cout: int
        Output channel size
    """
    def __init__(self, cin: int, cout: int):
        super().__init__()

        self.conv1 = nn.Conv2d(cin, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, cout, 3, padding=1)

    def forward(self, x):
        x = F.max_pool2d(F.selu(self.conv1(x)), 2)
        x = F.max_pool2d(F.selu(self.conv2(x)), 2)
        x = F.max_pool2d(F.selu(self.conv3(x)), 2)
        x = F.max_pool2d(F.selu(self.conv4(x)), 2)
        x = F.max_pool2d(F.selu(self.conv5(x)), 2)

        return x


class Decoder(nn.Module):
    """Convolutional Decoder

    Parameters
    ----------
    cin: int
        Input channel size
    cout: int
        Output channel size
    """
    def __init__(self, cin: int, cout: int):
        super().__init__()

        self.conv_transpose1 = nn.ConvTranspose2d(cin, 128, 2, stride=2)
        self.conv_transpose2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_transpose3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_transpose4 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_transpose5 = nn.ConvTranspose2d(16, cout, 2, stride=2)

    def forward(self, x):
        x = F.selu(self.conv_transpose1(x))
        x = F.selu(self.conv_transpose2(x))
        x = F.selu(self.conv_transpose3(x))
        x = F.selu(self.conv_transpose4(x))
        x = F.selu(self.conv_transpose5(x))

        return x


class SimilarityModel(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()

        self.enc = Encoder(cin, cout)
        self.dec = Decoder(cout, cin)

    def encoder(self, x) -> Tensor:
        """
        Returns
        -------
        Tensor
            Represents the encoder output
        """
        return self.enc(x)

    def forward(self, x) -> Tensor:
        """
        Returns
        -------
        Tensor
            Represents the decoder output
        """
        latent = self.enc(x)
        return self.dec(latent)


class TestModel:
    def test_encoder(self):
        encoder = Encoder(1, 256)
        output = encoder(torch.rand(10, 1, 28, 28))
        assert output.size() == torch.Size(
            [10, 1, 28, 28]), f"Encoder output shape: {output.size()}"

    def test_decoder(self):
        decoder = Decoder(256, 1)
        output = decoder(torch.rand(10, 32, 6, 6))
        assert output.size() == torch.Size(
            [10, 32, 6, 6]), f"Decoder output shape: {output.size()}"

    def test_similarity_model(self):
        model = SimilarityModel(1, 256)
        output = model(torch.rand(10, 1, 28, 28))
        assert output.size() == torch.Size(
            [10, 32, 6, 6]), f"SimilarityModel output shape: {output.size()}"

        output = model.encoder(torch.rand(10, 1, 28, 28))
        assert output.size() == torch.Size(
            [10, 32, 6, 6]), f"Encoder output shape: {output.size()}"
