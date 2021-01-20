from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Convolutional Encoder"""
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.max_pool2d(F.selu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.selu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.selu(self.bn3(self.conv3(x))), 2)

        return x


class Decoder(nn.Module):
    """Convolutional Decoder"""
    def __init__(self):
        super().__init__()

        self.conv_transpose3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_transpose4 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.conv_transpose5 = nn.ConvTranspose2d(8, 3, 2, stride=2)

    def forward(self, x):
        x = F.selu(self.conv_transpose3(x))
        x = F.selu(self.conv_transpose4(x))
        x = F.selu(self.conv_transpose5(x))

        return x


class SimilarityModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = Encoder()
        self.dec = Decoder()

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
        encoder = Encoder()
        output = encoder(torch.rand(10, 3, 256, 256))
        assert output.size() == torch.Size(
            [10, 256, 16, 16]), f"Encoder output shape: {output.size()}"

    def test_decoder(self):
        decoder = Decoder()
        output = decoder(torch.rand(10, 64, 64, 64))
        assert output.size() == torch.Size(
            [10, 3, 512, 512]), f"Decoder output shape: {output.size()}"

    def test_similarity_model(self):
        model = SimilarityModel()
        output = model(torch.rand(10, 3, 256, 256))
        assert output.size() == torch.Size(
            [10, 3, 512,
             512]), f"SimilarityModel output shape: {output.size()}"

        output = model.encoder(torch.rand(10, 3, 256, 256))
        assert output.size() == torch.Size(
            [10, 256, 16, 16]), f"Encoder output shape: {output.size()}"
