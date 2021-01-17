import os
import glob

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt

from data import MnistDataset
from model import SimilarityModel
from similarity import nearest_neighbors
from plot import plot_grid

TRAIN_DATA = "./data"
TEST_DATA = "./data/test"

EPOCHS = 2
BATCH_SIZE = 32
TRAIN_SIZE = 0.8
LEARNING_RATE = 1e-3

CHANNELS_IN = 1
CHANNELS_OUT = 32

EMBEDDING = "embedding.npy"
MODEL = "similarity_model.pt"


def train(model, data_loader, optimizer, loss_fn, device):
    model.train()

    for data in data_loader:
        data = data.to(device)

        output = model(data)

        loss = loss_fn(output, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def validate(model, data_loader, loss_fn, device):
    model.eval()

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            output = model(data)

            loss = loss_fn(output, data)

    return loss.item()


def create_embedding(model, data_loader, device):
    model.eval()

    embedding = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            enc_output = model.encoder(data).cpu()

            embedding.append(enc_output.view(BATCH_SIZE, -1))

    return torch.cat(embedding, dim=0)


def main(device):
    model = SimilarityModel(cin=CHANNELS_IN, cout=CHANNELS_OUT).to(device)
    dataset = MnistDataset(TRAIN_DATA)
    mse_loss = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    train_size = int(len(dataset) * TRAIN_SIZE)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # NOTE `Shuffle` is set to `False`, the ordering must be preserved
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, mse_loss, device)
        print(f"Epoch: {epoch}, Training Loss: {train_loss:.4f}")

        val_loss = validate(model, val_loader, mse_loss, device)
        print(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}")

    embedding = create_embedding(model, full_loader, device)
    embedding = embedding.cpu().detach().numpy()

    np.save(EMBEDDING, embedding)
    torch.save(model.state_dict(), MODEL)


def test(device):
    dataset = MnistDataset(TRAIN_DATA)

    imgs = glob.glob(os.path.join(TEST_DATA, "*.jpg"))
    nimgs = len(imgs)

    data_npy = np.array([np.array(Image.open(img)) for img in imgs])
    data = torch.from_numpy(data_npy).unsqueeze(1).float()

    embedding = np.load(EMBEDDING)
    model = SimilarityModel(cin=CHANNELS_IN, cout=CHANNELS_OUT).to(device)
    model.load_state_dict(torch.load(MODEL, map_location=device))
    model.eval()

    with torch.no_grad():
        encoding = model.encoder(data.to(device)).view(
            nimgs, -1).cpu().detach().numpy()

    for i in range(nimgs):
        indices = nearest_neighbors(
            9,
            encoding[i].reshape(1, -1),
            embedding,
        )

        plot_grid(data_npy[i],
                  dataset[indices].permute(0, 2, 3, 1).cpu().detach().numpy())


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    main(device)
    test(device)
