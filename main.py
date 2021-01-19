import os
import glob
import argparse

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader
import matplotlib.pyplot as plt
from google_drive_downloader import GoogleDriveDownloader as gdd

from data import ImageDataset
from model import SimilarityModel
from similarity import nearest_neighbors
from plot import plot_grid

# These variables are used only when zip folders are
# downloaded from google drive
TRAIN_DIR = "./data/dataset.zip"
TEST_DIR = "./data/test.zip"

# Names of the folders where extracted jpeg images are stored
TRAIN_DATA = "./data/dataset"
TEST_DATA = "./data/test"

EPOCHS = 50
BATCH_SIZE = 32
TRAIN_SIZE = 0.9
LEARNING_RATE = 1e-3

CHANNELS_IN = 3
CHANNELS_OUT = 256

EMBEDDING = "embedding_epoch_%d.npy"
MODEL = "model_epoch_%d.pt"

EMBEDDING_FINAL = "embedding_epoch_95.npy"
MODEL_FINAL = "model_epoch_95.pt"


def train(model, data_loader, optimizer, loss_fn, device):
    """Helper function to train the model

    Parameters
    ----------
    model: Module
        `model` being trained
    data_loader: DataLoader
        DataLoader object for feeding data
    optimizer: Optimizer
        Optimizer used for optmizing the weights
    loss_fn: Module
        Module for calculating the loss
    devie: str
        'cpu' or 'cuda' to use

    Returns
    -------
    float
        Loss value
    """
    # We switch to training mode
    model.train()

    loss_total = 0

    for data in data_loader:
        data = data.to(device)

        output = model(data)

        loss = loss_fn(output, data)

        # Updating the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    return loss_total / len(data_loader)


def validate(model, data_loader, loss_fn, device):
    """Helper function to validate the model

    Parameters
    ----------
    model: Module
        `model` being evaluated
    data_loader: DataLoader
    loss_fn: Module
    device: str

    Returns
    -------
    float
        Loss value
    """
    # We turn on evaluation mode
    model.eval()

    loss_total = 0

    # We turn off tracking gradients
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            output = model(data)

            loss_total += loss_fn(output, data).item()

    return loss_total / len(data_loader)


def create_embedding(model, data_loader, device):
    """Embeddings are created for all training examples"""
    model.eval()

    # List for capturing encodings
    embedding = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            # Running only the encoder
            enc_output = model.encoder(data)

            # Flattening and appending the output
            embedding.append(enc_output.view(data.size()[0], -1))

    return torch.cat(embedding, dim=0)


def main(dataset, device):
    model = SimilarityModel(cin=CHANNELS_IN, cout=CHANNELS_OUT).to(device)
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

        if epoch % 5 == 0:
            val_loss = validate(model, val_loader, mse_loss, device)
            print(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}")

            print("Saving model and embedding")

            embedding = create_embedding(model, full_loader, device)
            embedding = embedding.cpu().detach().numpy()

            np.save(EMBEDDING % epoch, embedding)
            torch.save(model.state_dict(), MODEL % epoch)


def test(dataset, test_data, device):
    imgs = glob.glob(os.path.join(test_data, "*.jpg"))
    nimgs = len(imgs)

    data_npy = np.array([np.array(Image.open(img)) for img in imgs])
    data = torch.from_numpy(data_npy).permute(0, 3, 1, 2).float()

    embedding = np.load(EMBEDDING_FINAL)
    model = SimilarityModel(cin=CHANNELS_IN, cout=CHANNELS_OUT).to(device)
    model.load_state_dict(torch.load(MODEL_FINAL, map_location=device))
    model.eval()

    with torch.no_grad():
        encoding = model.encoder(data.to(device)).reshape(
            nimgs, -1).cpu().detach().numpy()

    for i in range(nimgs):
        indices = nearest_neighbors(
            9,
            encoding[i].reshape(1, -1),
            embedding,
        )

        neighbors = np.stack([dataset.get_image(i) for i in indices])

        plot_grid(data_npy[i], neighbors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["train", "test"],
                        default="train",
                        type=str,
                        help="Train or Test the model")
    parser.add_argument("--drive-train-id",
                        type=str,
                        help="Google drive id for training data")
    parser.add_argument("--drive-test-id",
                        type=str,
                        help="Google drive id for testing data")
    parser.add_argument("--train-data",
                        default=TRAIN_DATA,
                        type=str,
                        help="Location of training data")
    parser.add_argument("--test-data",
                        default=TEST_DATA,
                        type=str,
                        help="Location of testing data")
    args = parser.parse_args()

    if args.drive_train_id is not None:
        gdd.download_file_from_google_drive(
            file_id=args.drive_train_id,
            dest_path=TRAIN_DIR,
            unzip=True,
        )
    if args.drive_test_id is not None:
        gdd.download_file_from_google_drive(
            file_id=args.drive_test_id,
            dest_path=TEST_DIR,
            unzip=True,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = ImageDataset(args.train_data)

    if args.mode == "train":
        main(dataset, device)
    else:
        test(dataset, args.test_data, device)
