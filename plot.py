import math

from numpy import ndarray
import matplotlib.pyplot as plt


def plot_grid(base_image: ndarray, images: ndarray):
    """Plots the base image with its neighbors in a single plot
    Parameters
    ----------
    base_image: ndarray
        This image gets plotted on the first row
    images: ndarray
        These are images that deemed to be similar by the neural net,
        they get plotted from the second row, three per row till exhaustion
    """
    # The number of columns is fixed
    ncols = 3

    # Rows are calculated based on `ncols`
    # We add one to account for the `base_image`
    nrows = math.ceil(images.shape[0] / ncols) + 1

    _, axes = plt.subplots(nrows, ncols)

    # We plot the `base_image` in one cell
    # and keep the other two empty
    axes[0][0].imshow(base_image.squeeze())
    axes[0][1].axis("off")
    axes[0][2].axis("off")

    for i, ax in enumerate(axes.flatten()[3:]):
        if i >= images.shape[0]:
            # Unfilled cells in the last row are made empty
            # by switchin off their axes
            ax.axis("off")
            continue

        # We squeeze the array before plotting
        # to account for grayscale images (which do not have channels)
        ax.imshow(images[i].squeeze())

    # We avoid overlaps
    plt.tight_layout()

    plt.show()
