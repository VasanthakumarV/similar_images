import math

from numpy import ndarray
import matplotlib.pyplot as plt


def plot_grid(base_image: ndarray, images: ndarray):
    ncols = 3
    nrows = math.ceil(images.shape[0] / ncols) + 1
    _, axes = plt.subplots(nrows, ncols)

    axes[0][0].imshow(base_image.squeeze())
    axes[0][1].axis("off")
    axes[0][2].axis("off")

    for i, ax in enumerate(axes.flatten()[3:]):
        if i >= images.shape[0]:
            ax.axis("off")
            continue

        ax.imshow(images[i].squeeze())

    plt.tight_layout()
    plt.show()
