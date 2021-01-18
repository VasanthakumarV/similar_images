from typing import List

from numpy import ndarray
from sklearn.neighbors import NearestNeighbors


def nearest_neighbors(n: int, image_encoding: ndarray,
                      embedding: ndarray) -> List[int]:
    """Returns `n` indices which are the closest

    Parameters
    ----------
    n: int
        Number of neighers to return
    image_encoding: ndarray
        A single image's encoding for which we are finding
        similar ones
    embedding: ndarray
        Encoding of all the examples

    Returns
    -------
    List[int]
        List of indices representing the training examples
        which are the closest
    """
    knn = NearestNeighbors(n_neighbors=n, metric="cosine")

    # Fitting the embedding of the training examples
    knn.fit(embedding)

    # Finding `n` neighbors closest to our input image encoding
    _, indices = knn.kneighbors(image_encoding)

    return list(indices[0])
