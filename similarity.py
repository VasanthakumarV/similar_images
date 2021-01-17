from typing import List

from numpy import ndarray
from sklearn.neighbors import NearestNeighbors


def nearest_neighbors(n: int, embedding: ndarray,
                      image_encoding: ndarray) -> List[int]:
    knn = NearestNeighbors(n_neighbors=n, metric="cosine")

    knn.fit(image_encoding)

    _, indices = knn.kneighbors(embedding)

    return list(indices[0])
