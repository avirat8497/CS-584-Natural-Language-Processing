import numpy as np
def knn(vector, matrix, k=10):

    nearest_idx = []

    matNorm = np.linalg.norm(matrix, axis=1)
    vecNorm = np.linalg.norm(vector)
    dist = np.dot(matrix, vector) / (matNorm * vecNorm)


    order = dist.argsort()[::-1]
    nearest_idx = order[:k]

    return nearest_idx
