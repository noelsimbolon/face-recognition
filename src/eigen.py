import cv2
import os
import json
from math import sqrt
try:
    import cupy as np
except ImportError:
    import numpy as np


def load_images(filepath, cov=True):
    files = []
    for folder in os.listdir(filepath):
        folder_path = os.path.join(filepath, folder)
        for idx, img in zip(range(3), os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img)
            files.append(img_path)
    with open("data/images.json", "w+") as f:
        json.dump(files, f)
    # cov = np.array([load_image(i) for i in files])
    images = np.array([load_image(i) for i in files])
    return get_cov(images) if cov else images, mean_images(images)


def load_image(file):
    # langkah 1
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float32)
    return img


def get_vectors(imgs):
    return np.array([i.flatten() for i in imgs])


def mean_images(imgs):
    # langkah 2
    vectors = get_vectors(imgs)
    mean = np.zeros((1, 256 * 256), dtype=np.float32)
    for i in vectors:
        mean = np.add(mean, i)
    mean = np.divide(mean, float(imgs.shape[0])).flatten()
    # mean = np.mean(vectors, axis=0)
    return mean


def normalize_tensors(imgs, tensors, mean):
    # langkah 3
    normalized = np.ndarray((imgs.shape[0], 256 * 256), dtype=np.float32)
    for i in range(imgs.shape[0]):
        normalized[i] = np.subtract(tensors[i], mean)
    return normalized


def get_cov(imgs):
    # langkah 4
    mean = mean_images(imgs)
    vectors = get_vectors(imgs)
    normalized = normalize_tensors(imgs, vectors, mean)
    cov = np.cov(normalized)
    return cov


def norm(matrix):
    return sqrt(sum([i ** 2 for i in matrix]))


def qr_decom(matrix):
    k = np.array(matrix, dtype=np.float32)
    c = np.copy(k)
    m, n = k.shape

    Q = np.zeros((m, n))

    for i in range(m):
        a = k[:, i]

        if i > 0:
            for j in range(i):
                prev = Q[:, j]
                temp = k[:, i]
                a -= (np.dot(temp, prev)) * prev

        Q[:, i] = a / np.sqrt(np.dot(a, a))

    R = np.matmul(np.transpose(Q), c)

    return -np.matmul(Q, np.sign(np.diagonal(np.sign(Q)))), -np.matmul(np.sign(np.diagonal(np.sign(Q))), R)


def eigen(matrix, iteration=100):
    m = np.array(matrix)
    n = m.shape[0]
    eigenvector = np.identity(n)

    # pake qr sendiri
    """ for i in range(iteration):
        Q, R = qr_decom(matrix)
        eigenvector = np.matmul(eigenvector,Q)
        matrix = np.matmul(R,Q) """

    # pake qr numpy
    for i in range(iteration):
        m = np.copy(matrix)
        Q, R = np.linalg.qr(m)
        matrix = np.matmul(R, Q)
        eigenvector = np.matmul(eigenvector, Q)

    eigenvalue = np.diagonal(matrix)

    idxsort = eigenvalue.argsort()[::-1]
    eigenvalue = eigenvalue[idxsort]
    eigenvector = eigenvector[:, idxsort]

    eigenvalue = np.around(eigenvalue, decimals=3)
    eigenvector = np.around(eigenvector, decimals=3)

    return eigenvalue, eigenvector


def eigenface(eig, mean_matrix):
    eigenvalue, eigenvector = eig
    n, _ = mean_matrix.shape

    eig_pairs = [(eigenvalue[i], eigenvector[:n, i]) for i in range(n)]
    eig_pairs.sort(reverse=True)
    eigvalues_sort = [eig_pairs[i][0] for i in range(n)]
    eigvectors_sort = [eig_pairs[i][1] for i in range(n)]

    reduced_eigenvector = np.array(eigvectors_sort[:n]).transpose()

    vectors = (mean_matrix.reshape((256, 256)) * 256.).astype(np.float32)
    proj_data = np.dot(vectors.transpose(), reduced_eigenvector)
    print(proj_data)
    return proj_data
    # eigenface = np.zeros((n, n), dtype=np.float32)
    # for i in range(n):
    #     eigenface[:, i] = eigenvector[:, i] + mean_matrix.flatten()
    # return eigenface

def euclidean_distance(eigenface_training, eigenface_testing):
    n = eigenface_testing.shape[0]

    smallest_distance = 999
    idx_smallest = 0

    for i in range(n):
        distance = np.sqrt(np.sum(np.square(eigenface_training - eigenface_testing[i])))

        if distance < smallest_distance:
            smallest_distance = distance
            idx_smallest = i

    return idx_smallest, smallest_distance


if __name__ == "__main__":
    # buat testing
    A = np.random.rand(20, 20)

    inp = np.loadtxt('matrix.txt')

    e = np.linalg.eigh(inp)
    eval = e[0]
    evec = e[1]

    eval = np.around(eval, decimals=3)
    evec = np.around(evec, decimals=3)

    print(eval)
    print(evec)

    result = eigen(inp, 100000)  # iterationnya bebas diganti berapa
    """ np.savetxt('eigval.txt',result[0])
    np.savetxt('eigvec.txt',result[1]) """
    print(result[0])
    print(result[1])