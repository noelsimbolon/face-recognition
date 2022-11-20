import cv2
import numpy as np
import os
from math import sqrt


def get_covariance_dir(filepath) -> np.array:
    files = []
    for folder in os.listdir(filepath):
        folder_path = os.path.join(filepath, folder)
        for idx, img in zip(range(10), os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img)
            files.append(img_path)
    cov = np.array([get_covariance_image(i) for i in files])
    return cov


def get_image_mean(file):
    # langkah 1
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = np.array(img, dtype=np.float64)

    # langkah 2
    mean = np.mean(img)

    # langkah 3
    img -= mean
    return img


def get_covariance_image(file):
    # langkah 4
    return np.cov(get_image_mean(file))


def norm(matrix):
    return sqrt(sum([i ** 2 for i in matrix]))


def qr_decom(matrix):
    k = np.array(matrix, dtype=np.float64)
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

    return -Q, -R


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


def eface(eigenvector, avrmatrix):
    eigenface = list()
    n = avrmatrix.shape[0]
    for i in range(n):
        eigenface.append(np.matmul(eigenvector, avrmatrix[0]))

    return eigenface


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
