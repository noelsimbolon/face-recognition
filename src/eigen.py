import cv2
import os
import json
try:
    import cupy as np
except ImportError:
    import numpy as np


def load_images(filepath):
    files = []
    for folder in os.listdir(filepath):
        folder_path = os.path.join(filepath, folder)
        for idx, img in zip(range(3), os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img)
            files.append(img_path)
    with open("data/images.json", "w+") as f:
        json.dump(files, f)
    images = np.array([load_image(i) for i in files])
    return images.transpose()


def load_image(file):
    # langkah 1
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img.reshape(256 * 256,)
    return img


def process_image(file, mean):
    img = load_image(file)
    img = img.reshape(256 * 256, 1)
    img = np.asarray(img)
    return img - mean


def mean_images(imgs):
    # langkah 2
    mean = imgs.mean(axis=1)
    mean = mean.reshape(imgs.shape[0], 1)
    return mean, imgs - mean  # langkah 3


def get_cov(imgs):
    # langkah 4
    cov = (imgs).dot(np.transpose(imgs))
    return cov


def norm(matrix):
    return np.sqrt(np.sum(np.square(matrix)))


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

def linear_combination(eigenface_training, normalized_matrix):
    m, n = normalized_matrix.shape
    coeff_matrix = np.empty((m,0))

    for i in range(n):
        result = np.linalg.solve(np.transpose(eigenface_training),normalized_matrix[:,i])
        coeff_matrix = np.append(coeff_matrix, np.transpose(result), axis=1)

    return coeff_matrix

def euclidean_distance(eigenface_training, eigenface_testing):
    n = eigenface_training.shape[0]

    smallest_distance = 99999
    idx_smallest = 0

    for i in range(n):
        distance = norm(eigenface_training[:, i] - eigenface_testing[i, :])

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