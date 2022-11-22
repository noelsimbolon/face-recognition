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
    return images


def load_image(file):
    # langkah 1
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img.flatten()
    return img


def process_image(file, mean):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = cv2.resize(img, (256, 256))
    img = img.reshape(256, 256)
    img = np.asarray(img)
    mean = mean.reshape(256, 256)
    return (img - mean).reshape(256*256,)


def mean_images(imgs):
    # langkah 2
    mean = imgs.mean(axis=0)
    return mean, imgs - mean  # langkah 3


def get_cov(imgs):
    # langkah 4
    return imgs @ imgs.T


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


def eig(matrix, normalized, iteration=100):
    # m = np.asarray(matrix)
    n = matrix.shape[0]
    eigenvector = np.identity(n)

    # pake qr sendiri
    """ for i in range(iteration):
        Q, R = qr_decom(matrix)
        eigenvector = np.matmul(eigenvector,Q)
        matrix = np.matmul(R,Q) """
    # pake qr numpy
    for i in range(iteration):
        Q, R = np.linalg.qr(matrix)
        matrix = np.matmul(R, Q)
        eigenvector = np.matmul(eigenvector, Q)

    eigenvalue = np.diagonal(matrix)

    # idxsort = eigenvalue.argsort()[::-1]
    # eigenvalue = eigenvalue[idxsort]
    # eigenvector = eigenvector[:, idxsort]
    # normal = normalized[idxsort, :]

    # eigenvalue = np.around(eigenvalue, decimals=3)
    # eigenvector = np.around(eigenvector, decimals=3)

    return eigenvalue, eigenvector#, normal


def linear_combination(eigenface, normalized):
    coeff_vector = np.array([])

    for normal in normalized:
        normal = normal.reshape((256, 256))
        for face in eigenface:
            face = face.reshape((256, 256))
            res = np.linalg.solve(face.T, normal)
            coeff_vector = np.append(coeff_vector, res)

    return coeff_vector


def euclidean_distance(weights, test_weights):
    small = 1_000_000_000
    idx_small = 0
    dist = 0

    for idx, weight in enumerate(weights):
        dist = norm(weight - test_weights)

        if dist < small:
            small = dist
            idx_small = idx
    return idx_small, dist


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

    result = eig(inp, 100000)  # iterationnya bebas diganti berapa
    """ np.savetxt('eigval.txt',result[0])
    np.savetxt('eigvec.txt',result[1]) """
    print(result[0])
    print(result[1])