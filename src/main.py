import argparse
import os
import eigen
import json
try:
    import cupy as np
except ImportError:
    import numpy as np


def _dir_path(dir_path):
    if os.path.isdir(dir_path):
        return dir_path
    else:
        raise NotADirectoryError(dir_path)


def load_training_data(filename):
    _eigenvector = np.load(filename + "_eigenvector.npy")
    _eigenface = np.load(filename + "_eigenface.npy")
    _mean = np.load(filename + "_mean.npy")
    # _mean = _mean.reshape(256 * 256, 1)
    _weights = np.load(filename + "_weights.npy")
    return _eigenvector, _eigenface, _mean, _weights


def save_training_data(filename, m, egf, egv, w):
    np.save(filename + "_mean", m)
    np.save(filename + "_eigenface", egf)
    np.save(filename + "_eigenvector", egv)
    np.save(filename + "_weights", w)


def get_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to image folder", required=True, type=_dir_path)
    args = parser.parse_args()
    return args.path


def generate_dataset(image_dir, filename):
    images = eigen.load_images(image_dir)
    total_images = images.shape[0]
    _mean, normalized = eigen.mean_images(images)

    # langkah 4
    cov_images = eigen.get_cov(normalized)

    # langkah 5
    _, _eigenvector = eigen.eig(cov_images, normalized)
    print(f"Estimated eigenvector file size: {_eigenvector.nbytes / 1024 / 1024} MB")

    # langkah 6
    k = total_images // 3
    _eigenface = np.array([], dtype=np.float32)
    for i, vector in zip(range(k), _eigenvector.T):
        face = normalized.T.dot(vector)
        face /= eigen.norm(face)
        _eigenface = np.append(_eigenface, face)
    _eigenface = _eigenface.reshape((k, 65536))
    print(f"Estimated eigenface file size: {_eigenface.nbytes / 1024 / 1024} MB")
    _weights = normalized @ _eigenface.T
    print(f"Estimated weight file size: {_weights.nbytes / 1024 / 1024} MB")
    save_training_data(filename, _mean, _eigenface, _eigenvector, _weights)


if __name__ == "__main__":
    data_filename = r"data\training_data"
    path = get_path()
    if not (os.path.isfile(data_filename + "_eigenface.npy")
            and os.path.isfile(data_filename + "_mean.npy")
            and os.path.isfile(data_filename + "_eigenvector.npy")
            and os.path.isfile(data_filename + "_weights.npy")):
        print("Dataset not found! Generating new file...")
        generate_dataset(path, data_filename)
        print("Dataset successfully generated.")
    else:
        print("Dataset found! Loading dataset...")

    eigenvector, eigenface, mean, weights = load_training_data(data_filename)
    print("Loading complete!")
    files = json.load(open("data/images.json", "r"))
    while True:
        n = input("Enter the filename (type 'exit' to exit): ")
        if n.lower() == "exit":
            break
        if not n.lower().endswith(".jpg"):
            n += ".jpg"
        print("Please wait")
        try:
            image = eigen.process_image(n, mean)
        except Exception:
            print("File not found!")
            continue
        test_weights = image.T @ eigenface.T
        idx, ref = eigen.euclidean_distance(weights, test_weights)
        print(idx, ref)
        print(files[idx])
        print("Done!")
