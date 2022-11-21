import argparse
import os
import eigen
import matplotlib.pyplot as plt
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


def save_training_data(filename, m, egf, egv):
    np.save(filename + "_mean", m)
    np.save(filename + "_eigenface", egf)
    np.save(filename + "_eigenvector", egv)


def get_path():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to image folder", required=True, type=_dir_path)
    args = parser.parse_args()
    return args.path


def generate_dataset(image_dir, filename):
    images = eigen.load_images(image_dir)
    _mean, normalized = eigen.mean_images(images)

    # langkah 4
    cov_images = eigen.get_cov(normalized.T)

    # langkah 5
    _, _eigenvector = eigen.eigen(cov_images)

    # langkah 6
    _eigenface = _eigenvector.dot(normalized.T)
    print(f"Estimated eigenface file size: {_eigenface.nbytes / 1024 / 1024} MB")
    print(f"Estimated eigenvector file size: {_eigenvector.nbytes / 1024 / 1024} MB")
    save_training_data(filename, _mean, _eigenface, _eigenvector)


if __name__ == "__main__":
    data_filename = r"data\training_data"
    path = get_path()
    if not (os.path.isfile(data_filename + "_eigenface.npy")
            and os.path.isfile(data_filename + "_mean.npy")
            and os.path.isfile(data_filename + "_eigenvector.npy")):
        print("Dataset not found! Generating new file...")
        generate_dataset(path, data_filename)
        print("Dataset successfully generated.")
    else:
        print("Dataset found! Loading dataset...")

    eigenvector = np.load(data_filename + "_eigenvector.npy")
    eigenface = np.load(data_filename + "_eigenface.npy")
    mean = np.load(data_filename + "_mean.npy")
    mean = mean.reshape(256 * 256, 1)
    print("Loading complete!")
    files = json.load(open("data/images.json", "r"))
    while True:
        n = input("Enter the filename (type 'exit' to exit): ")
        if n.lower() == "exit":
            break
        if not n.lower().endswith(".jpg"):
            n += ".jpg"
        print("Please wait")
        # filepath = fr"..\test\{n}"
        eigenface_testing = eigen.process_image(n, mean)
        # weight = eigenface_testing.T.dot(eigenface.T)
        # weight = np.argmin(eigen.norm(weight))
        idx, _ = eigen.euclidean_distance(eigenface, eigenface_testing)
        print(idx)
        print(f"Result: {files[idx]}")
        plt.imshow(eigenface_testing.get().reshape(256, 256), cmap="gray")
        plt.show()
        print("Done!")
