import argparse
import os
import eigen
import matplotlib.pyplot as plt
import json
try:
    import cupy as np
except ImportError:
    import numpy as np


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled = int(length * iteration // total)
    bar = fill * filled + '-' * (length - filled)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total:
        print()


def _dir_path(dir_path):
    if os.path.isdir(dir_path):
        return dir_path
    else:
        raise NotADirectoryError(dir_path)


def read_training_data(filename):
    with open(filename, 'r') as file:
        return tuple(json.load(file))


def save_training_data(filename, input_data):
    images, mean = input_data
    data = eigen.eigen(images)
    to_save = data[0].tolist(), data[1].tolist()
    json.dump(to_save, open(filename + ".json", 'w+'))
    json.dump(mean.tolist(), open(filename + "_mean.json", 'w+'))


def load_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="path to image folder", required=True, type=_dir_path)
    args = parser.parse_args()
    return args.path


def generate_dataset(image_dir: str, filename):
    images, mean = eigen.load_images(image_dir)
    print(images.shape)
    print(f"Estimated file size: {images.nbytes / 1024 / 1024} MB")
    save_training_data(filename, (images, mean))


if __name__ == "__main__":
    data_filename = r"data\training_data"
    path = load_main()
    if not (os.path.isfile(data_filename + ".json") and os.path.isfile(data_filename + "_mean.json")):
        print("Dataset not found! Generating new file...")
        generate_dataset(path, data_filename)
        print("Dataset successfully generated.")
    else:
        print("Dataset found! Loading dataset...")

    eigenvalue, eigenvector = read_training_data(data_filename + ".json")
    eigenvalue = np.array(eigenvalue, dtype=np.float64)
    eigenvector = np.array(eigenvector, dtype=np.float64)
    print("Loading complete!")
    mean = np.array(json.load(open(data_filename + "_mean.json")), dtype=np.float64)
    # files = json.load(open("data/images.json", "r"))[:16]
    # fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(10, 10))
    # example = np.array(eigen.eface(eigenvector, mean))
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(np.asnumpy(example[i]))
    #     ax.set(xticks=[], yticks=[])
    # plt.imshow(eigen.eface(eigenvector, eigen.load_image("saul.jpg")), cmap="gray")
    while True:
        n = input("Enter the filename (type 'exit' to exit): ")
        if n.lower() == "exit":
            break
        if n.lower()[:-4] != ".jpg":
            n += ".jpg"
        print("Please wait")
        image = eigen.load_image(n)
        vectors = image.flatten()
        normalized = vectors - mean
        normalized = normalized.reshape((256, 256))
        # np.resize(eigenvector, (256, 256))
        # weights = np.dot(normalized.reshape((256, 256)).T, eigenvector)
        plt.imshow(eigen.eigenface((eigenvalue, eigenvector), mean.reshape((256, 256))).get(), cmap="jet")
        # plt.imshow(normalized.get(), cmap="gray")
        plt.show()
        print("Done!")
