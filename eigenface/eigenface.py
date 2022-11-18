import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor


def get_covariance(filepath):
    files = []
    for folder in os.listdir(filepath):
        folder_path = os.path.join(filepath, folder)
        for idx, img in zip(range(10), os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img)
            files.append(img_path)
    # langkah 1
    # with Pool(processes=os.cpu_count()) as pool:
    #     img = pool.map(cv2.imread, files, chunksize=32)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        img = executor.map(cv2.imread, files, chunksize=32)
    # img = [cv2.imread(file) for file in files]
    img = [cv2.resize(i, (256, 256)) for i in img]
    img = [i.flatten() for i in img]
    img = np.array(img, dtype=np.float32)

    # langkah 2
    mean = np.mean(img)
    # langkah 3
    img = img - mean
    # langkah 4
    cov = np.cov(img)
    return cov
