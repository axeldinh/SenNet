import os
import cv2
import numpy as np
from tqdm import tqdm

data_path = "C:/Users/Axeld/Desktop/SenNet/blood-vessel-segmentation/train/"


def make_mesh(labels_folder, min_area, mesh_name, verbose=True):
    files = os.listdir(labels_folder)
    files = [file for file in files if file.endswith(".tif")]

    f = open(mesh_name, "w")

    if verbose:
        files = tqdm(files)

    for file in files:
        z = np.ndarray([int(file.split(".")[0])])

        mask = cv2.imread(os.path.join(labels_folder, file), cv2.IMREAD_GRAYSCALE)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        contours = [
            contour for contour in contours if cv2.contourArea(contour) > min_area
        ]

        if len(contours) == 0:
            continue

        contours = np.concatenate(contours, axis=0)

        xs = contours[:, 0, 0]
        ys = contours[:, 0, 1]
        zs = np.ones_like(xs) * z

        for x, y, z in zip(xs, ys, zs):
            f.write(f"v {x} {y} {z}\n")

    f.close()


if __name__ == "__main__":
    import cc3d

    labels_folder = os.path.join(data_path, "kidney_1_dense/labels/")
    mesh_name = "test.obj"
    verbose = True
    min_area = 0
    min_volume = 100_000 / 8

    files = os.listdir(labels_folder)
    files = [file for file in files if file.endswith(".tif")]
    files = sorted(files, key=lambda x: int(x.split(".")[0]))
    files = files[::2]

    mask = cv2.imread(os.path.join(labels_folder, files[0]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (mask.shape[0] // 2, mask.shape[1] // 2))
    volume = np.zeros([mask.shape[0], mask.shape[1], len(files)], dtype=np.uint8)

    pb = range(len(files))
    if verbose:
        pb = tqdm(pb)

    for i in pb:
        mask = cv2.imread(os.path.join(labels_folder, files[i]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (mask.shape[0] // 2, mask.shape[1] // 2))

        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

        # Remove contours with area smaller than min_area
        for c in contours:
            if cv2.contourArea(c) < min_area:
                cv2.drawContours(mask, [c], -1, 0, -1)

        volume[:, :, i] = mask

    # Only keep components with volume larger than min_volume
    cc3d.dust(volume, threshold=min_volume, connectivity=6, in_place=True)

    f = open(mesh_name, "w")

    for slice in range(volume.shape[2]):
        mask = volume[:, :, slice]
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        contours = [
            contour for contour in contours if cv2.contourArea(contour) > min_area
        ]

        if len(contours) == 0:
            continue

        contours = np.concatenate(contours, axis=0)

        xs = contours[:, 0, 0]
        ys = contours[:, 0, 1]
        zs = np.ones_like(xs) * np.array([slice])

        for x, y, z in zip(xs, ys, zs):
            f.write(f"v {x} {y} {z}\n")

    f.close()
