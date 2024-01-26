import os

from PIL.Image import open as open_image
import numpy as np
from torch.utils.data import Dataset

import torch
from torchvision import tv_tensors
from torchvision.tv_tensors import Image, Mask


class SingleKidneyDataset(Dataset):
    def __init__(
        self,
        imgs_path,
        masks_path,
        resolution,
        ratio_segmented=1,
        dimension="2d",  # Determines the size of the imgs, if '2d' -> (B, 1, W, H), if '2.5d' -> (B, D, W, H), if '3d' -> (B, 1, D, W, H)
        volume_depth=1,
        transform=None,
    ):
        super().__init__()

        assert (
            type(volume_depth) == int and volume_depth >= 1
        ), "volume_depth must be an integer > 1"
        assert not (
            dimension == "2d" and volume_depth > 1
        ), f"For dimension '2d' the volume_depth must be 1, got {volume_depth}"
        assert not (
            dimension in ["2.5d", "3d"] and volume_depth == 1
        ), "volume_depth must be > 1 for dimension '2.5d' and '3d'"
        assert dimension in [
            "2d",
            "2.5d",
            "3d",
        ], f"dimension should be one of {['2d', '2.5d', '3d']}"

        self.imgs_path = imgs_path
        self.masks_path = masks_path
        self.resolution = resolution  # Distance between two pixels in micrometers
        self.ratio_segmented = ratio_segmented  # How much of the image is segmented (if 0.6, 40% of the vessel is not segmented)
        self.dimension = dimension
        self.volume_depth = volume_depth  # How many images are used to create a volume
        self.transform = transform
        if self.masks_path is None:
            self.files = os.listdir(self.imgs_path)
        else:
            self.files = os.listdir(self.masks_path)
        self.files.sort(key=lambda x: int(x.split(".")[0]))

    def __len__(self):
        return np.ceil(len(self.files) / self.volume_depth).astype(int)

    def __getitem__(self, idx):
        img, mask, img_paths, mask_paths = self.get_images_masks(idx)
        original_shape = img.shape[-2:]
        if self.transform is not None:
            with tv_tensors.set_return_type("TVTensor"):
                img = Image(img.unsqueeze(1))
                if mask is not None:
                    mask = Mask(mask)
                    img, mask = self.transform(img, mask)
                else:
                    img = self.transform(img)
                img = img.squeeze(1)
        if self.dimension == "2d" and mask is not None:
            mask = mask.squeeze(0)
        if self.dimension == "3d":
            img = img.unsqueeze(0)
        data = {
            "img": img,
            "mask": mask,
            "orignal_shape": original_shape,
            "resolution": self.resolution,
            "ratio_segmented": self.ratio_segmented,
            "img_paths": img_paths,
            "mask_paths": mask_paths,
        }
        return data

    def get_images_masks(self, idx):
        imgs, masks = [], []
        img_paths, mask_paths = [], []
        idx = idx * self.volume_depth
        if idx + self.volume_depth > len(self.files):
            idx = max(0, len(self.files) - self.volume_depth)
        for i in range(self.volume_depth):
            if idx + i >= len(self.files):
                img_path = None
                img_paths.append(img_path)
                img = np.zeros_like(imgs[-1])
            else:
                img_path = os.path.join(self.imgs_path, self.files[idx + i])
                img_paths.append(img_path)
                img = np.float32(open_image(img_path)) / np.iinfo(np.uint16).max  # type: ignore
                if i == 0:
                    imgs = np.zeros((self.volume_depth,) + img.shape, dtype=np.float32)
                imgs[i] = img
                if self.masks_path is not None:
                    mask_path = os.path.join(self.masks_path, self.files[idx + i])
                    mask_paths.append(mask_path)
                    mask = np.array(open_image(mask_path))
                    if i == 0:
                        masks = np.zeros(
                            (self.volume_depth,) + mask.shape, dtype=np.uint8
                        )
                    masks[i] = mask
        imgs = torch.from_numpy(imgs)
        if self.masks_path is not None:
            masks = torch.from_numpy(masks) / 255
        else:
            masks = None
        return imgs, masks, img_paths, mask_paths


class MultiKidneyDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = []
        dimension = datasets[0].dimension
        for dataset in datasets:
            if isinstance(dataset, SingleKidneyDataset):
                self.datasets.append(dataset)
            else:
                raise ValueError("All datasets must be of type SingleKidneyDataset")

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                return dataset[idx]
            else:
                idx -= len(dataset)
        raise IndexError("Index out of range")


datasets_configs = {
    "patient_1": [
        {
            "imgs_path": "kidney_1_dense/images",
            "masks_path": "kidney_1_dense/labels",
            "resolution": 50.0,
            "ratio_segmented": 1.0,
        },
        {
            "imgs_path": "kidney_1_voi/images",
            "masks_path": "kidney_1_voi/labels",
            "resolution": 5.2,
            "ratio_segmented": 1.0,
        },
    ],
    "patient_2": [
        {
            "imgs_path": "kidney_2/images",
            "masks_path": "kidney_2/labels",
            "resolution": 50,
            "ratio_segmented": 0.65,
        },
    ],
    "patient_3": [
        {
            "imgs_path": "kidney_3_sparse/images",
            "masks_path": "kidney_3_dense/labels",
            "resolution": 50.16,
            "ratio_segmented": 1.0,
        },
        {
            "imgs_path": "kidney_3_sparse/images",
            "masks_path": "kidney_3_sparse/labels",
            "resolution": 50.16,
            "ratio_segmented": 0.85,
        },
    ],
}


def get_dataloaders(
    data_path,
    val_patients,
    batch_size,
    num_workers,
    transform=None,
    volume_depth=1,
    dimension="2d",
):
    assert isinstance(val_patients, list) or isinstance(
        val_patients, tuple
    ), "val_patients must be a list or a tuple"
    train_datasets = []
    val_datasets = []
    for patient, configs in datasets_configs.items():
        for config in configs:
            config["imgs_path"] = os.path.join(data_path, config["imgs_path"])
            config["masks_path"] = os.path.join(data_path, config["masks_path"])
            config["volume_depth"] = volume_depth
            config["dimension"] = dimension
            config["transform"] = transform
            dataset = SingleKidneyDataset(**config)
            if patient in val_patients:
                val_datasets.append(dataset)
            else:
                train_datasets.append(dataset)

    train_dataset = MultiKidneyDataset(*train_datasets)
    val_dataset = MultiKidneyDataset(*val_datasets)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader
