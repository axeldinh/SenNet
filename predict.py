import os
import matplotlib.pyplot as plt
from PIL.Image import open

import numpy as np
import torch
from torchvision.transforms.v2 import Compose, Resize, Normalize

from lightning_module import LitModel
from dataset import SingleKidneyDataset


model_path = (
    "C:/Users/Axeld/Desktop/SenNet/checkpoints/model-epoch=00-val/SurfaceDice=0.02.ckpt"
)


model = LitModel.load_from_checkpoint(model_path)

test_path = "C:/Users/Axeld/Desktop/SenNet/blood-vessel-segmentation/test"

# Get the resizing parameters
size = [
    x.size for x in model.transform.transforms if "Resize" in str(x.__class__.__name__)
][0]
transform = Compose(
    [x for x in model.transform.transforms if "Normalize" in str(x.__class__.__name__)]
    + [Resize(size, antialias=True)]
)


for dataset in os.listdir(test_path):
    dataset_path = os.path.join(test_path, dataset, "images")

    dataset = SingleKidneyDataset(
        dataset_path,
        None,
        resolution=None,
        ratio_segmented=None,
        dimension="2d",
        transform=transform,
    )
