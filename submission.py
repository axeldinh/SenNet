import os

import numpy as np
import cv2
import torch

from torchvision.transforms.v2.functional import resize
from torchvision.transforms.v2 import InterpolationMode

from lightning_module import LitModel
from dataset import SingleKidneyDataset

import warnings

import pandas as pd

warnings.filterwarnings("ignore", message=".*is an instance of `nn.Module`.*")


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def save_predictions(img_path, pred, path, save_overlay=False):
    """
    img and pred are (W, H) np.ndarray
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the pure masks
    pred = pred * 255
    pred = np.uint8(pred)
    cv2.imwrite(path, pred)  # type: ignore

    if not save_overlay:
        return

    # Save the overlay
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred = np.stack([np.zeros_like(pred), np.zeros_like(pred), pred], axis=-1)
    overlay = img.copy()
    overlay[pred > 0] = 0.5 * img[pred > 0] + 0.5 * pred[pred > 0]  # type: ignore
    cv2.imwrite(path.replace(".png", "_overlay.png"), overlay)


def predict(
    model_path,
    imgs_folder,
    save_path="predictions",
    save_masks=False,
    save_overlay=False,
    return_submission=True,
):
    os.makedirs(save_path, exist_ok=True)
    model = LitModel.load_from_checkpoint(model_path)
    model.eval()

    slices = []
    rles = []

    dataset = SingleKidneyDataset(
        imgs_folder,
        None,
        resolution=0,
        volume_depth=model.volume_depth,
        ratio_segmented=0,
        dimension=model.dimension,
        transform=model.predict_transform,
    )

    for i in range(len(dataset)):
        data = dataset[i]
        img = data["img"]
        shape = data["orignal_shape"]
        y = model(img.unsqueeze(0))
        y = y.squeeze(0).cpu().detach()
        if len(y.shape) == 2:
            y = y.unsqueeze(0)
        y = torch.sigmoid(y)
        y = resize(y, shape, antialias=True).numpy()
        for i in range(len(y)):
            img_path = data["img_paths"][i]
            if img_path is None:
                continue
            slice_ = os.path.basename(img_path).split(".")[0]
            rle = rle_encode((y[i] > 0.5).astype(np.uint8))
            rle = "1 0" if rle == "" else rle
            slices.append(slice_)
            rles.append(rle)
            if not save_masks:
                continue
            save_path_tmp = os.path.join(save_path, slice_ + ".png")
            save_predictions(
                img_path,
                y[i],
                save_path_tmp,
                save_overlay=save_overlay,
            )
    submission_df = pd.DataFrame({"id": slices, "rle": rles})
    if return_submission:
        return submission_df


def make_submission(
    test_path, model_path, submission_path, results_path="./results/", save_results=True
):
    dfs = []
    for dataset_name in os.listdir(test_path):
        imgs_folder = os.path.join(test_path, dataset_name, "images")
        save_path = os.path.join(results_path, dataset_name)
        dfs.append(
            predict(
                model_path,
                imgs_folder,
                save_path=save_path,
                save_masks=save_results,
                save_overlay=save_results,
                return_submission=True,
            )
        )
    submission_df = pd.concat(dfs)
    submission_df.to_csv(submission_path, index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--test_path",
        type=str,
        default="C:/Users/Axeld/Desktop/SenNet/blood-vessel-segmentation/test",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="C:/Users/Axeld/Desktop/SenNet/checkpoints/model-best.ckpt",
    )
    parser.add_argument("--submission_path", type=str, default="./submission.csv")
    parser.add_argument("--results_path", type=str, default="./results/")
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()

    make_submission(**vars(args))
