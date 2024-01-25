import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler
from pytorch_lightning import Trainer

from torchvision.transforms import v2

from lightning_module import LitModel
from utils.transforms import RemoveSmallVessel
from utils.focal_loss import FocalLoss
from models import UNet
from dataset import get_dataloaders


def train(
    pathdata,
    batch_size=1,
    epochs=1,
    img_size=64,
    volume_depth=8,
    min_area=50,
    unet_depth=2,
    init_feature=16,
    dimension="2.5d",
    val_patient=["patient_3"],
    debug=False,
):
    data_path = pathdata

    transform = v2.Compose(
        [
            RemoveSmallVessel(min_area=min_area),
            v2.Normalize(mean=[0.3535], std=[0.0396]),
            v2.RandomResizedCrop(
                img_size, scale=(0.5, 2.0), ratio=(0.75, 4 / 3), antialias=False
            ),
            v2.RandomAffine(
                degrees=15,  # type: ignore
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=15,
                interpolation=v2.InterpolationMode.NEAREST,
            ),
            v2.ElasticTransform(
                alpha=(0, img_size * 0.1),
                sigma=(5, 10),
                interpolation=v2.InterpolationMode.NEAREST,
            ),
            v2.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.1, hue=0.1),
            v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.01, 2.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ]
    )

    train_loader, val_loader = get_dataloaders(
        data_path,
        val_patients=val_patient,
        volume_depth=volume_depth,
        transform=transform,
        batch_size=batch_size,
        dimension=dimension,
        num_workers=0,
    )

    model = UNet(
        in_channels=1 if dimension in ["2d", "3d"] else volume_depth,
        out_channels=1 if dimension in ["2d", "3d"] else volume_depth,
        init_features=init_feature,
        depth=unet_depth,
        residual=True,
        attention=True,
        activation=nn.ReLU(),
        dimension=3 if dimension == "3d" else 2,
    )

    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    lit_model = LitModel(
        model,
        criterion,
        optimizer,
        scheduler,
        batch_size=batch_size,
        transform=transform,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/SurfaceDice",
        dirpath="checkpoints",
        filename="model-{epoch:02d}-{val/SurfaceDice:.2f}",
        save_top_k=1,
        mode="max",
    )

    if debug:
        from time import time

        start = time()
        for batch in train_loader:
            print("Time to load batch:", time() - start)
            img = batch["img"]
            mask = batch["mask"]
            print("Image Shape:", img.shape)
            print("Mask Shape:", mask.shape)
            print(batch.keys())
            start = time()
            y = lit_model(img)
            print("Model output shape:", y.shape)
            print("Time to forward pass:", time() - start)
            break
        profiler = AdvancedProfiler(filename="profiler.txt")
        trainer = Trainer(
            max_epochs=1,
            callbacks=[checkpoint_callback],
            limit_train_batches=0.002,
            limit_val_batches=0.01,
            log_every_n_steps=1,
            profiler=profiler,
        )
    else:
        trainer = Trainer(
            max_epochs=epochs,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
        )

    trainer.fit(lit_model, train_loader, val_loader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "-p",
        "--pathdata",
        type=str,
        default="C:/Users/Axeld/Desktop/SenNet/blood-vessel-segmentation/train",
        help="Data path",
    )
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    args = vars(args)
    del args["seed"]

    train(**args)
