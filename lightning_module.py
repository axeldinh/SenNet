import torch
import pytorch_lightning as pl
from torchmetrics import MetricCollection, Dice, Accuracy, F1Score, Precision, Recall
from utils.surface_dice_coefficient import SurfaceDice


class LitModel(pl.LightningModule):
    def __init__(
        self, model, criterion, optimizer, scheduler, batch_size=1, transform=None
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = MetricCollection(
            Dice(ignore_index=0),
            Accuracy(task="binary", ignore_index=0),
            F1Score(task="binary", ignore_index=0),
            Precision(task="binary", ignore_index=0),
            Recall(task="binary", ignore_index=0),
            SurfaceDice(tolerance=0),
        )
        self.batch_size = batch_size
        self.transform = transform
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["img"]
        y = batch["mask"]
        y_hat = self.model(x).squeeze(1)
        loss = self.criterion(y_hat, y)
        self.log(
            "train/loss",
            loss,
            batch_size=self.batch_size,
        )
        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat > 0.5).to(torch.uint8)
        y = y.to(torch.uint8)
        if len(y.shape) == 3:
            y = y.unsqueeze(1)
        if len(y_hat.shape) == 3:
            y_hat = y_hat.unsqueeze(1)
        self.metrics(y_hat, y)
        for name, metric in self.metrics.items():
            self.log(
                f"train/{name}",
                metric,
                batch_size=self.batch_size,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["img"]
        y = batch["mask"]
        y_hat = self.model(x).squeeze(1)
        loss = self.criterion(y_hat, y)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat > 0.5).to(torch.uint8)
        y = y.to(torch.uint8)
        if len(y.shape) == 3:
            y = y.unsqueeze(1)
        if len(y_hat.shape) == 3:
            y_hat = y_hat.unsqueeze(1)
        self.metrics(y_hat, y)
        for name, metric in self.metrics.items():
            self.log(
                f"val/{name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.batch_size,
            )
        return loss

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
            "monitor": "train/loss",
        }
