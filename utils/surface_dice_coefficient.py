from surface_distance import (
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
)
import torch
from torchmetrics import Metric


class SurfaceDice(Metric):
    def __init__(self, tolerance=0, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.dice = torch.tensor(0.0)
        self.numel = 0
        self.tolerance = tolerance
        self.add_state("surface_dice", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert (
            preds.shape == targets.shape
        ), "Predictions and targets must be of same shape"
        assert preds.ndim == 4, "Predictions must be of shape (N, D, H, W)"
        assert targets.ndim == 4, "Targets must be of shape (N, D, H, W)"
        assert preds.dtype == torch.uint8, "Predictions must be of type torch.uint8"
        assert targets.dtype == torch.uint8, "Targets must be of type torch.uint8"
        surface_dice_at_tolerance = 0
        for pred, target in zip(preds, targets):
            pred = pred.cpu().numpy().astype(bool)
            target = target.cpu().numpy().astype(bool)
            surface_distances = compute_surface_distances(pred, target, (1, 1, 1))
            surface_dice_at_tolerance += compute_surface_dice_at_tolerance(
                surface_distances, self.tolerance
            )
        self.dice += surface_dice_at_tolerance / preds.shape[0]
        self.numel += 1

    def compute(self):
        return self.dice / self.numel
