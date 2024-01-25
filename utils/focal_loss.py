import torch


class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        loss = -self.alpha * (1 - outputs) ** self.gamma * targets * torch.log(
            outputs
        ) - (1 - self.alpha) * outputs**self.gamma * (1 - targets) * torch.log(
            1 - outputs
        )
        return loss.mean()
