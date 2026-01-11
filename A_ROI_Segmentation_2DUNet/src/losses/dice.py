import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation using logits.
    Internally applies sigmoid(logits).
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)

        pred = pred.contiguous()
        target = target.contiguous()

        # pred/target shape: (B,1,H,W)
        intersection = (pred * target).sum(dim=(2, 3))
        denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()
