import torch
import torch.nn as nn


class UNet2D(nn.Module):
    """
    A small 2D U-Net for binary segmentation.
    IMPORTANT: outputs logits (no sigmoid).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        logits = self.out_conv(d1)  # logits (no sigmoid)
        return logits
