from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import ConvBlock, DownBlock, UpBlock
from .film import FiLMConditioner


class ConditionalHoleUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 32,
        tabular_dim: int = 69,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        widths = [base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 16]
        # Four-level U-Net: the stem is the first encoder level, followed by
        # three downsampling levels. The decoder restores all four skip
        # levels. The final adapter returns predictions to the existing 1/4
        # output grid used by target encoding and decoding.
        self.stem = ConvBlock(input_channels, widths[0], stride=2)
        self.down1 = DownBlock(widths[0], widths[1])
        self.down2 = DownBlock(widths[1], widths[2])
        self.down3 = DownBlock(widths[2], widths[3])
        self.film = FiLMConditioner(tabular_dim, mlp_hidden, widths[3], dropout)
        self.up3 = UpBlock(widths[3], widths[2], widths[2])
        self.up2 = UpBlock(widths[2], widths[1], widths[1])
        self.up1 = UpBlock(widths[1], widths[0], widths[0])
        self.output_adapter = ConvBlock(widths[0], widths[1], stride=2)
        self.shared = ConvBlock(widths[1], widths[1])
        self.heatmap_head = nn.Conv2d(widths[1], 3, 1)
        self.offset_head = nn.Conv2d(widths[1], 6, 1)
        self.mask_head = nn.Conv2d(widths[1], 3, 1)
        self.density_head = nn.Conv2d(widths[1], 3, 1)
        self.radius_head = nn.Conv2d(widths[1], 3, 1)
        # Four mutually exclusive classes: background, top, web, bottom.
        self.region_head = nn.Conv2d(widths[1], 4, 1)
        self.count_head = nn.Linear(widths[1], 3)
        # A 1% initial centre probability keeps the focal-loss background term
        # and its AMP-scaled gradients bounded on the large output grid.
        nn.init.constant_(self.heatmap_head.bias, -4.595)

    def encode(self, image: torch.Tensor):
        x0 = self.stem(image)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        return x0, x1, x2, x3

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0, x1, x2, bottleneck = self.encode(image)
        bottleneck = self.film(bottleneck, tabular)
        decoded = self.up3(bottleneck, x2)
        decoded = self.up2(decoded, x1)
        decoded = self.up1(decoded, x0)
        decoded = self.output_adapter(decoded)
        decoded = self.shared(decoded)
        pooled = decoded.mean(dim=(-2, -1))
        return {
            "heatmap_logits": self.heatmap_head(decoded),
            "offset": self.offset_head(decoded),
            "mask_logits": self.mask_head(decoded),
            "density": self.density_head(decoded),
            "log_radius": self.radius_head(decoded),
            "region_logits": self.region_head(decoded),
            "count": F.softplus(self.count_head(pooled)),
        }

    def freeze_encoder(self, frozen: bool = True) -> None:
        modules = [self.stem, self.down1, self.down2, self.down3]
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = not frozen
