from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableBitonicPrior(nn.Module):
    """
    Differentiable local adaptive bitonic-like filter.

    It computes a robust weighted local average where weights depend on:
    1) spatial Gaussian distance,
    2) intensity distance to center pixel with adaptive threshold tau.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        softness: float = 20.0,
        max_chunk_pixels: int = 256 * 256,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        if max_chunk_pixels <= 0:
            raise ValueError("max_chunk_pixels must be positive")

        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.softness = softness
        self.max_chunk_pixels = int(max_chunk_pixels)
        self.tile_size = max(1, math.isqrt(self.max_chunk_pixels))

        coords = torch.arange(kernel_size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        center = (kernel_size - 1) / 2.0
        dist2 = (xx - center) ** 2 + (yy - center) ** 2
        spatial = torch.exp(-dist2 / (2.0 * (0.35 * kernel_size) ** 2))
        spatial = spatial / spatial.sum()
        self.register_buffer("spatial_weights", spatial.view(1, 1, -1, 1), persistent=False)

    def _forward_chunk(self, x_padded: torch.Tensor, tau_core: torch.Tensor) -> torch.Tensor:
        """Apply the filter to one pre-padded tile and return only its core region."""
        b, c, padded_h, padded_w = x_padded.shape
        out_h = padded_h - 2 * self.pad
        out_w = padded_w - 2 * self.pad
        if out_h <= 0 or out_w <= 0:
            raise ValueError("Bitonic filter tile is smaller than the kernel.")

        patches = F.unfold(x_padded, kernel_size=self.kernel_size, padding=0)
        patches = patches.view(b, c, self.kernel_size * self.kernel_size, out_h * out_w)

        center_idx = (self.kernel_size * self.kernel_size) // 2
        center = patches[:, :, center_idx : center_idx + 1, :]
        abs_diff = (patches - center).abs()

        tau = tau_core.reshape(b, 1, 1, out_h * out_w).clamp(1e-4, 1.0)
        range_w = torch.sigmoid((tau - abs_diff) * self.softness)
        weights = (range_w * self.spatial_weights).clamp_min(1e-8)

        weighted = (patches * weights).sum(dim=2)
        norm = weights.sum(dim=2)
        out = weighted / norm
        return out.view(b, c, out_h, out_w)

    def forward(self, x: torch.Tensor, tau_map: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] in [0, 1]
        tau_map: [B, 1, H, W] adaptive threshold in [0, 1]

        Large images are processed in overlapping tiles so that the temporary
        [B, C, K*K, H*W] unfold tensor does not have to be materialized for the
        entire image. The output remains at the original resolution.
        """
        b, c, h, w = x.shape
        x_padded = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="constant", value=0.0)

        if h * w <= self.max_chunk_pixels:
            return self._forward_chunk(x_padded, tau_map)

        row_outputs: list[torch.Tensor] = []
        for top in range(0, h, self.tile_size):
            tile_h = min(self.tile_size, h - top)
            col_outputs: list[torch.Tensor] = []

            for left in range(0, w, self.tile_size):
                tile_w = min(self.tile_size, w - left)
                x_tile = x_padded[
                    :,
                    :,
                    top : top + tile_h + 2 * self.pad,
                    left : left + tile_w + 2 * self.pad,
                ]
                tau_tile = tau_map[:, :, top : top + tile_h, left : left + tile_w]
                col_outputs.append(self._forward_chunk(x_tile, tau_tile))

            row_outputs.append(torch.cat(col_outputs, dim=3))

        return torch.cat(row_outputs, dim=2)


class TauPredictor(nn.Module):
    """Predict adaptive bitonic threshold from image and sigma map."""

    def __init__(self, in_channels: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, sigma_map: torch.Tensor) -> torch.Tensor:
        tau = self.net(torch.cat([x, sigma_map], dim=1))
        # Keep useful dynamic range, avoid too small tau.
        return 0.02 + 0.25 * tau
