from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianBlur(nn.Module):
    """Fixed depthwise Gaussian blur used by the bitonic correction step."""

    def __init__(self, radius: int = 2, sigma: float = 1.0) -> None:
        super().__init__()
        if radius < 1:
            raise ValueError("radius must be positive")
        self.radius = int(radius)
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.view(1, 1, 2 * radius + 1, 2 * radius + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        kernel = self.kernel.to(device=x.device, dtype=x.dtype).repeat(channels, 1, 1, 1)
        return F.conv2d(x, kernel, padding=self.radius, groups=channels)


class DifferentiableRankBitonicFilter(nn.Module):
    """Differentiable rank-opening/closing approximation of Treece's Bitonic filter.

    The original Bitonic filter uses a ranking/morphological framework followed by
    Gaussian correction of the opening and closing residuals. Exact rank-opening
    implementations are non-differentiable; this module uses torch.sort to obtain
    low/high local ranks, which keeps gradients with respect to pixel values.

    This is intentionally named a rank-bitonic approximation, not an exact
    reimplementation of the MATLAB rankopen2 routine.
    """

    def __init__(
        self,
        radius: int = 2,
        centile: float = 8.0,
        correction_sigma: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if radius < 1:
            raise ValueError("radius must be positive")
        if not (0.0 < centile < 50.0):
            raise ValueError("centile must be in (0, 50)")

        self.radius = int(radius)
        self.kernel_size = 2 * radius + 1
        self.centile = float(centile)
        self.eps = float(eps)
        self.blur = GaussianBlur(radius=radius, sigma=correction_sigma)

    def forward(self, x: torch.Tensor, threshold: torch.Tensor | None = None) -> torch.Tensor:
        """Return a rank-bitonic filtered image.

        x: [B, C, H, W], values typically in [0, 1].
        threshold: optional [B, 1, H, W] noise threshold, reserved for a future
                   soft rank/threshold extension. The current approximation uses
                   a fixed centile as in the reference implementation default.
        """
        del threshold  # reserved for a learned soft-rank extension

        b, c, h, w = x.shape
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.radius,
        )
        patches = patches.view(b, c, self.kernel_size * self.kernel_size, h * w)

        # Differentiable local ranking. torch.sort routes gradients through the
        # selected values, unlike kthvalue/index_select.
        sorted_patches, _ = torch.sort(patches, dim=2)
        rank = int(round(self.centile / 100.0 * (self.kernel_size * self.kernel_size - 1)))
        rank = max(0, min(rank, self.kernel_size * self.kernel_size - 1))
        high_rank = self.kernel_size * self.kernel_size - 1 - rank

        a_open = sorted_patches[:, :, rank, :].view(b, c, h, w)
        a_close = sorted_patches[:, :, high_rank, :].view(b, c, h, w)

        e_open = self.blur(x - a_open)
        e_close = self.blur(a_close - x)

        a_open_corrected = a_open + e_open
        a_close_corrected = a_close - e_close

        w_open = (e_open.abs() + self.eps).pow(3)
        w_close = (e_close.abs() + self.eps).pow(3)
        return (w_open * a_close_corrected + w_close * a_open_corrected) / (
            w_open + w_close + self.eps
        )


# Backward-compatible alias for scripts that still refer to the old class name.
DifferentiableBitonicPrior = DifferentiableRankBitonicFilter
