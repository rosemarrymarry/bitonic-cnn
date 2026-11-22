from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitonic import DifferentiableRankBitonicFilter
from .legacy_model import TinyUNet


class BitonicGuidedDenoiser(nn.Module):
    """Full hybrid: rank-bitonic prior + CNN branch + learned gate."""

    def __init__(self, base_channels: int = 32, bitonic_radius: int = 2) -> None:
        super().__init__()
        self.bitonic = DifferentiableRankBitonicFilter(
            radius=bitonic_radius,
            centile=8.0,
            correction_sigma=1.0,
        )
        self.backbone = TinyUNet(in_ch=10, base=base_channels)
        self.head_denoise = nn.Conv2d(base_channels, 3, 3, padding=1)
        self.head_gate = nn.Sequential(nn.Conv2d(base_channels, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, y: torch.Tensor, sigma_map: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        xb = self.bitonic(y)
        residual = y - xb
        z = torch.cat([y, xb, residual, sigma_map], dim=1)
        feat = self.backbone(z)
        xd = torch.sigmoid(self.head_denoise(feat))
        gate = self.head_gate(feat)
        x_hat = (gate * xb + (1.0 - gate) * xd).clamp(0.0, 1.0)
        return x_hat, {
            "xb": xb,
            "xd": xd,
            "gate": gate,
            "tau": sigma_map.mean(dim=1, keepdim=True),
        }


class FixedFusionDenoiser(BitonicGuidedDenoiser):
    """Full branches with a fixed 0.5 fusion weight."""

    def forward(self, y: torch.Tensor, sigma_map: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        xb = self.bitonic(y)
        residual = y - xb
        z = torch.cat([y, xb, residual, sigma_map], dim=1)
        feat = self.backbone(z)
        xd = torch.sigmoid(self.head_denoise(feat))
        gate = torch.full((y.size(0), 1, y.size(2), y.size(3)), 0.5, device=y.device, dtype=y.dtype)
        x_hat = (0.5 * xb + 0.5 * xd).clamp(0.0, 1.0)
        return x_hat, {
            "xb": xb,
            "xd": xd,
            "gate": gate,
            "tau": sigma_map.mean(dim=1, keepdim=True),
        }


class DeepOnlyDenoiser(nn.Module):
    """Same-capacity U-Net control with the bitonic branch removed."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.backbone = TinyUNet(in_ch=10, base=base_channels)
        self.head_denoise = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, y: torch.Tensor, sigma_map: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zeros = torch.zeros_like(y)
        z = torch.cat([y, zeros, zeros, sigma_map], dim=1)
        feat = self.backbone(z)
        xd = torch.sigmoid(self.head_denoise(feat)).clamp(0.0, 1.0)
        return xd, {
            "xb": y,
            "xd": xd,
            "gate": torch.zeros((y.size(0), 1, y.size(2), y.size(3)), device=y.device, dtype=y.dtype),
            "tau": sigma_map.mean(dim=1, keepdim=True),
        }


class BitonicOnlyDenoiser(nn.Module):
    """Non-learned control: only the rank-bitonic filter."""

    def __init__(self, bitonic_radius: int = 2) -> None:
        super().__init__()
        self.bitonic = DifferentiableRankBitonicFilter(radius=bitonic_radius, centile=8.0)

    def forward(self, y: torch.Tensor, sigma_map: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        xb = self.bitonic(y).clamp(0.0, 1.0)
        return xb, {
            "xb": xb,
            "xd": xb,
            "gate": torch.zeros((y.size(0), 1, y.size(2), y.size(3)), device=y.device, dtype=y.dtype),
            "tau": sigma_map.mean(dim=1, keepdim=True),
        }


def build_model(kind: str, base_channels: int = 32, bitonic_radius: int = 2) -> nn.Module:
    if kind == "hybrid":
        return BitonicGuidedDenoiser(base_channels=base_channels, bitonic_radius=bitonic_radius)
    if kind == "fixed_fusion":
        return FixedFusionDenoiser(base_channels=base_channels, bitonic_radius=bitonic_radius)
    if kind == "deep_only":
        return DeepOnlyDenoiser(base_channels=base_channels)
    if kind == "bitonic_only":
        return BitonicOnlyDenoiser(bitonic_radius=bitonic_radius)
    raise ValueError(f"Unsupported model kind: {kind}")
