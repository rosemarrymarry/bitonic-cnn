from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.inference_mode()
def _forward_full(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    noisy_rgb = noisy.unsqueeze(0).to(device)
    sigma_map = torch.full(
        (1, 1, noisy_rgb.shape[2], noisy_rgb.shape[3]),
        float(sigma) / 75.0,
        dtype=noisy_rgb.dtype,
        device=device,
    )
    pred, _ = model(noisy_rgb, sigma_map)
    return pred.squeeze(0).detach().cpu().clamp(0.0, 1.0)


@torch.inference_mode()
def infer_tiled(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    sigma: float,
    device: torch.device,
    tile_size: int = 256,
    tile_halo: int = 96,
) -> torch.Tensor:
    """Run a fully convolutional denoiser in overlapping tiles.

    Core output tiles are non-overlapping, while each model input includes a
    context halo around the core tile. This keeps native output resolution and
    avoids one enormous activation tensor for high-resolution images.
    """
    if noisy.ndim != 3:
        raise ValueError(f"Expected noisy tensor [C, H, W], got shape {tuple(noisy.shape)}")
    if tile_halo < 0:
        raise ValueError("tile_halo must be non-negative")

    channels, height, width = noisy.shape
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        return _forward_full(model, noisy, sigma, device)
    if tile_halo >= tile_size:
        raise ValueError("tile_halo must be smaller than tile_size")

    padded = F.pad(noisy, (tile_halo, tile_halo, tile_halo, tile_halo), mode="replicate")
    output = torch.empty((channels, height, width), dtype=noisy.dtype)

    for top in range(0, height, tile_size):
        core_h = min(tile_size, height - top)
        for left in range(0, width, tile_size):
            core_w = min(tile_size, width - left)
            tile = padded[
                :,
                top : top + core_h + 2 * tile_halo,
                left : left + core_w + 2 * tile_halo,
            ]
            pred = _forward_full(model, tile, sigma, device)
            output[
                :,
                top : top + core_h,
                left : left + core_w,
            ] = pred[:, tile_halo : tile_halo + core_h, tile_halo : tile_halo + core_w]

    return output
