from __future__ import annotations

import torch

from src.dl_bitonic.model import build_model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.rand(1, 3, 64, 64, device=device)
    sigma_map = torch.full((1, 1, 64, 64), 25.0 / 75.0, device=device)

    for kind in ["hybrid", "fixed_fusion", "deep_only", "bitonic_only"]:
        model = build_model(kind, base_channels=32, bitonic_radius=2).to(device)
        model.eval()
        with torch.inference_mode():
            pred, aux = model(y, sigma_map)
        params = sum(p.numel() for p in model.parameters())
        print(
            kind,
            "params=", params,
            "pred=", tuple(pred.shape),
            "xb=", tuple(aux["xb"].shape),
            "xd=", tuple(aux["xd"].shape),
            "gate=", tuple(aux["gate"].shape),
            "range=", (float(pred.min()), float(pred.max())),
        )


if __name__ == "__main__":
    main()
