from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_real_noise import RealNoisyPairDataset, center_crop_pair, load_rgb, resize_to_max_side
from src.dl_bitonic.metrics import psnr, ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SIDD noisy-clean pairing without loading a denoiser.")
    parser.add_argument("--noisy-root", type=str, required=True)
    parser.add_argument("--clean-root", type=str, required=True)
    parser.add_argument("--show-pairs", type=int, default=10, help="Print the first N matched pairs.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Number of pairs used for noisy-input PSNR/SSIM. Use 0 to scan all pairs.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="0 keeps native resolution. Positive values resize images and are for smoke tests only.",
    )
    parser.add_argument("--save-manifest", type=str, default="")
    parser.add_argument("--save-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = RealNoisyPairDataset(
        noisy_root=args.noisy_root,
        clean_root=args.clean_root,
        max_side=args.max_side,
    )

    print(f"Matched pairs: {len(dataset)}")
    for noisy_path, clean_path in dataset.pairs[: max(0, args.show_pairs)]:
        print(f"  {noisy_path.name}  <->  {clean_path.name}")

    if args.save_manifest:
        dataset.write_manifest(args.save_manifest)
        print(f"Saved pair manifest to: {args.save_manifest}")

    limit = len(dataset.pairs) if args.max_samples == 0 else min(len(dataset.pairs), args.max_samples)
    psnr_values: list[float] = []
    ssim_values: list[float] = []

    with torch.no_grad():
        for index in range(limit):
            noisy_path, clean_path = dataset.pairs[index]
            noisy = load_rgb(noisy_path)
            clean = load_rgb(clean_path)
            noisy, clean = center_crop_pair(noisy, clean)
            noisy = resize_to_max_side(noisy, args.max_side)
            clean = resize_to_max_side(clean, args.max_side)
            psnr_values.append(float(psnr(noisy.unsqueeze(0), clean.unsqueeze(0))))
            ssim_values.append(float(ssim(noisy.unsqueeze(0), clean.unsqueeze(0))))

    result = {
        "matched_pairs": len(dataset),
        "checked_pairs": limit,
        "noisy_psnr_mean": float(np.mean(psnr_values)) if psnr_values else None,
        "noisy_ssim_mean": float(np.mean(ssim_values)) if ssim_values else None,
        "max_side": args.max_side,
    }
    print(json.dumps(result, indent=2))

    if args.save_json:
        output = Path(args.save_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
