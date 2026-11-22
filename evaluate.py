from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dl_bitonic.dataset import GaussianNoisyPairDataset
from src.dl_bitonic.metrics import psnr, ssim
from src.dl_bitonic.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate bitonic-guided denoiser")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sigma", type=float, default=25.0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=123, help="Seed used for deterministic data-loader noise generation.")
    p.add_argument(
        "--eval-patch-size",
        type=int,
        default=0,
        help="0 evaluates each full image. Positive values use a center crop for smoke tests only.",
    )
    p.add_argument("--save-json", type=str, default="")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    model_kind = ckpt_args.get("model_kind", "hybrid")
    model = build_model(
        model_kind,
        base_channels=int(ckpt_args.get("base_ch", 32)),
        bitonic_radius=int(ckpt_args.get("bitonic_radius", 2)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    ds = GaussianNoisyPairDataset(
        root=args.data_root,
        split="val",
        patch_size=128,
        fixed_sigma=args.sigma,
        train=False,
        seed=args.seed,
        eval_patch_size=args.eval_patch_size,
        deterministic_noise=True,
        noise_seed=args.seed + 1000,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers)

    psnr_vals, ssim_vals = [], []
    noisy_psnr_vals, noisy_ssim_vals, psnr_gain_vals = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval sigma={args.sigma}", ncols=120):
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)
            sigma_map = batch["sigma_map"].to(device)

            pred, _ = model(noisy, sigma_map)
            pred_psnr = psnr(pred, clean)
            noisy_psnr = psnr(noisy, clean)
            psnr_vals.append(pred_psnr)
            ssim_vals.append(ssim(pred, clean))
            noisy_psnr_vals.append(noisy_psnr)
            noisy_ssim_vals.append(ssim(noisy, clean))
            psnr_gain_vals.append(pred_psnr - noisy_psnr)

    result = {
        "sigma": args.sigma,
        "num_samples": len(psnr_vals),
        "psnr_noisy_mean": float(np.mean(noisy_psnr_vals)),
        "ssim_noisy_mean": float(np.mean(noisy_ssim_vals)),
        "psnr_mean": float(np.mean(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "psnr_gain_mean": float(np.mean(psnr_gain_vals)),
        "eval_patch_size": args.eval_patch_size,
        "seed": args.seed,
        "noise_seed": args.seed + 1000,
    }

    print(json.dumps(result, indent=2))
    if args.save_json:
        output = Path(args.save_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
