from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dl_bitonic.inference import infer_tiled
from src.dl_bitonic.metrics import psnr, ssim
from src.dl_bitonic.model import build_model
from src.dl_bitonic.legacy_model import BitonicGuidedDenoiser as LegacyBitonicGuidedDenoiser

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_rgb(path: str | Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def center_crop_pair(noisy: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = min(noisy.shape[1], clean.shape[1])
    w = min(noisy.shape[2], clean.shape[2])
    if h <= 0 or w <= 0:
        raise ValueError("Image size is invalid after crop.")

    top_noisy = (noisy.shape[1] - h) // 2
    left_noisy = (noisy.shape[2] - w) // 2
    top_clean = (clean.shape[1] - h) // 2
    left_clean = (clean.shape[2] - w) // 2

    noisy = noisy[:, top_noisy : top_noisy + h, left_noisy : left_noisy + w]
    clean = clean[:, top_clean : top_clean + h, left_clean : left_clean + w]
    return noisy, clean


def resize_to_max_side(img: torch.Tensor, max_side: int = 0) -> torch.Tensor:
    if max_side <= 0:
        return img

    h, w = img.shape[1], img.shape[2]
    if max(h, w) <= max_side:
        return img

    scale = max_side / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def normalize_name(name: str) -> str:
    """Remove only modality markers while preserving scene/crop identifiers.

    Examples:
      0001_NOISY_SRGB_010.PNG -> 0001_srgb_010
      0001_GT_SRGB_010.PNG    -> 0001_srgb_010
      0400_NOISY_SRGB_010.PNG -> 0400_srgb_010
    """
    s = Path(name).name.strip().lower()
    s = re.sub(r"^(noisy|gt|clean)[_-]?", "", s)
    s = re.sub(r"_(noisy|gt|clean)(?=_|\.)", "", s)
    if Path(s).suffix.lower() in VALID_SUFFIXES:
        s = Path(s).stem
    return s.replace(" ", "")


def pair_key(path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root)
    return rel.parent.as_posix().lower(), normalize_name(path.name)


def _is_noisy_like(path: Path) -> bool:
    low = path.name.lower()
    return low.startswith("noisy") or "_noisy" in low or "noisy_" in low


def _is_clean_like(path: Path) -> bool:
    low = path.name.lower()
    return (
        low.startswith("gt")
        or low.startswith("clean")
        or "_gt" in low
        or "gt_" in low
        or "_clean" in low
        or "clean_" in low
    )


def _build_file_index(root: Path, files: list[Path]) -> tuple[dict[tuple[str, str], Path], dict[str, list[Path]]]:
    rel_map: dict[tuple[str, str], Path] = {}
    name_map: dict[str, list[Path]] = {}

    for path in files:
        key = pair_key(path, root)
        if key in rel_map:
            raise ValueError(
                "Duplicate canonical pair key detected. Refusing to silently overwrite files:\n"
                f"key={key}\nfirst={rel_map[key]}\nsecond={path}"
            )
        rel_map[key] = path
        name_map.setdefault(normalize_name(path.name), []).append(path)

    return rel_map, name_map


class RealNoisyPairDataset(Dataset):
    def __init__(
        self,
        noisy_root: str | Path,
        clean_root: str | Path,
        sigma: float = 25.0,
        max_samples: int | None = None,
        max_side: int = 0,
    ):
        self.noisy_root = Path(noisy_root)
        self.clean_root = Path(clean_root)
        self.sigma = float(sigma)
        self.max_side = int(max_side)

        if not self.noisy_root.exists():
            raise FileNotFoundError(f"Noisy root not found: {self.noisy_root}")
        if not self.clean_root.exists():
            raise FileNotFoundError(f"Clean root not found: {self.clean_root}")

        noisy_files = sorted(
            [
                p
                for p in self.noisy_root.rglob("*")
                if p.is_file()
                and p.suffix.lower() in VALID_SUFFIXES
                and (_is_noisy_like(p) or not _is_clean_like(p))
            ]
        )
        clean_files = sorted(
            [
                p
                for p in self.clean_root.rglob("*")
                if p.is_file()
                and p.suffix.lower() in VALID_SUFFIXES
                and (_is_clean_like(p) or not _is_noisy_like(p))
            ]
        )

        if not noisy_files:
            raise FileNotFoundError(f"No candidate noisy images found under: {self.noisy_root}")
        if not clean_files:
            raise FileNotFoundError(f"No candidate clean images found under: {self.clean_root}")

        clean_rel_map, clean_name_map = _build_file_index(self.clean_root, clean_files)

        self.pairs: list[tuple[Path, Path]] = []
        missing: list[Path] = []
        ambiguous: list[Path] = []

        for noisy_path in noisy_files:
            rel_key = pair_key(noisy_path, self.noisy_root)
            clean_path = clean_rel_map.get(rel_key)

            if clean_path is None:
                candidates = clean_name_map.get(normalize_name(noisy_path.name), [])
                if len(candidates) == 1:
                    clean_path = candidates[0]
                elif len(candidates) > 1:
                    ambiguous.append(noisy_path)
                    continue

            if clean_path is None:
                missing.append(noisy_path)
                continue
            if clean_path.resolve() == noisy_path.resolve():
                continue

            self.pairs.append((noisy_path, clean_path))

        if ambiguous:
            preview = "\n".join(str(p) for p in ambiguous[:10])
            raise ValueError(
                "Ambiguous noisy-clean matching: multiple clean files share the same canonical basename. "
                "Use matching relative directory structure or an explicit manifest.\n"
                f"Ambiguous noisy files:\n{preview}"
            )
        if missing:
            preview = "\n".join(str(p) for p in missing[:10])
            raise FileNotFoundError(
                f"No matching clean image found for {len(missing)} noisy files. First unmatched files:\n{preview}"
            )
        if not self.pairs:
            raise FileNotFoundError(
                f"No matched noisy-clean pairs found between {self.noisy_root} and {self.clean_root}."
            )

        self.pairs.sort(key=lambda item: item[0].relative_to(self.noisy_root).as_posix())
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        noisy_path, clean_path = self.pairs[index]
        noisy = load_rgb(noisy_path)
        clean = load_rgb(clean_path)
        noisy, clean = center_crop_pair(noisy, clean)
        noisy = resize_to_max_side(noisy, self.max_side)
        clean = resize_to_max_side(clean, self.max_side)

        sigma_map = torch.full((1, noisy.shape[1], noisy.shape[2]), self.sigma / 75.0, dtype=noisy.dtype)
        return {
            "clean": clean,
            "noisy": noisy,
            "sigma_map": sigma_map,
        }

    def write_manifest(self, output_path: str | Path) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["relative_parent", "canonical_name", "noisy_path", "clean_path"],
            )
            writer.writeheader()
            for noisy_path, clean_path in self.pairs:
                key_parent, key_name = pair_key(noisy_path, self.noisy_root)
                writer.writerow(
                    {
                        "relative_parent": key_parent,
                        "canonical_name": key_name,
                        "noisy_path": str(noisy_path),
                        "clean_path": str(clean_path),
                    }
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a denoiser on real noisy-clean pairs.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint .pt file.")
    parser.add_argument("--noisy-root", type=str, default="", help="Folder containing noisy images.")
    parser.add_argument("--clean-root", type=str, default="", help="Folder containing clean references.")
    parser.add_argument("--noisy-img", type=str, default="", help="Single noisy image path for one-pair evaluation.")
    parser.add_argument("--clean-img", type=str, default="", help="Single clean image path for one-pair evaluation.")
    parser.add_argument("--sigma", type=float, default=25.0, help="Approximate noise level used to build sigma_map.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="0 keeps native resolution. Positive values resize images and are for smoke tests only.",
    )
    parser.add_argument("--show-pairs", type=int, default=5, help="Print the first N matched pairs.")
    parser.add_argument("--tile-size", type=int, default=256, help="Core tile size for full-resolution inference. Use 0 to disable tiling.")
    parser.add_argument("--tile-halo", type=int, default=96, help="Context halo around each inference tile.")
    parser.add_argument("--save-manifest", type=str, default="", help="Optional CSV path for the matched pair manifest.")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save evaluation JSON.")
    return parser.parse_args()


def evaluate_single_pair(
    model: torch.nn.Module,
    noisy_img: str | Path,
    clean_img: str | Path,
    sigma: float,
    device: torch.device,
    max_side: int = 0,
    tile_size: int = 256,
    tile_halo: int = 96,
) -> dict[str, float]:
    noisy = load_rgb(noisy_img)
    clean = load_rgb(clean_img)
    noisy, clean = center_crop_pair(noisy, clean)
    noisy = resize_to_max_side(noisy, max_side)
    clean = resize_to_max_side(clean, max_side)

    model.eval()
    pred = infer_tiled(
        model,
        noisy,
        sigma=sigma,
        device=device,
        tile_size=tile_size,
        tile_halo=tile_halo,
    )

    noisy_batch = noisy.unsqueeze(0)
    clean_batch = clean.unsqueeze(0)
    pred_batch = pred.unsqueeze(0)
    noisy_psnr = float(psnr(noisy_batch, clean_batch))
    noisy_ssim = float(ssim(noisy_batch, clean_batch))
    pred_psnr = float(psnr(pred_batch, clean_batch))
    pred_ssim = float(ssim(pred_batch, clean_batch))
    return {
        "psnr_noisy": noisy_psnr,
        "ssim_noisy": noisy_ssim,
        "psnr": pred_psnr,
        "ssim": pred_ssim,
        "psnr_gain": pred_psnr - noisy_psnr,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    if "model_kind" in ckpt_args:
        model = build_model(
            ckpt_args["model_kind"],
            base_channels=int(ckpt_args.get("base_ch", 32)),
            bitonic_radius=int(ckpt_args.get("bitonic_radius", 2)),
        )
    else:
        model = LegacyBitonicGuidedDenoiser(
            use_sigma_aware_tau=not bool(ckpt_args.get("disable_sigma_aware_tau", False)),
            base_channels=int(ckpt_args.get("base_ch", 32)),
        )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    if args.max_side > 0:
        print(
            "WARNING: --max-side > 0 resizes real-noise images and changes the evaluation protocol. "
            "Use native resolution for final paper results."
        )

    if args.noisy_img and args.clean_img:
        result = evaluate_single_pair(
            model,
            args.noisy_img,
            args.clean_img,
            args.sigma,
            device,
            max_side=args.max_side,
            tile_size=args.tile_size,
            tile_halo=args.tile_halo,
        )
        payload = {"single_pair": True, **result}
        print(json.dumps(payload, indent=2))
        if args.save_json:
            Path(args.save_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    if not args.noisy_root or not args.clean_root:
        raise ValueError("Please provide either --noisy-img/--clean-img or both --noisy-root/--clean-root.")

    dataset = RealNoisyPairDataset(
        noisy_root=args.noisy_root,
        clean_root=args.clean_root,
        sigma=args.sigma,
        max_samples=args.max_samples,
        max_side=args.max_side,
    )

    print(f"Matched pairs: {len(dataset)}")
    for noisy_path, clean_path in dataset.pairs[: max(0, args.show_pairs)]:
        print(f"  {noisy_path.name}  <->  {clean_path.name}")

    if args.save_manifest:
        dataset.write_manifest(args.save_manifest)
        print(f"Saved pair manifest to: {args.save_manifest}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    pred_psnr_vals: list[float] = []
    pred_ssim_vals: list[float] = []
    noisy_psnr_vals: list[float] = []
    noisy_ssim_vals: list[float] = []
    psnr_gain_vals: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating real-noise pairs", ncols=120):
            clean = batch["clean"].to(device)
            noisy = batch["noisy"].to(device)
            sigma_map = batch["sigma_map"].to(device)

            pred_cpu = infer_tiled(
                model,
                noisy.squeeze(0).detach().cpu(),
                sigma=args.sigma,
                device=device,
                tile_size=args.tile_size,
                tile_halo=args.tile_halo,
            )
            pred = pred_cpu.unsqueeze(0).to(device)

            noisy_psnr = float(psnr(noisy, clean))
            noisy_ssim = float(ssim(noisy, clean))
            pred_psnr = float(psnr(pred, clean))
            pred_ssim = float(ssim(pred, clean))

            noisy_psnr_vals.append(noisy_psnr)
            noisy_ssim_vals.append(noisy_ssim)
            pred_psnr_vals.append(pred_psnr)
            pred_ssim_vals.append(pred_ssim)
            psnr_gain_vals.append(pred_psnr - noisy_psnr)

    result = {
        "sigma": args.sigma,
        "num_samples": len(pred_psnr_vals),
        "psnr_noisy_mean": float(np.mean(noisy_psnr_vals)),
        "ssim_noisy_mean": float(np.mean(noisy_ssim_vals)),
        "psnr_mean": float(np.mean(pred_psnr_vals)),
        "ssim_mean": float(np.mean(pred_ssim_vals)),
        "psnr_gain_mean": float(np.mean(psnr_gain_vals)),
        "max_side": args.max_side,
        "tile_size": args.tile_size,
        "tile_halo": args.tile_halo,
    }
    print(json.dumps(result, indent=2))
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
