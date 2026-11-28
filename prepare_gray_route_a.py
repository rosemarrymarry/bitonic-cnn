from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def seed_from_key(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def list_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )


def export_route_a(
    source_root: Path,
    out_root: Path,
    dataset_name: str,
    sigma: float,
    seed: int,
    overwrite: bool,
) -> int:
    files = list_images(source_root)
    if not files:
        raise FileNotFoundError(f"No images found under: {source_root}")

    sigma_dir = out_root / dataset_name / f"sigma{int(sigma)}"
    clean_out = sigma_dir / "clean"
    noisy_out = sigma_dir / "noisy"
    clean_out.mkdir(parents=True, exist_ok=True)
    noisy_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for src in files:
        rel = src.relative_to(source_root).with_suffix(".png")
        clean_target = clean_out / rel
        noisy_target = noisy_out / rel
        clean_target.parent.mkdir(parents=True, exist_ok=True)
        noisy_target.parent.mkdir(parents=True, exist_ok=True)

        if (
            not overwrite
            and clean_target.exists()
            and noisy_target.exists()
        ):
            continue

        gray_img = Image.open(src).convert("L")
        gray_img.save(clean_target, format="PNG")

        gray = np.asarray(gray_img, dtype=np.float32)[..., None] / 255.0
        gray_rgb = np.repeat(gray, 3, axis=2)

        key = f"{dataset_name}/{rel.as_posix()}/routeA_sigma{int(sigma)}"
        rng = np.random.default_rng(seed_from_key(seed, key))
        noise = rng.normal(
            loc=0.0,
            scale=float(sigma) / 255.0,
            size=gray_rgb.shape,
        ).astype(np.float32)
        noisy = np.clip(gray_rgb + noise, 0.0, 1.0)
        noisy_uint8 = (noisy * 255.0).round().astype(np.uint8)

        Image.fromarray(noisy_uint8).save(noisy_target, format="PNG")
        written += 1

    print(f"dataset={dataset_name} sigma={int(sigma)} written={written}")
    print(f"clean={clean_out}")
    print(f"noisy={noisy_out}")
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build route-A grayscale evaluation data: grayscale clean targets "
            "and RGB independent-noise inputs for an RGB-trained denoiser."
        )
    )
    parser.add_argument("--source-root", type=str, required=True, help="Clean image folder, color or grayscale")
    parser.add_argument("--out-root", type=str, required=True, help="Destination benchmark root")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name under out-root")
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_route_a(
        source_root=Path(args.source_root),
        out_root=Path(args.out_root),
        dataset_name=args.dataset_name,
        sigma=args.sigma,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
