#!/usr/bin/env python3
"""
This is a script that segments coronal holes (CHs) and active regions (ARs) using 
Basic Computer Vison Operations (BCVO)

It can be called as follows:

python bcvo.py --input_npy sample_193.npy --output_dir out_sample

The input can be an SDO/AIA sample 193Å (or 171Å) (from SDOMLv2), 512x512:

The script consists of
- 5-class Multi-Otsu
- CH = class 1 (darkest), AR = class 5 (brightest)
- Morphological opening + closing + small-object removal
- Saves CH/AR masks (.npy) and an overlay PNG
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_multiotsu
from skimage.morphology import disk, opening, closing, remove_small_objects
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity


def robust_normalize(img: np.ndarray, p_low=1.0, p_high=99.5) -> np.ndarray:
    """Percentile clip + rescale to [0, 1] for stable thresholding."""
    finite = np.isfinite(img)
    x = img[finite]
    lo, hi = np.percentile(x, [p_low, p_high])
    imgc = np.clip(img, lo, hi)
    img01 = rescale_intensity(imgc, in_range=(lo, hi), out_range=(0.0, 1.0))
    img01[~finite] = 0.0
    return img01.astype(np.float32)


def keep_largest_k(mask: np.ndarray, k: int) -> np.ndarray:
    """Keep largest k connected components in a binary mask."""
    if k <= 0:
        return mask
    lab = label(mask)
    props = regionprops(lab)
    if not props:
        return np.zeros_like(mask, dtype=bool)
    props_sorted = sorted(props, key=lambda r: r.area, reverse=True)
    keep_labels = {p.label for p in props_sorted[:k]}
    return np.isin(lab, list(keep_labels))


def postprocess(mask: np.ndarray, open_r: int, close_r: int, min_area: int, keep_k: int) -> np.ndarray:
    m = mask.astype(bool)

    # 1) Opening: remove small speckles / thin noise
    if open_r > 0:
        m = opening(m, disk(open_r))

    # 2) Closing: connect nearby regions / fill small holes
    if close_r > 0:
        m = closing(m, disk(close_r))

    # 3) Remove small objects (false positives)
    if min_area > 0:
        m = remove_small_objects(m, min_size=min_area)

    # 4) Optionally keep only the largest k components
    if keep_k > 0:
        m = keep_largest_k(m, keep_k)

    return m.astype(bool)


def save_overlay(img01: np.ndarray, ch: np.ndarray, ar: np.ndarray, out_png: Path) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(img01, cmap="gray", origin="lower")

    # CH blue, AR red
    ch_rgba = np.zeros((*img01.shape, 4), dtype=np.float32)
    ch_rgba[..., 2] = 1.0
    ch_rgba[..., 3] = 0.35 * ch.astype(np.float32)

    ar_rgba = np.zeros((*img01.shape, 4), dtype=np.float32)
    ar_rgba[..., 0] = 1.0
    ar_rgba[..., 3] = 0.35 * ar.astype(np.float32)

    plt.imshow(ch_rgba, origin="lower")
    plt.imshow(ar_rgba, origin="lower")
    plt.title("AIA 193Å (512x512) — CH (blue), AR (red) — 5-class Multi-Otsu")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npy", type=str, required=True,
                    help="Path to a single 512x512 image stored as .npy (float or uint).")
    ap.add_argument("--output_dir", type=str, default="out_otsu_193_512")
    ap.add_argument("--classes", type=int, default=5)

    # Normalization
    ap.add_argument("--p_low", type=float, default=1.0)
    ap.add_argument("--p_high", type=float, default=99.5)

    # Tuned defaults for AIA 193Å @ 512x512
    ap.add_argument("--ch_open", type=int, default=2)
    ap.add_argument("--ch_close", type=int, default=9)
    ap.add_argument("--ch_min_area", type=int, default=1200)
    ap.add_argument("--ch_keep_k", type=int, default=0)

    ap.add_argument("--ar_open", type=int, default=2)
    ap.add_argument("--ar_close", type=int, default=6)
    ap.add_argument("--ar_min_area", type=int, default=400)
    ap.add_argument("--ar_keep_k", type=int, default=0)

    args = ap.parse_args()

    in_path = Path(args.input_npy)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = np.load(in_path).astype(np.float32)
    if img.shape != (512, 512):
        raise ValueError(f"Expected (512,512). Got {img.shape}.")

    img01 = robust_normalize(img, p_low=args.p_low, p_high=args.p_high)

    # 5-class Multi-Otsu
    thresholds = threshold_multiotsu(img01, classes=args.classes)
    regions = np.digitize(img01, bins=thresholds)  # 0..classes-1

    # Class mapping (typical for 193Å):
    #  - CH: darkest -> 0
    #  - AR: brightest -> 4 (for 5 classes)
    ch_raw = (regions == 0)
    ar_raw = (regions == (args.classes - 1))

    # If you ever want the opposite (AR=darkest, CH=brightest), swap:
    # ar_raw = (regions == 0)
    # ch_raw = (regions == (args.classes - 1))

    ch = postprocess(ch_raw, args.ch_open, args.ch_close, args.ch_min_area, args.ch_keep_k)
    ar = postprocess(ar_raw, args.ar_open, args.ar_close, args.ar_min_area, args.ar_keep_k)

    # Save results
    np.save(out_dir / "mask_CH.npy", ch.astype(np.uint8))
    np.save(out_dir / "mask_AR.npy", ar.astype(np.uint8))
    np.savetxt(out_dir / "thresholds.txt", thresholds, fmt="%.6f")

    save_overlay(img01, ch, ar, out_dir / "overlay.png")

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Input: {in_path}\n")
        f.write(f"Image shape: {img.shape}\n")
        f.write(f"Classes: {args.classes}\n")
        f.write(f"p_low/p_high: {args.p_low}/{args.p_high}\n")
        f.write(f"Thresholds: {thresholds.tolist()}\n")
        f.write(f"CH pixels: {int(ch.sum())}\n")
        f.write(f"AR pixels: {int(ar.sum())}\n")
        f.write(f"CH params: open={args.ch_open}, close={args.ch_close}, min_area={args.ch_min_area}, keep_k={args.ch_keep_k}\n")
        f.write(f"AR params: open={args.ar_open}, close={args.ar_close}, min_area={args.ar_min_area}, keep_k={args.ar_keep_k}\n")

    print(f"Saved to: {out_dir.resolve()}")
    print(" - mask_CH.npy, mask_AR.npy, overlay.png, thresholds.txt, summary.txt")


if __name__ == "__main__":
    main()
