#!/usr/bin/env python3
"""
Detect the non-black content bounding box in piotrAnims MKV videos by
sampling frames and finding where the solid black border ends.

Run locally on the downloaded examples to find the crop rectangle per
camera angle, then hard-code those values into extract_logos_features.py.

Usage:
    python detect_content_crop.py /Users/piotr/Desktop/ngt_examples
"""

import sys
from pathlib import Path
import cv2
import numpy as np


def detect_crop(video_path: str, n_samples: int = 20, threshold: int = 15) -> tuple:
    """Return (x, y, w, h) content bounding box by taking pixel-wise max
    brightness across n_samples evenly-spaced frames."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # accumulate per-pixel max brightness across sampled frames
    max_gray = np.zeros((H, W), dtype=np.uint8)
    indices = np.linspace(0, total - 1, min(n_samples, total), dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        np.maximum(max_gray, gray, out=max_gray)
    cap.release()

    mask = (max_gray > threshold).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return (0, 0, W, H)
    x, y, w, h = cv2.boundingRect(coords)
    return (x, y, w, h)


def main():
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    mkv_files = sorted(directory.rglob("*_piotr_*.mkv"))

    if not mkv_files:
        print("No piotrAnims MKV files found.")
        return

    print(f"{'File':<55} {'Orig WxH':>12}  {'Crop x,y,w,h':>20}  {'Content %':>10}")
    print("-" * 105)
    for f in mkv_files:
        cap = cv2.VideoCapture(str(f))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        x, y, w, h = detect_crop(str(f))
        pct = 100 * w * h / (W * H)
        print(f"{f.name:<55} {W}x{H:>4}  {x:>4},{y:>4},{w:>4},{h:>4}  {pct:>9.1f}%")


if __name__ == "__main__":
    main()
