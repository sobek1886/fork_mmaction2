#!/usr/bin/env python3
"""Crop black borders from piotrAnims MKV files and write corrected MP4s.

Detects the content bounding box for each video by taking pixel-wise max
brightness across sampled frames, then re-encodes with that crop applied.
Output filenames are identical to the input stems (used by
extract_logos_features.py to build the .npy names).

Usage:
    python crop_piotr_videos.py \
        --input_dir   /scratch-shared/psobecki/piotrAnims/mkv \
        --output_dir  /scratch-shared/psobecki/piotrAnims/mkv_cropped \
        --workers     8
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _detect_crop(frames, threshold: int = 15):
    """Return (x, y, w, h) content bbox from pixel-wise max across sampled frames."""
    import cv2
    H, W = frames[0].shape[:2]
    max_gray = np.zeros((H, W), dtype=np.uint8)
    indices = np.linspace(0, len(frames) - 1, min(20, len(frames)), dtype=int)
    for i in indices:
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        np.maximum(max_gray, gray, out=max_gray)
    mask = (max_gray > threshold).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return (0, 0, W, H)
    return cv2.boundingRect(coords)


def _process_one(args):
    in_path, out_path, overwrite = args
    if not overwrite and os.path.exists(out_path):
        return out_path, 'skip'
    try:
        import cv2
        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            return out_path, 'error: no frames'

        x, y, w, h = _detect_crop(frames)

        # Square crop focused on the upper body / signing space
        sq = min(w, h)
        if h >= w:
            # Portrait (LEFT / MIDDLE cameras): top-aligned so head stays in frame
            x_sq, y_sq = x, y
        else:
            # Landscape (RIGHT camera): centre-aligned horizontally
            x_sq = x + (w - sq) // 2
            y_sq = y

        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (sq, sq),
        )
        for frame in frames:
            writer.write(frame[y_sq:y_sq + sq, x_sq:x_sq + sq])
        writer.release()
        return out_path, f'ok content=({x},{y},{w},{h}) sq={sq}x{sq}'
    except Exception as e:
        return out_path, f'error: {e}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input_dir',  required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--workers',    type=int, default=4)
    ap.add_argument('--overwrite',  action='store_true')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mkv_files = sorted(Path(args.input_dir).glob('*.mkv'))
    print(f'Found {len(mkv_files)} MKV files')

    work = [
        (str(f), os.path.join(args.output_dir, f.stem + '.mp4'), args.overwrite)
        for f in mkv_files
    ]

    done = skipped = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_one, item): item for item in work}
        for fut in tqdm(as_completed(futures), total=len(work), desc='Cropping'):
            out_path, status = fut.result()
            if status == 'skip':
                skipped += 1
            elif status.startswith('ok'):
                done += 1
            else:
                tqdm.write(f'  WARN {out_path}: {status}')
                errors += 1

    print(f'\nDone: {done}  Skipped: {skipped}  Errors: {errors}')
    print(f'Output: {args.output_dir}')


if __name__ == '__main__':
    main()
