#!/usr/bin/env python3
"""Crop black borders from NGT OBS-exported MKV files and write square MP4s.

The RIGHT camera produces 1920x1080 frames where content sits in the top-left
1280x720 region. The signer is positioned right-of-centre, so the square crop
is right-aligned within the content width to keep hands in frame.

Handles the dated subfolder structure:
    input_dir/{date}/*_RIGHT_*.mkv  →  output_dir/{date}/{stem}.mp4

Usage:
    python crop_ngt_videos.py \
        --input_dir   /Users/piotr/Projects/Thesis/Data/obs_right_export \
        --output_dir  /Users/piotr/Projects/Thesis/Data/obs_right_export_cropped \
        --workers     4
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _detect_crop(frames, threshold: int = 15):
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
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
        sq = min(w, h)

        # RIGHT camera: landscape content, signer is right-of-centre.
        # Right-align the square crop to keep hands in frame.
        x_sq = x + (w - sq)
        y_sq = y

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (sq, sq),
        )
        for frame in frames:
            writer.write(frame[y_sq:y_sq + sq, x_sq:x_sq + sq])
        writer.release()
        return out_path, f'ok content=({x},{y},{w},{h}) sq={sq}x{sq} x_sq={x_sq}'
    except Exception as e:
        return out_path, f'error: {e}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input_dir',  required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--workers',    type=int, default=4)
    ap.add_argument('--overwrite',  action='store_true')
    args = ap.parse_args()

    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)

    mkv_files = sorted(input_root.rglob('*_RIGHT_*.mkv'))
    print(f'Found {len(mkv_files)} RIGHT MKV files')

    work = []
    for f in mkv_files:
        rel = f.relative_to(input_root)
        out_path = output_root / rel.parent / (f.stem + '.mp4')
        work.append((str(f), str(out_path), args.overwrite))

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
    print(f'Output: {output_root}')


if __name__ == '__main__':
    main()
