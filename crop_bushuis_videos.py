#!/usr/bin/env python3
"""Centre-square-crop Bushuis MP4 files and write corrected MP4s.

The Bushuis recordings are 1440×1252 (nearly square). The signer is centred
in the frame. This script trims the wider dimension to produce a square output
that feeds cleanly into extract_logos_features.py (which resizes directly to
224×224 without any further cropping).

Usage:
    python crop_bushuis_videos.py \
        --input_dir   /scratch-shared/psobecki/Bushuis/videos \
        --output_dir  /scratch-shared/psobecki/Bushuis/videos_cropped \
        --workers     8
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _process_one(args):
    in_path, out_path, overwrite = args
    if not overwrite and os.path.exists(out_path):
        return out_path, 'skip'
    try:
        import cv2
        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            return out_path, 'error: no frames'

        H, W = frames[0].shape[:2]
        sq = min(W, H)
        x0 = (W - sq) // 2
        y0 = (H - sq) // 2

        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (sq, sq),
        )
        for frame in frames:
            writer.write(frame[y0:y0 + sq, x0:x0 + sq])
        writer.release()
        return out_path, f'ok {W}x{H} → {sq}x{sq}'
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

    video_files = sorted(
        p for p in Path(args.input_dir).iterdir()
        if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}
    )
    print(f'Found {len(video_files)} video files')

    work = [
        (str(f), os.path.join(args.output_dir, f.stem + '.mp4'), args.overwrite)
        for f in video_files
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
