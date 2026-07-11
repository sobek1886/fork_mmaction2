"""Extract MViTv2-S features from Flux-augmented NGT frame sequences.

NGT counterpart of extract_logos_features_asl_citizen_aug.py, handling the
dated directory layout produced by yamls/flux-augment-ngt.yaml (nano_banana):
  {flux_root}/{date}/augmented_frames/{video_id}/{aug_name}/0000.jpg ...
  e.g. flux-aug/ngt/2026-03-16/augmented_frames/
         M20260227_5998_260316_1_RIGHT_2026-03-16_12-55-25/signer_swap/0000.jpg

Output feature naming (consumed by make_ngt_singleview_manifest.py):
  ngt_flux_{video_id}_{aug_name}.npy
  e.g. ngt_flux_M20260227_5998_260316_1_RIGHT_2026-03-16_12-55-25_signer_swap.npy

Reuses the preprocessing worker, backbone loader and GPU batching from
extract_logos_features_asl_citizen_aug.py so features are byte-identical in
pipeline to every other Logos extraction in this project.

For the single-view Flux-vs-Unreal experiment only the piotrAnims-session
date batch matters (2026-03-16 — the session whose sign ids match the pair
manifest); restrict with --dates to avoid extracting the other sessions.

Must be run in the logos conda environment with mmaction2 installed.
Run from the mmaction2 directory where the Logos fork has been applied.

Usage:
    python extract_logos_features_ngt_flux.py \\
        --flux_root   /scratch-shared/psobecki/NGT_Flux/flux-aug/ngt \\
        --output_dir  /scratch-shared/psobecki/NGT_Flux/logos_features \\
        --checkpoint  data/model/logos_autsl_wlasl_model.pth \\
        --dates       2026-03-16 \\
        --aug_names   signer_swap skin_mst_diffusion glasses shirt_1
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch

from extract_logos_features_asl_citizen_aug import (
    _preprocess_worker, load_backbone, gpu_inference,
)


def build_work_list(flux_root: Path, output_dir: Path, overwrite: bool,
                    frame_interval: int, dates=None, aug_names=None):
    """{date}/augmented_frames/{video_id}/{aug_name}/*.jpg
       → ngt_flux_{video_id}_{aug_name}.npy"""
    work = []
    for date_dir in sorted(flux_root.iterdir()):
        if not date_dir.is_dir():
            continue
        if dates and date_dir.name not in dates:
            continue
        frames_root = date_dir / 'augmented_frames'
        if not frames_root.is_dir():
            continue
        for video_id_dir in sorted(frames_root.iterdir()):
            if not video_id_dir.is_dir():
                continue
            video_id = video_id_dir.name
            for aug_dir_entry in sorted(video_id_dir.iterdir()):
                if not aug_dir_entry.is_dir():
                    continue
                if not any(aug_dir_entry.glob('*.jpg')):
                    continue
                aug_name = aug_dir_entry.name
                if aug_names and aug_name not in aug_names:
                    continue
                key = f'ngt_flux_{video_id}_{aug_name}'
                out_path = output_dir / f'{key}.npy'
                work.append((str(aug_dir_entry), str(out_path), overwrite,
                             frame_interval))
    return work


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flux_root', required=True,
                        help='flux-aug/ngt root (contains {date}/augmented_frames/)')
    parser.add_argument('--output_dir', required=True,
                        help='Where to write ngt_flux_*.npy feature files')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dates', nargs='+', default=None,
                        help='Only these date subdirs (e.g. 2026-03-16, the '
                             'piotrAnims session). Default: all dates.')
    parser.add_argument('--aug_names', nargs='+', default=None,
                        help='Only these variant subdirs (default: all present)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--prefetch', type=int, default=None)
    parser.add_argument('--frame_interval', type=int, default=2,
                        help='Temporal stride between sampled frames (default 2). '
                             'Use 1 for augmentations generated with --stride 2.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    flux_root = Path(args.flux_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefetch = args.prefetch or args.workers * 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading backbone from {args.checkpoint} ...')
    backbone = load_backbone(args.checkpoint, device)

    work = build_work_list(flux_root, output_dir, args.overwrite,
                           args.frame_interval, args.dates, args.aug_names)
    print(f'Found {len(work)} augmented frame sequences to process')

    skipped = errors = done = 0
    pending_clips = []
    pending_paths = []

    def flush_gpu():
        nonlocal done
        if not pending_clips:
            return
        all_clips = np.concatenate(pending_clips, axis=0)
        all_feats = gpu_inference(backbone, all_clips, device, args.batch_size)
        offset = 0
        for out_path, n in pending_paths:
            np.save(out_path, all_feats[offset:offset + n])
            offset += n
            done += 1
        pending_clips.clear()
        pending_paths.clear()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        pbar = tqdm(total=len(work), desc='Extracting')
        for chunk_start in range(0, len(work), prefetch):
            chunk = work[chunk_start:chunk_start + prefetch]
            futures = {executor.submit(_preprocess_worker, item): item
                       for item in chunk}
            for future in as_completed(futures):
                out_path, clips, status = future.result()
                if status == 'skip':
                    skipped += 1
                elif status == 'ok':
                    pending_clips.append(clips)
                    pending_paths.append((out_path, len(clips)))
                    if sum(c.shape[0] for c in pending_clips) >= args.batch_size:
                        flush_gpu()
                else:
                    tqdm.write(f'  WARN {futures[future][0]}: {status}')
                    errors += 1
                pbar.update(1)
        pbar.close()
        flush_gpu()

    print(f'\nDone: {done} extracted, {skipped} skipped, {errors} errors '
          f'→ {output_dir}')


if __name__ == '__main__':
    main()
