"""Extract MViTv2-S features from WLASL-100 videos using the Logos pre-trained model.

Handles two video sources under a single WLASL root:
  augmented_videos/{video_id}/{variant}.mp4  →  wlasl100_{video_id}_{variant}.npy
  test/{video_id}.mp4                        →  wlasl100_test_{video_id}.npy

Empty augmented sub-folders are skipped transparently; already-extracted .npy
files are also skipped, so the job can be re-submitted after rsyncing the
remaining half of augmented_videos and it will pick up exactly where it left off.

Must be run in the logos conda environment with mmaction2 installed.
Run from the mmaction2 directory where the Logos fork has been applied.

Usage:
    python extract_logos_features_wlasl.py \\
        --wlasl_dir   /scratch-shared/psobecki/wlasl100 \\
        --output_dir  /scratch-shared/psobecki/wlasl100/logos_features \\
        --checkpoint  data/model/logos_autsl_wlasl_model.pth
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ── Preprocessing constants (from Logos val_pipeline) ────────────────────────
CLIP_LEN       = 32
FRAME_INTERVAL = 2
CLIP_STRIDE    = 32
RESIZE         = 300
INPUT_SIZE     = 224
MEAN = np.array([140.99762122, 129.92701646, 125.25081198], dtype=np.float32)
STD  = np.array([62.07248248,  62.94645644,  61.42221137],  dtype=np.float32)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def _preprocess_worker(args):
    vpath, out_path, overwrite = args

    if not overwrite and os.path.exists(out_path):
        return out_path, None, 'skip'

    try:
        import cv2
        cap = cv2.VideoCapture(vpath)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            return out_path, None, 'error: no frames decoded'
    except Exception as e:
        return out_path, None, f'error: {e}'

    try:
        import cv2
        T = len(frames)
        processed = []
        for frame in frames:
            h, w = frame.shape[:2]
            scale = RESIZE / min(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            pad_h = max(0, RESIZE - nh)
            pad_w = max(0, RESIZE - nw)
            frame = np.pad(frame,
                           ((pad_h // 2, pad_h - pad_h // 2),
                            (pad_w // 2, pad_w - pad_w // 2),
                            (0, 0)),
                           mode='constant')
            y0 = (frame.shape[0] - INPUT_SIZE) // 2
            x0 = (frame.shape[1] - INPUT_SIZE) // 2
            frame = frame[y0:y0 + INPUT_SIZE, x0:x0 + INPUT_SIZE]
            frame = (frame.astype(np.float32) - MEAN) / STD
            processed.append(frame.transpose(2, 0, 1))

        processed = np.stack(processed)
        effective_span = CLIP_LEN * FRAME_INTERVAL

        if T < effective_span:
            starts = [0]
        else:
            starts = list(range(0, T - effective_span + 1, CLIP_STRIDE))
            if not starts:
                starts = [0]
            elif starts[-1] + effective_span < T:
                starts.append(T - effective_span)

        clips = []
        for start in starts:
            indices = [min(start + i * FRAME_INTERVAL, T - 1) for i in range(CLIP_LEN)]
            clip = processed[indices]
            clip = clip.transpose(1, 0, 2, 3)
            clips.append(clip)

        clips_np = np.stack(clips)
        return out_path, clips_np, 'ok'

    except Exception as e:
        return out_path, None, f'error: {e}'


def load_backbone(checkpoint_path, device):
    from mmaction.registry import MODELS

    backbone = MODELS.build(dict(
        type='MViT',
        arch='small',
        drop_path_rate=0.1,
        dim_mul_in_attention=False,
    ))

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)

    backbone_state = {
        k[len('backbone.'):]: v
        for k, v in state.items()
        if k.startswith('backbone.')
    }
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    backbone_missing = [k for k in missing if 'head' not in k]
    if backbone_missing:
        print(f'WARNING: {len(backbone_missing)} backbone keys missing, e.g. {backbone_missing[:3]}')
    if unexpected:
        print(f'WARNING: {len(unexpected)} unexpected keys ignored')

    return backbone.eval().to(device)


@torch.no_grad()
def gpu_inference(backbone, clips_np, device, batch_size):
    clips = torch.from_numpy(clips_np).float().to(device)
    parts = []
    for i in range(0, len(clips), batch_size):
        feat = backbone(clips[i:i + batch_size])
        while isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.ndim == 5:
            feat = feat.mean(dim=[2, 3, 4])
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)
        parts.append(feat.cpu().numpy())
    return np.concatenate(parts, axis=0)


def build_work_list(wlasl_dir: Path, output_dir: Path, overwrite: bool):
    """Return list of (vpath, out_path, overwrite) tuples for all videos."""
    work = []

    # Augmented videos: wlasl_dir/augmented_videos/{video_id}/{variant}.mp4
    aug_root = wlasl_dir / 'augmented_videos'
    if aug_root.is_dir():
        for video_id_dir in sorted(aug_root.iterdir()):
            if not video_id_dir.is_dir():
                continue
            video_id = video_id_dir.name
            for vpath in sorted(video_id_dir.iterdir()):
                if vpath.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                key = f'wlasl100_{video_id}_{vpath.stem}'
                out_path = output_dir / f'{key}.npy'
                work.append((str(vpath), str(out_path), overwrite))

    # Flat split directories: wlasl_dir/{split}/{video_id}.mp4
    for split in ('val', 'test'):
        split_root = wlasl_dir / split
        if not split_root.is_dir():
            continue
        for vpath in sorted(split_root.iterdir()):
            if vpath.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            key = f'wlasl100_{split}_{vpath.stem}'
            out_path = output_dir / f'{key}.npy'
            work.append((str(vpath), str(out_path), overwrite))

    return work


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wlasl_dir',  required=True,
                        help='Root WLASL directory containing augmented_videos/ and test/')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save .npy feature files (in scratch-shared)')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to logos_autsl_wlasl_model.pth')
    parser.add_argument('--workers',    type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--prefetch',   type=int, default=None)
    parser.add_argument('--overwrite',  action='store_true')
    args = parser.parse_args()

    wlasl_dir  = Path(args.wlasl_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefetch = args.prefetch or args.workers * 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'CPU workers: {args.workers}  |  GPU batch: {args.batch_size}  |  Prefetch: {prefetch}')

    print(f'Loading backbone from {args.checkpoint} ...')
    backbone = load_backbone(args.checkpoint, device)
    print('Model loaded.')

    work = build_work_list(wlasl_dir, output_dir, args.overwrite)
    print(f'Found {len(work)} videos total (already-done will be skipped)')

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
            futures = {executor.submit(_preprocess_worker, item): item for item in chunk}

            for future in as_completed(futures):
                out_path, clips, status = future.result()

                if status == 'skip':
                    skipped += 1
                elif status == 'ok':
                    pending_clips.append(clips)
                    pending_paths.append((out_path, len(clips)))
                    total_pending = sum(c.shape[0] for c in pending_clips)
                    if total_pending >= args.batch_size:
                        flush_gpu()
                else:
                    tqdm.write(f'  WARN {futures[future][0]}: {status}')
                    errors += 1

                pbar.update(1)

        pbar.close()

    flush_gpu()

    print(f'\nExtracted: {done}  |  Skipped: {skipped}  |  Errors: {errors}')
    print(f'Output: {output_dir}')
    print(f'Shape per file: (N_clips, 768)')


if __name__ == '__main__':
    main()
