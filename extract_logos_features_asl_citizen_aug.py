"""Extract MViTv2-S features from augmented ASL-Citizen frame sequences.

Handles the nested structure of pre-extracted JPEG frame directories:
  augmented_frames/{video_id}-{WORD}/{aug_name}/0000.jpg ...
  e.g. augmented_frames/4758969817179144-MARCH/glasses/0000.jpg

Output feature naming:
  asl_citizen_{video_id}_{aug_name}.npy
  e.g. asl_citizen_4758969817179144-MARCH_glasses.npy

This matches what SignCLIPVideoCSVMetaProcessor expects when train.csv contains:
  Video file = 4758969817179144-MARCH_glasses.mp4

Must be run in the logos conda environment with mmaction2 installed.
Run from the mmaction2 directory where the Logos fork has been applied.

Usage:
    python extract_logos_features_asl_citizen_aug.py \\
        --aug_dir     /scratch-shared/psobecki/ASL_Citizen/augmented_frames \\
        --output_dir  /home/psobecki/ASL_Citizen/logos_features \\
        --checkpoint  data/model/logos_autsl_wlasl_model.pth
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


CLIP_LEN       = 32
FRAME_INTERVAL = 2
CLIP_STRIDE    = 32
RESIZE         = 300
INPUT_SIZE     = 224
MEAN = np.array([140.99762122, 129.92701646, 125.25081198], dtype=np.float32)
STD  = np.array([62.07248248,  62.94645644,  61.42221137],  dtype=np.float32)

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def _preprocess_worker(args):
    frame_dir, out_path, overwrite, frame_interval = args

    if not overwrite and os.path.exists(out_path):
        return out_path, None, 'skip'

    try:
        import cv2
        from pathlib import Path as _Path
        frame_paths = sorted(_Path(frame_dir).glob('*.jpg'))
        frames = []
        for fp in frame_paths:
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frames:
            return out_path, None, 'error: no frames decoded'
    except Exception as e:
        return out_path, None, f'error: {e}'

    try:
        import cv2
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
        T = len(processed)
        effective_span = CLIP_LEN * frame_interval

        if T < effective_span:
            starts = [0]
        else:
            starts = list(range(0, T - effective_span + 1, CLIP_STRIDE)) or [0]
            if starts[-1] + effective_span < T:
                starts.append(T - effective_span)

        clips = []
        for start in starts:
            indices = [min(start + i * frame_interval, T - 1) for i in range(CLIP_LEN)]
            clip = processed[indices].transpose(1, 0, 2, 3)
            clips.append(clip)

        return out_path, np.stack(clips), 'ok'
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
    backbone_state = {k[len('backbone.'):]: v for k, v in state.items() if k.startswith('backbone.')}
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    backbone_missing = [k for k in missing if 'head' not in k]
    if backbone_missing:
        print(f'WARNING: {len(backbone_missing)} backbone keys missing')
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


def build_work_list(aug_dir: Path, output_dir: Path, overwrite: bool, frame_interval: int):
    """augmented_frames/{video_id}/{aug_name}/*.jpg → asl_citizen_{video_id}_{aug_name}.npy"""
    work = []
    for video_id_dir in sorted(aug_dir.iterdir()):
        if not video_id_dir.is_dir():
            continue
        video_id = video_id_dir.name
        for aug_dir_entry in sorted(video_id_dir.iterdir()):
            if not aug_dir_entry.is_dir():
                continue
            if not any(aug_dir_entry.glob('*.jpg')):
                continue
            aug_name = aug_dir_entry.name   # e.g. "glasses"
            key = f'asl_citizen_{video_id}_{aug_name}'
            out_path = output_dir / f'{key}.npy'
            work.append((str(aug_dir_entry), str(out_path), overwrite, frame_interval))
    return work


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_dir', required=True,
                        help='augmented_frames/ directory (contains {video_id}/{aug_name}/*.jpg)')
    parser.add_argument('--output_dir', required=True,
                        help='Where to write asl_citizen_*.npy feature files')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--prefetch', type=int, default=None)
    parser.add_argument('--frame_interval', type=int, default=2,
                        help='Temporal stride between sampled frames (default 2). '
                             'Use 1 for augmentations generated with --stride 2.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    aug_dir = Path(args.aug_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefetch = args.prefetch or args.workers * 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading backbone from {args.checkpoint} ...')
    backbone = load_backbone(args.checkpoint, device)

    work = build_work_list(aug_dir, output_dir, args.overwrite, args.frame_interval)
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
            futures = {executor.submit(_preprocess_worker, item): item for item in chunk}
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
    print(f'\nExtracted: {done}  |  Skipped: {skipped}  |  Errors: {errors}')


if __name__ == '__main__':
    main()
