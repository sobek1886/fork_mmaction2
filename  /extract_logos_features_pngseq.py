"""Extract MViTv2-S features from PNG frame-sequence directories using the
Logos pre-trained model.

Each sub-directory of --render_dir is treated as one "video": all *.png files
inside it are read in sorted order (Unreal Engine zero-padded frame names sort
correctly lexicographically).

Must be run in the logos conda environment with mmaction2 installed.
Run from the mmaction2 directory where the Logos fork has been applied.

Output:
    One .npy file per sequence at {output_dir}/{dataset_name}_{seq_name}.npy
    Shape: (N_clips, 768) — one 768-dim feature per 64-frame sliding-window clip.

Usage:
    python extract_logos_features_pngseq.py \\
        --render_dir  /scratch-shared/psobecki/Custom_from_FBX/Saved/Renders \\
        --output_dir  /home/psobecki/NGT_Aug/logos_features \\
        --checkpoint  data/model/logos_autsl_wlasl_model.pth

Install prerequisites (logos conda env):
    pip install opencv-python-headless tqdm
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ── Preprocessing constants (from Logos val_pipeline) ────────────────────────
CLIP_LEN       = 32    # frames fed to MViTv2-S
FRAME_INTERVAL = 2     # sample every 2nd frame → each clip spans 64 consecutive frames
CLIP_STRIDE    = 32    # non-overlapping clips
RESIZE         = 300   # resize short side to this
INPUT_SIZE     = 224   # final spatial crop
MEAN = np.array([140.99762122, 129.92701646, 125.25081198], dtype=np.float32)
STD  = np.array([62.07248248,  62.94645644,  61.42221137],  dtype=np.float32)


# ── Preprocessing (CPU, runs in worker processes) ─────────────────────────────

def _preprocess_worker(args):
    """Read one PNG-sequence directory and build MViT clips. CPU-only subprocess.

    Returns: (out_path, clips_np, status)
        clips_np: (N, 3, CLIP_LEN, 224, 224) float32, or None on skip/error
        status:   'ok' | 'skip' | 'error: <msg>'
    """
    seq_dir, out_path, overwrite = args

    if not overwrite and os.path.exists(out_path):
        return out_path, None, 'skip'

    try:
        import cv2
        png_files = sorted(Path(seq_dir).glob('*.png'))
        if not png_files:
            return out_path, None, 'error: no PNG files found'

        frames = []
        for p in png_files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not frames:
            return out_path, None, 'error: all PNG reads failed'

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
            # Square-pad to RESIZE × RESIZE
            pad_h = max(0, RESIZE - nh)
            pad_w = max(0, RESIZE - nw)
            frame = np.pad(frame,
                           ((pad_h // 2, pad_h - pad_h // 2),
                            (pad_w // 2, pad_w - pad_w // 2),
                            (0, 0)),
                           mode='constant')
            # Center crop to INPUT_SIZE × INPUT_SIZE
            y0 = (frame.shape[0] - INPUT_SIZE) // 2
            x0 = (frame.shape[1] - INPUT_SIZE) // 2
            frame = frame[y0:y0 + INPUT_SIZE, x0:x0 + INPUT_SIZE]
            # Normalise
            frame = (frame.astype(np.float32) - MEAN) / STD   # (H, W, 3)
            processed.append(frame.transpose(2, 0, 1))         # (3, H, W)

        processed = np.stack(processed)  # (T, 3, H, W)

        effective_span = CLIP_LEN * FRAME_INTERVAL  # 64 frames

        if T < effective_span:
            starts = [0]
        else:
            starts = list(range(0, T - effective_span + 1, CLIP_STRIDE))
            if not starts:
                starts = [0]

        clips = []
        for start in starts:
            indices = [min(start + i * FRAME_INTERVAL, T - 1) for i in range(CLIP_LEN)]
            clip = processed[indices]           # (CLIP_LEN, 3, H, W)
            clip = clip.transpose(1, 0, 2, 3)  # (3, CLIP_LEN, H, W)
            clips.append(clip)

        clips_np = np.stack(clips)  # (N_clips, 3, CLIP_LEN, 224, 224)
        return out_path, clips_np, 'ok'

    except Exception as e:
        return out_path, None, f'error: {e}'


# ── Model loading ─────────────────────────────────────────────────────────────

def load_backbone(checkpoint_path, device):
    """Build MViTv2-S backbone and load weights from Logos checkpoint."""
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


# ── GPU inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def gpu_inference(backbone, clips_np, device, batch_size):
    """clips_np: (N, 3, T, 224, 224)  →  (N, 768) numpy."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render_dir',   required=True,
                        help='Parent directory whose sub-dirs are PNG frame sequences')
    parser.add_argument('--output_dir',   required=True,
                        help='Directory to save .npy feature files')
    parser.add_argument('--checkpoint',   required=True,
                        help='Path to logos_autsl_wlasl_model.pth')
    parser.add_argument('--dataset_name', default='ngt_aug',
                        help='Prefix for output filenames (default: ngt_aug)')
    parser.add_argument('--workers',      type=int, default=4,
                        help='CPU worker processes for frame decoding')
    parser.add_argument('--batch_size',   type=int, default=8,
                        help='Clips per GPU forward pass')
    parser.add_argument('--prefetch',     type=int, default=None,
                        help='Sequences to prefetch (default: workers * 8)')
    parser.add_argument('--overwrite',    action='store_true')
    args = parser.parse_args()

    prefetch = args.prefetch or args.workers * 8
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'CPU workers: {args.workers}  |  GPU batch: {args.batch_size}  |  Prefetch: {prefetch}')

    print(f'Loading backbone from {args.checkpoint} ...')
    backbone = load_backbone(args.checkpoint, device)
    print('Model loaded.')

    # Collect sub-directories that contain at least one PNG
    seq_dirs = sorted(
        d for d in Path(args.render_dir).iterdir()
        if d.is_dir() and any(d.glob('*.png'))
    )
    print(f'Found {len(seq_dirs)} PNG-sequence directories')

    work = []
    for seq_dir in seq_dirs:
        out_path = os.path.join(args.output_dir,
                                f'{args.dataset_name}_{seq_dir.name}.npy')
        work.append((str(seq_dir), out_path, args.overwrite))

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
    print(f'Output:  {args.output_dir}/{args.dataset_name}_<seq_name>.npy')
    print(f'Shape:   (N_clips, 768)')


if __name__ == '__main__':
    main()
