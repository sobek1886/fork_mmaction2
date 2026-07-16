"""Extract MViTv2-S features from ASL Citizen videos using the Logos pre-trained model.

Must be run in the logos conda environment with mmaction2 installed.
Run from the mmaction2 directory where the Logos fork has been applied.

Output:
    One .npy file per video at {output_dir}/{dataset_name}_{video_id}.npy
    Shape: (N_clips, 768) — one 768-dim feature per 64-frame sliding-window clip.

Usage:
    python extract_logos_features.py \\
        --video_dir   /home/psobecki/ASL_Citizen/videos \\
        --output_dir  /home/psobecki/ASL_Citizen/logos_features \\
        --checkpoint  data/model/logos_autsl_wlasl_model.pth

Install prerequisites (logos conda env):
    pip install opencv-python-headless tqdm
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ── Preprocessing constants ───────────────────────────────────────────────────
CLIP_LEN      = 32     # frames fed to MViTv2-S
FRAME_INTERVAL = 2     # sample every 2nd frame → each clip spans 64 consecutive frames
CLIP_STRIDE   = 32     # clips are non-overlapping (stride = CLIP_LEN)
INPUT_SIZE    = 224    # spatial size expected by MViTv2-S
RESIZE        = 300    # logos_native: long side resized to this before pad + crop
PAD_VALUE     = 114    # grey pad value (matches Logos SquarePadding default)
MEAN = np.array([140.99762122, 129.92701646, 125.25081198], dtype=np.float32)
STD  = np.array([62.07248248,  62.94645644,  61.42221137],  dtype=np.float32)


def preprocess_frame(frame_rgb, mode):
    """Resize + normalise one RGB frame -> (3, 224, 224) float32.

    mode='logos_native': resize LONG side -> 300, pad to 300x300 (grey 114),
                         center-crop 224. Aspect-preserving — matches how the Logos
                         model was trained; correct for non-square inputs (ASL-Citizen).
    mode='direct224':    resize directly to 224x224 (legacy; only correct if the frame
                         is already square, else aspect-distorts).
    """
    import cv2
    if mode == "direct224":
        f = cv2.resize(frame_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    elif mode == "logos_native":
        h, w = frame_rgb.shape[:2]
        scale = RESIZE / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        f = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        ph, pw = RESIZE - nh, RESIZE - nw
        f = np.pad(f, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)),
                   mode="constant", constant_values=PAD_VALUE)
        off = (RESIZE - INPUT_SIZE) // 2
        f = f[off:off + INPUT_SIZE, off:off + INPUT_SIZE]
    elif mode == "letterbox224":
        # resize LONG side -> 224, pad short side to 224 (grey). Aspect-preserving AND
        # keeps the whole frame (no crop) — avoids clipping hands at the sides.
        h, w = frame_rgb.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        f = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        ph, pw = INPUT_SIZE - nh, INPUT_SIZE - nw
        f = np.pad(f, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)),
                   mode="constant", constant_values=PAD_VALUE)
    else:
        raise ValueError(f"unknown preproc mode: {mode}")
    f = (f.astype(np.float32) - MEAN) / STD       # (H, W, 3)
    return f.transpose(2, 0, 1)                    # (3, H, W)


# ── Preprocessing (CPU, runs in worker processes) ─────────────────────────────

def _preprocess_worker(args):
    """Decode one video and build MViT clips. Runs in a subprocess (CPU only).

    Returns: (out_path, clips_np, status)
        clips_np: (N, 3, CLIP_LEN, 224, 224) float32, or None on skip/error
        status:   'ok' | 'skip' | 'error: <msg>'
    """
    vpath, out_path, overwrite, preproc, frame_interval = args

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
        T = len(frames)

        # NOTE: the original baseline + endanchor feature sets (logos_features,
        # logos_features_endanchor) were extracted with direct resize-to-224, i.e.
        # preproc='direct224'. ASL-Citizen frames are NOT square, so this aspect-
        # distorts them. Kept here for reference:
        #     processed = []
        #     for frame in frames:
        #         frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE),
        #                            interpolation=cv2.INTER_LINEAR)
        #         frame = (frame.astype(np.float32) - MEAN) / STD  # (H, W, 3)
        #         processed.append(frame.transpose(2, 0, 1))       # (3, H, W)
        #     processed = np.stack(processed)
        # Now switchable via `preproc` (default 'logos_native', aspect-preserving):
        processed = np.stack([preprocess_frame(frame, preproc) for frame in frames])  # (T,3,H,W)

        # Build clips with FRAME_INTERVAL subsampling
        effective_span = CLIP_LEN * FRAME_INTERVAL  # 64 frames

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
    import mmaction.models  # noqa: F401 — triggers @MODELS.register_module() decorators

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
    clips = torch.from_numpy(clips_np).float().to(device)  # (N, 3, T, H, W)
    parts = []
    for i in range(0, len(clips), batch_size):
        feat = backbone(clips[i:i + batch_size])
        # Unwrap nested lists/tuples until we reach a tensor
        while isinstance(feat, (list, tuple)):
            feat = feat[-1]
        if feat.ndim == 5:   # (B, C, T, H, W) → global avg pool
            feat = feat.mean(dim=[2, 3, 4])
        elif feat.ndim == 3: # (B, seq, C) → mean pool over sequence
            feat = feat.mean(dim=1)
        # feat: (B, 768)
        parts.append(feat.cpu().numpy())
    return np.concatenate(parts, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir',    required=True,
                        help='Directory containing .mp4 videos (searched recursively)')
    parser.add_argument('--output_dir',   required=True,
                        help='Directory to save .npy feature files')
    parser.add_argument('--checkpoint',   required=True,
                        help='Path to logos_autsl_wlasl_model.pth')
    parser.add_argument('--dataset_name', default='asl_citizen',
                        help='Prefix for output filenames (default: asl_citizen)')
    parser.add_argument('--workers',      type=int, default=4,
                        help='CPU worker processes for video decoding')
    parser.add_argument('--batch_size',   type=int, default=64,
                        help='Clips per GPU forward pass')
    parser.add_argument('--preproc', choices=['logos_native', 'direct224', 'letterbox224'],
                        default='logos_native',
                        help='Frame preprocessing. direct224 = resize whole frame to 224x224 '
                             '(keeps all content, mild stretch; best in eval). letterbox224 = '
                             'resize long side to 224 + grey-pad (keeps all, no distortion, '
                             'smaller signer). logos_native = resize-300+pad+center-crop-224 '
                             '(aspect-preserving but CLIPS hands at the sides on 4:3 frames).')
    parser.add_argument('--prefetch',     type=int, default=None,
                        help='Videos to prefetch (default: workers * 8)')
    parser.add_argument('--overwrite',    action='store_true')
    args = parser.parse_args()

    prefetch = args.prefetch or args.workers * 8
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'CPU workers: {args.workers}  |  GPU batch: {args.batch_size}  |  Prefetch: {prefetch}')
    print(f'Preprocessing: {args.preproc}')

    print(f'Loading backbone from {args.checkpoint} ...')
    backbone = load_backbone(args.checkpoint, device)
    print('Model loaded.')

    video_files = [
        str(p) for p in Path(args.video_dir).rglob('*')
        if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    ]
    video_files = sorted(video_files)
    print(f'Found {len(video_files)} videos')

    work = []
    for vpath in video_files:
        vid = Path(vpath).stem
        out_path = os.path.join(args.output_dir, f'{args.dataset_name}_{vid}.npy')
        work.append((vpath, out_path, args.overwrite, args.preproc))

    skipped = errors = done = 0
    pending_clips = []   # list of (N_i, 3, T, H, W) arrays
    pending_paths = []   # list of (out_path, N_i)

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
    print(f'Output:  {args.output_dir}/{args.dataset_name}_<video_id>.npy')
    print(f'Shape:   (N_clips, 768)')


if __name__ == '__main__':
    main()
