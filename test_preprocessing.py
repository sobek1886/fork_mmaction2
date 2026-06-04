#!/usr/bin/env python3
"""
Visualise the proposed cropping strategy + Logos preprocessing pipeline for
all NGT data types and both train and val/test pipelines.

Uses the two example clips already on the Desktop to generate a pipeline strip
for every video type so that the result can be inspected visually.

Pipeline per strip:
  Original  |  Proposed crop  |  300×300 padded  |  Val 224×224  |  Train 224×224 ×3

Logos preprocessing (matches the actual config val_pipeline / train_pipeline):
  - Resize: scale long side to 300 (fit within 300×300)
  - SquarePadding: pad short side to 300×300 with value 114 (grey)
  - CenterCrop / RandomCrop: 224×224

Usage:
    python test_preprocessing.py

Output:
    /Users/piotr/Desktop/ngt_examples/preprocessing_test/
"""

import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
EX1   = Path('/Users/piotr/Desktop/ngt_examples')
EX2   = EX1 / '2ndexample'
OUT   = EX1 / 'preprocessing_test'
OUT.mkdir(exist_ok=True)

# ── Logos constants ────────────────────────────────────────────────────────────
RESIZE     = 300
INPUT_SIZE = 224
PAD_VALUE  = 114   # grey, matches Logos SquarePadding default

# ── Utilities ──────────────────────────────────────────────────────────────────

def mid_frame(video_path: Path) -> np.ndarray:
    """Return the middle frame of a video as a BGR array."""
    cap = cv2.VideoCapture(str(video_path))
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'Could not read frame from {video_path}')
    return frame  # BGR


# ── Content-detection helpers ──────────────────────────────────────────────────

def detect_black_border(bgr: np.ndarray, threshold: int = 15):
    """Return (x, y, w, h) content bounding box by masking out black regions."""
    gray   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask   = (gray > threshold).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    H, W   = bgr.shape[:2]
    return cv2.boundingRect(coords) if coords is not None else (0, 0, W, H)


def detect_nongreen_bbox(rgb: np.ndarray,
                         h_lo: int = 45, h_hi: int = 85, s_min: int = 80):
    """Return (x, y, w, h) bounding box of non-green-screen content."""
    H, W  = rgb.shape[:2]
    hsv   = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    green = ((hsv[:, :, 0] >= h_lo) & (hsv[:, :, 0] <= h_hi) &
             (hsv[:, :, 1] >= s_min)).astype(np.uint8)
    coords = cv2.findNonZero(1 - green)
    return cv2.boundingRect(coords) if coords is not None else (0, 0, W, H)


def bbox_to_square(x, y, w, h, W, H, margin: int = 20):
    """Expand (x,y,w,h) to a square that fits in a W×H frame."""
    sq  = min(max(w, h) + 2 * margin, W, H)
    cx  = x + w // 2
    cy  = y + h // 2
    x0  = int(np.clip(cx - sq // 2, 0, W - sq))
    y0  = int(np.clip(cy - sq // 2, 0, H - sq))
    return x0, y0, sq


# ── Crop strategies ────────────────────────────────────────────────────────────

def crop_piotr_portrait(bgr: np.ndarray):
    """piotrAnims LEFT/MIDDLE: remove black border → top-aligned square.

    Top-aligned so that the signer's head stays in the crop; the feet fall
    outside the square (irrelevant for sign language).
    Returns (cropped_rgb, bbox_in_original_or_None).
    """
    x, y, w, h = detect_black_border(bgr)
    content = bgr[y:y + h, x:x + w]
    sq = min(w, h)
    # top-left of content = upper-body region
    cropped = content[:sq, :sq]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), (x, y, w, h)


def crop_piotr_landscape(bgr: np.ndarray):
    """piotrAnims RIGHT: remove black border → centre-aligned square."""
    x, y, w, h = detect_black_border(bgr)
    content = bgr[y:y + h, x:x + w]
    sq  = min(w, h)
    x0  = (w - sq) // 2
    cropped = content[:sq, x0:x0 + sq]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), (x, y, w, h)


def crop_palmer(bgr: np.ndarray):
    """palmer/digits: detect non-green bounding box → square with margin."""
    rgb       = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W      = rgb.shape[:2]
    x, y, w, h = detect_nongreen_bbox(rgb)
    x0, y0, sq = bbox_to_square(x, y, w, h, W, H, margin=20)
    return rgb[y0:y0 + sq, x0:x0 + sq], (x, y, w, h)


def crop_bushuis(bgr: np.ndarray):
    """Bushuis: centre square crop (signer is already centred in the frame)."""
    H, W = bgr.shape[:2]
    sq   = min(W, H)
    x0   = (W - sq) // 2
    y0   = (H - sq) // 2
    return cv2.cvtColor(bgr[y0:y0 + sq, x0:x0 + sq], cv2.COLOR_BGR2RGB), None


# ── Logos preprocessing ────────────────────────────────────────────────────────

def logos_resize_pad(rgb: np.ndarray) -> np.ndarray:
    """Resize long side to RESIZE, pad short side to RESIZE×RESIZE with PAD_VALUE.

    Matches Logos val_pipeline:
        Resize(scale=(300,300))   # mmcv.rescale_size → long side = 300
        SquarePadding((300,300))  # BORDER_CONSTANT, value=114
    """
    h, w   = rgb.shape[:2]
    scale  = RESIZE / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((RESIZE, RESIZE, 3), PAD_VALUE, dtype=np.uint8)
    y0      = (RESIZE - nh) // 2
    x0      = (RESIZE - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def val_crop(padded: np.ndarray) -> np.ndarray:
    """Val/test pipeline: CenterCrop(224)."""
    off = (RESIZE - INPUT_SIZE) // 2
    return padded[off:off + INPUT_SIZE, off:off + INPUT_SIZE]


def train_crop(padded: np.ndarray, seed: int = 0) -> np.ndarray:
    """Train pipeline: RandomCrop(224) — one random realisation."""
    rng = np.random.RandomState(seed)
    y0  = rng.randint(0, RESIZE - INPUT_SIZE + 1)
    x0  = rng.randint(0, RESIZE - INPUT_SIZE + 1)
    return padded[y0:y0 + INPUT_SIZE, x0:x0 + INPUT_SIZE]


# ── Visualisation ──────────────────────────────────────────────────────────────
TRAIN_SEEDS = [7, 42, 99]


def save_strip(title: str, bgr: np.ndarray, crop_fn, out_path: Path):
    """Build and save a 7-panel pipeline strip."""
    orig_rgb          = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W              = orig_rgb.shape[:2]
    cropped, bbox     = crop_fn(bgr)
    sq                = cropped.shape[0]
    padded            = logos_resize_pad(cropped)
    val_224           = val_crop(padded)
    train_224s        = [train_crop(padded, s) for s in TRAIN_SEEDS]

    fig, axes = plt.subplots(1, 7, figsize=(28, 5))

    # ── panel 1: original (scaled for display) with bbox overlay ──
    disp_s    = min(400 / W, 400 / H)
    orig_disp = cv2.resize(orig_rgb, (int(W * disp_s), int(H * disp_s)))
    axes[0].imshow(orig_disp)
    axes[0].set_title(f'Original\n{W}×{H}', fontsize=9)
    if bbox is not None:
        x, y, w, h = bbox
        axes[0].add_patch(mpatches.Rectangle(
            (x * disp_s, y * disp_s), w * disp_s, h * disp_s,
            linewidth=2, edgecolor='yellow', facecolor='none'))

    # ── panel 2: proposed crop ──
    axes[1].imshow(cropped)
    axes[1].set_title(f'Proposed crop\n{sq}×{sq}', fontsize=9)

    # ── panel 3: 300×300 padded + crop-rect overlays ──
    axes[2].imshow(padded)
    off = (RESIZE - INPUT_SIZE) // 2
    axes[2].add_patch(mpatches.Rectangle(        # val = green
        (off, off), INPUT_SIZE, INPUT_SIZE,
        linewidth=2, edgecolor='lime', facecolor='none'))
    rng = np.random.RandomState(TRAIN_SEEDS[0])
    ty0 = rng.randint(0, RESIZE - INPUT_SIZE + 1)
    tx0 = rng.randint(0, RESIZE - INPUT_SIZE + 1)
    axes[2].add_patch(mpatches.Rectangle(        # one train example = orange dashed
        (tx0, ty0), INPUT_SIZE, INPUT_SIZE,
        linewidth=2, edgecolor='orange', facecolor='none', linestyle='--'))
    axes[2].set_title('300×300 padded\n'
                      '■ lime = val (centre)   ■ orange = train (random ex.)',
                      fontsize=8)

    # ── panel 4: val output ──
    axes[3].imshow(val_224)
    axes[3].set_title('Val / extraction\n224×224  (centre crop)', fontsize=9)

    # ── panels 5-7: train random crop examples ──
    for i, (tc, s) in enumerate(zip(train_224s, TRAIN_SEEDS)):
        axes[4 + i].imshow(tc)
        axes[4 + i].set_title(f'Train 224×224\nrandom (seed={s})', fontsize=9)

    for ax in axes:
        ax.axis('off')

    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out_path.name}')


# ── Sample definitions ─────────────────────────────────────────────────────────
SAMPLES = [
    # ── piotrAnims (MKV with black borders) ────────────────────────────────────
    dict(video=EX1 / 'M20241106_6046_piotr_middle.mkv', fn=crop_piotr_portrait,
         out='6046_piotr_middle.png',
         title='6046  piotrAnims MIDDLE  –  black-border removal + top-aligned square'),
    dict(video=EX1 / 'M20241106_6046_piotr_left.mkv',   fn=crop_piotr_portrait,
         out='6046_piotr_left.png',
         title='6046  piotrAnims LEFT  –  black-border removal + top-aligned square'),
    dict(video=EX1 / 'M20241106_6046_piotr_right.mkv',  fn=crop_piotr_landscape,
         out='6046_piotr_right.png',
         title='6046  piotrAnims RIGHT  –  black-border removal + centre square'),
    dict(video=EX2 / 'M20241106_6037_piotr_middle.mkv', fn=crop_piotr_portrait,
         out='6037_piotr_middle.png',
         title='6037  piotrAnims MIDDLE  –  black-border removal + top-aligned square'),
    dict(video=EX2 / 'M20241106_6037_piotr_left.mkv',   fn=crop_piotr_portrait,
         out='6037_piotr_left.png',
         title='6037  piotrAnims LEFT  –  black-border removal + top-aligned square'),
    dict(video=EX2 / 'M20241106_6037_piotr_right.mkv',  fn=crop_piotr_landscape,
         out='6037_piotr_right.png',
         title='6037  piotrAnims RIGHT  –  black-border removal + centre square'),

    # ── palmer ─────────────────────────────────────────────────────────────────
    dict(video=EX1 / 'M20241106_6046_palmer_cam2.mp4',  fn=crop_palmer,
         out='6046_palmer_cam2.png',
         title='6046  palmer cam2  –  non-green bbox + square'),
    dict(video=EX1 / 'M20241106_6046_palmer_cam3.mp4',  fn=crop_palmer,
         out='6046_palmer_cam3.png',
         title='6046  palmer cam3  –  non-green bbox + square'),
    dict(video=EX1 / 'M20241106_6046_palmer_cam4.mp4',  fn=crop_palmer,
         out='6046_palmer_cam4.png',
         title='6046  palmer cam4  –  non-green bbox + square'),
    dict(video=EX2 / 'M20241106_6037_palmer_cam3.mp4',  fn=crop_palmer,
         out='6037_palmer_cam3.png',
         title='6037  palmer cam3  –  non-green bbox + square'),

    # ── digits ─────────────────────────────────────────────────────────────────
    dict(video=EX1 / 'M20241106_6046_digits_cam2.mp4',  fn=crop_palmer,
         out='6046_digits_cam2.png',
         title='6046  digits cam2  –  non-green bbox + square'),
    dict(video=EX1 / 'M20241106_6046_digits_cam3.mp4',  fn=crop_palmer,
         out='6046_digits_cam3.png',
         title='6046  digits cam3  –  non-green bbox + square'),
    dict(video=EX1 / 'M20241106_6046_digits_cam4.mp4',  fn=crop_palmer,
         out='6046_digits_cam4.png',
         title='6046  digits cam4  –  non-green bbox + square'),
    dict(video=EX2 / 'M20241106_6037_digits_cam3.mp4',  fn=crop_palmer,
         out='6037_digits_cam3.png',
         title='6037  digits cam3  –  non-green bbox + square'),

    # ── Bushuis ────────────────────────────────────────────────────────────────
    dict(video=EX1 / 'M20241106_6046_bushuis.MP4',       fn=crop_bushuis,
         out='6046_bushuis.png',
         title='6046  Bushuis  –  centre square crop'),
    dict(video=EX2 / 'M20241106_6037_bushuis.MP4',        fn=crop_bushuis,
         out='6037_bushuis.png',
         title='6037  Bushuis  –  centre square crop'),
]


def main():
    print(f'Output directory: {OUT}')
    ok = err = skipped = 0
    for s in SAMPLES:
        vp = s['video']
        if not vp.exists():
            print(f'SKIP (not found): {vp.name}')
            skipped += 1
            continue
        print(f'Processing {vp.name} ...')
        try:
            frame = mid_frame(vp)
            save_strip(s['title'], frame, s['fn'], OUT / s['out'])
            ok += 1
        except Exception as exc:
            print(f'  ERROR: {exc}', file=sys.stderr)
            err += 1

    print(f'\nDone: {ok} strips saved, {skipped} skipped, {err} errors')


if __name__ == '__main__':
    main()
