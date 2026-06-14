#!/usr/bin/env python3
"""
test_preprocessing_asl_citizen.py — compare the 3 preprocessing variants on ASL-Citizen.

For 2 ASL-Citizen samples and each appearance variant (original / glasses / shirt_1),
visualise the MIDDLE frame under all three preprocessing modes side by side:

    raw  |  direct224  |  resize300crop  |  logos_native

so you can eyeball aspect distortion and signer framing before committing to logos_native
for the fine-tuning + re-extraction. (Only the SPATIAL op matters for framing; the real
pipeline additionally subtracts MEAN/STD, which is irrelevant to this visual check.)

The three modes mirror the code paths:
    direct224     = extract_logos_features.py legacy path (square-resize; aspect-DISTORTS
                    non-square ASL-Citizen frames). Used by the old baseline/endanchor.
    resize300crop = extract_logos_features_asl_citizen_aug.py path (resize SHORT side->300,
                    pad, center-crop 224; crops the long-axis edges).
    logos_native  = the correct, aspect-preserving path now used everywhere (resize LONG
                    side->300, pad to 300x300 grey-114, center-crop 224).

Expected local layout (sync from Snellius first — see commands below):
    {examples_dir}/{video_id}.mp4                            (original)
    {examples_dir}/augmented_frames/{video_id}/{aug}/*.jpg   (augmented frames)

# 1) pick 2 ids that HAVE augmentations:
ssh snellius 'ls /scratch-shared/psobecki/ASL_Citizen/augmented_frames | head'
# 2) for each chosen ID, sync the original mp4 + its augmented frame dirs:
mkdir -p ~/Desktop/asl_citizen_examples/augmented_frames
rsync -avz snellius:/scratch-shared/psobecki/ASL_Citizen/videos/<ID>.mp4 \
      ~/Desktop/asl_citizen_examples/
rsync -avz snellius:/scratch-shared/psobecki/ASL_Citizen/augmented_frames/<ID> \
      ~/Desktop/asl_citizen_examples/augmented_frames/

Usage:
    python test_preprocessing_asl_citizen.py            # autodiscovers *.mp4 (first 2)
    python test_preprocessing_asl_citizen.py --video_ids 1234-HATCH 5678-MARCH
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_SIZE = 224
RESIZE     = 300
PAD_VALUE  = 114

MODES = ["direct224", "resize300crop", "logos_native"]


# ── Spatial preprocessing (display-only; mirrors the real pipelines, sans normalize) ──

def prep(frame_rgb, mode):
    if mode == "direct224":
        return cv2.resize(frame_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    if mode in ("logos_native", "resize300crop"):
        h, w = frame_rgb.shape[:2]
        scale = RESIZE / (max(h, w) if mode == "logos_native" else min(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        f = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        ph, pw = max(0, RESIZE - nh), max(0, RESIZE - nw)
        f = np.pad(f, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)),
                   mode="constant", constant_values=PAD_VALUE)
        off_y = (f.shape[0] - INPUT_SIZE) // 2
        off_x = (f.shape[1] - INPUT_SIZE) // 2
        return f[off_y:off_y + INPUT_SIZE, off_x:off_x + INPUT_SIZE]
    raise ValueError(mode)


# ── Frame loading ──

def mid_frame_mp4(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame from {path}")
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def mid_frame_jpgdir(d):
    files = sorted(Path(d).glob("*.jpg"))
    if not files:
        raise RuntimeError(f"no .jpg frames in {d}")
    fr = cv2.imread(str(files[len(files) // 2]))
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def sources_for(examples_dir, vid, augs):
    """Return [(label, rgb_frame), ...] for original + each present augmentation."""
    out = []
    mp4 = examples_dir / f"{vid}.mp4"
    if mp4.exists():
        out.append(("original", mid_frame_mp4(mp4)))
    else:
        print(f"  (no original mp4 for {vid} at {mp4})")
    for a in augs:
        d = examples_dir / "augmented_frames" / vid / a
        if d.is_dir():
            out.append((a, mid_frame_jpgdir(d)))
    return out


def make_fig(vid, srcs, out_path):
    nrows, ncols = len(srcs), 1 + len(MODES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for r, (label, frame) in enumerate(srcs):
        h, w = frame.shape[:2]
        axes[r][0].imshow(frame)
        axes[r][0].set_title(f"{label}\nraw {w}x{h}  (AR {w/h:.2f})", fontsize=9)
        for c, mode in enumerate(MODES, start=1):
            axes[r][c].imshow(prep(frame, mode))
            axes[r][c].set_title(f"{mode}\n{INPUT_SIZE}x{INPUT_SIZE}", fontsize=9)
        for c in range(ncols):
            axes[r][c].axis("off")
    fig.suptitle(f"ASL-Citizen preprocessing comparison — {vid}",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples_dir", type=Path,
                    default=Path.home() / "Desktop" / "asl_citizen_examples")
    ap.add_argument("--video_ids", nargs="*", default=None,
                    help="video_id stems (default: autodiscover *.mp4, first 2)")
    ap.add_argument("--augs", nargs="+", default=["glasses", "shirt_1"])
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    ed = args.examples_dir
    if not ed.exists():
        sys.exit(f"examples_dir not found: {ed}\nSync 2 samples from Snellius first "
                 f"(see the rsync commands in this file's header).")
    out_dir = args.out_dir or (ed / "preprocessing_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = args.video_ids or sorted(p.stem for p in ed.glob("*.mp4"))
    if not vids:
        sys.exit(f"No .mp4 originals found in {ed}. Sync samples first.")
    vids = vids[:args.n_samples]
    print(f"examples_dir: {ed}\nsamples: {vids}")

    for vid in vids:
        print(f"Processing {vid} ...")
        srcs = sources_for(ed, vid, args.augs)
        if not srcs:
            print(f"  skip {vid}: no sources found")
            continue
        make_fig(vid, srcs, out_dir / f"{vid}_preproc.png")

    print(f"\nDone. Open the PNGs in {out_dir}")


if __name__ == "__main__":
    main()
