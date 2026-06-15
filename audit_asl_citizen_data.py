#!/usr/bin/env python3
"""
audit_asl_citizen_data.py — check ASL-Citizen data completeness (read-only).

Reports:
  1) augmented_frames/{video_id}/{aug}/ : how many variant dirs are EMPTY (0 jpg)
     vs populated, broken down per augmentation (glasses / shirt_1 / ...).
  2) splits_aug/train.csv aug rows that resolve to empty/missing frame dirs
     (these are exactly what crashes the Run-3 / aug-as-data training).
  3) originals in splits/{train,val,test}.csv : missing or 0-byte mp4s.

Run on Snellius (where the data lives). Writes the list of empty dirs to a file
so you can re-generate/re-sync just those.

Usage:
  python audit_asl_citizen_data.py
  python audit_asl_citizen_data.py --aug_names glasses shirt_1 skin_mst_diffusion
"""

import argparse
import csv
import os
from pathlib import Path

DEF_AUG   = '/Users/piotr/Projects/Thesis/nano_banana/output/flux-aug/asl_citizen/augmented_frames' #"/scratch-shared/psobecki/ASL_Citizen/augmented_frames"
DEF_VIDEO = "/scratch-shared/psobecki/ASL_Citizen/videos"
DEF_SPLITS = "/home/psobecki/ASL_Citizen/splits"
DEF_SPLITS_AUG = "/home/psobecki/ASL_Citizen/splits_aug"
AUG_NAMES = ("glasses", "shirt_1")


def count_jpgs(d):
    n = 0
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.name.lower().endswith(".jpg"):
                    n += 1
    except OSError:
        return -1
    return n


def split_aug_suffix(basename, aug_names):
    for a in aug_names:
        if basename.endswith("_" + a):
            return basename[: -(len(a) + 1)], a
    return basename, None


def audit_aug_dirs(aug_dir, aug_names):
    aug_dir = Path(aug_dir)
    per_aug = {a: {"total": 0, "empty": 0} for a in aug_names}
    empties = []
    n_vid = 0
    for vd in os.scandir(aug_dir):
        if not vd.is_dir():
            continue
        n_vid += 1
        for a in aug_names:
            d = Path(vd.path) / a
            if d.is_dir():
                per_aug[a]["total"] += 1
                if count_jpgs(d) <= 0:
                    per_aug[a]["empty"] += 1
                    empties.append(str(d))
    return n_vid, per_aug, empties


def audit_splits_aug(csv_path, aug_dir, aug_names):
    aug_dir = Path(aug_dir)
    if not Path(csv_path).exists():
        return None
    n_aug_rows = empty = missing = ok = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            base = os.path.splitext(row["Video file"])[0]
            vid, aug = split_aug_suffix(base, aug_names)
            if aug is None:
                continue
            n_aug_rows += 1
            d = aug_dir / vid / aug
            if not d.is_dir():
                missing += 1
            elif count_jpgs(d) <= 0:
                empty += 1
            else:
                ok += 1
    return n_aug_rows, ok, empty, missing


def audit_originals(video_dir, splits_dir):
    video_dir = Path(video_dir)
    out = {}
    for split in ["train", "val", "test"]:
        p = Path(splits_dir) / f"{split}.csv"
        if not p.exists():
            continue
        ok = missing = zero = 0
        ex = []
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                base = os.path.splitext(row["Video file"])[0]
                fp = None
                for ext in (".mp4", ".MP4", ".avi", ".mov", ".mkv", ".webm"):
                    cand = video_dir / (base + ext)
                    if cand.exists():
                        fp = cand
                        break
                if fp is None:
                    missing += 1
                    if len(ex) < 10:
                        ex.append(base)
                elif fp.stat().st_size == 0:
                    zero += 1
                else:
                    ok += 1
        out[split] = dict(ok=ok, missing=missing, zero=zero, examples=ex)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aug_frames_dir", default=DEF_AUG)
    ap.add_argument("--video_dir", default=DEF_VIDEO)
    ap.add_argument("--splits_dir", default=DEF_SPLITS)
    ap.add_argument("--splits_aug", default=DEF_SPLITS_AUG)
    ap.add_argument("--aug_names", nargs="+", default=list(AUG_NAMES))
    ap.add_argument("--empties_out", default="empty_aug_dirs.txt")
    args = ap.parse_args()

    W = 64
    print("=" * W)
    print("AUGMENTED FRAME DIRS  (augmented_frames/{id}/{aug})")
    print("=" * W)
    n_vid, per_aug, empties = audit_aug_dirs(args.aug_frames_dir, args.aug_names)
    print(f"  video_id dirs under augmented_frames: {n_vid}")
    for a in args.aug_names:
        t, e = per_aug[a]["total"], per_aug[a]["empty"]
        pct = (100.0 * e / t) if t else 0.0
        print(f"  {a:20s}  dirs={t:6d}  empty={e:6d}  ({pct:5.1f}% empty)  populated={t-e}")
    with open(args.empties_out, "w") as f:
        f.write("\n".join(empties))
    print(f"  -> {len(empties)} empty dirs listed in {args.empties_out}")

    print("\n" + "=" * W)
    print("SPLITS_AUG TRAIN ROWS  (what Run-3 / aug-as-data trains on)")
    print("=" * W)
    sa = audit_splits_aug(Path(args.splits_aug) / "train.csv", args.aug_frames_dir, args.aug_names)
    if sa is None:
        print("  (splits_aug/train.csv not found)")
    else:
        n, ok, empty, missing = sa
        print(f"  aug rows: {n}  |  ok={ok}  empty_dir={empty}  missing_dir={missing}")
        if empty or missing:
            print(f"  *** {empty+missing} aug rows are unusable — regenerate splits_aug "
                  f"or re-create those frames ***")

    print("\n" + "=" * W)
    print("ORIGINAL VIDEOS  (splits/{train,val,test}.csv)")
    print("=" * W)
    orig = audit_originals(args.video_dir, args.splits_dir)
    for split, r in orig.items():
        print(f"  {split:5s}  ok={r['ok']:6d}  missing={r['missing']:5d}  zero_byte={r['zero']:4d}")
        if r["examples"]:
            print(f"         e.g. missing: {r['examples'][:5]}")
    print("=" * W)


if __name__ == "__main__":
    main()
