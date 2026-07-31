#!/usr/bin/env python3
"""Build the 450-gloss "augmented subset" splits for the sub450 from-scratch arms.

Design
------
Exactly 6,000 of the 40,154 ASL-Citizen train originals carry Flux augmentations
(4 variants each = 24,000 aug videos). A train video is a PARENT iff rows
    <vid>_glasses / _shirt_1 / _signer_swap / _skin_mst_diffusion
exist in splits_aug/train.csv. Those 6,000 parents cover 450 of the 2,731 glosses.

Outputs (pure ROW SUBSETS of the existing files -- header/column layout and row order
are preserved verbatim, so each pipeline sees exactly the format it already reads):

  splits_sub450/      (plain format: Participant ID,Video file,Gloss,ASL-LEX Code)
      train.csv   6,000 parent originals            <- CE-sub arm
      val.csv     splits/val.csv  filtered to the 450 glosses
      test.csv    splits/test.csv filtered to the 450 glosses

  splits_aug_sub450/  (train = aug format: Video file,Gloss ; val/test = plain, as in
                       splits_aug/ where val.csv/test.csv are copies of the plain ones)
      train.csv   30,000 rows = 6,000 parents + their 24,000 augs   <- AUG-sub arm
      val.csv     same filtered val
      test.csv    same filtered test

Label-space mechanics (why this naming matters)
-----------------------------------------------
train_logos_asl_citizen.py does
    base_train_csv = args.train_csv.replace("splits_aug", "splits")
    gloss_to_idx   = build_gloss_to_idx(base_train_csv)     # sorted(set(glosses))
".../splits_aug_sub450/train.csv".replace("splits_aug","splits") ==
".../splits_sub450/train.csv", so BOTH arms build their gloss vocab from the same
6,000-row parent file -> identical 450-way label indices. The head width is a
SEPARATE flag (--num_classes, default 2731) and must be passed as 450.
ClipDataset drops any row whose gloss is not in gloss_to_idx, so filtered val/test
needs no code change -- we still materialise it explicitly for the eval chain.
"""
import argparse
import csv
import os
import random
import statistics
from collections import Counter, defaultdict

AUGS = ["glasses", "shirt_1", "signer_swap", "skin_mst_diffusion"]


def read_rows(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write_rows(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def split_aug_suffix(basename):
    for a in AUGS:
        if basename.endswith("_" + a):
            return basename[: -(len(a) + 1)], a
    return basename, None


def stem(row):
    return os.path.splitext(row["Video file"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/ASL_Citizen"))
    ap.add_argument("--video_dir",
                    default="/scratch-shared/psobecki/ASL_Citizen/videos")
    ap.add_argument("--aug_frames_dir",
                    default="/scratch-shared/psobecki/ASL_Citizen/augmented_frames_512_x")
    args = ap.parse_args()
    R = args.root

    plain_fn, plain_train = read_rows(f"{R}/splits/train.csv")
    aug_fn, aug_train = read_rows(f"{R}/splits_aug/train.csv")
    val_fn, val_rows = read_rows(f"{R}/splits/val.csv")
    test_fn, test_rows = read_rows(f"{R}/splits/test.csv")
    print(f"source  splits/train.csv     {len(plain_train):>6} rows  {plain_fn}")
    print(f"source  splits_aug/train.csv {len(aug_train):>6} rows  {aug_fn}")

    # ---- identify parents deterministically -------------------------------
    have = defaultdict(set)
    for r in aug_train:
        vid, aug = split_aug_suffix(stem(r))
        if aug:
            have[vid].add(aug)
    parents = {v for v, s in have.items() if len(s) == len(AUGS)}
    partial = {v: sorted(s) for v, s in have.items() if len(s) != len(AUGS)}
    print(f"parents with all {len(AUGS)} augs: {len(parents)}   "
          f"(partial-aug parents: {len(partial)})")
    assert not partial, f"parents with an incomplete aug set: {list(partial)[:5]}"

    # ---- CE arm: 6,000 parent originals, plain format ---------------------
    ce_train = [r for r in plain_train if stem(r) in parents]
    assert len(ce_train) == len(parents), (len(ce_train), len(parents))
    glosses = sorted({r["Gloss"].strip() for r in ce_train})
    gset = set(glosses)

    # ---- AUG arm: parents + their augs, aug format ------------------------
    def keep_aug(r):
        vid, _ = split_aug_suffix(stem(r))
        return vid in parents
    aug_sub = [r for r in aug_train if keep_aug(r)]

    # ---- val / test filtered to the 450 glosses ---------------------------
    val_sub = [r for r in val_rows if r["Gloss"].strip() in gset]
    test_sub = [r for r in test_rows if r["Gloss"].strip() in gset]

    write_rows(f"{R}/splits_sub450/train.csv", plain_fn, ce_train)
    write_rows(f"{R}/splits_sub450/val.csv", val_fn, val_sub)
    write_rows(f"{R}/splits_sub450/test.csv", test_fn, test_sub)
    write_rows(f"{R}/splits_aug_sub450/train.csv", aug_fn, aug_sub)
    write_rows(f"{R}/splits_aug_sub450/val.csv", val_fn, val_sub)
    write_rows(f"{R}/splits_aug_sub450/test.csv", test_fn, test_sub)

    # ---- sanity report ----------------------------------------------------
    print("\n===== row counts / gloss counts =====")
    for name, rows, want in [
        ("splits_sub450/train.csv", ce_train, 6000),
        ("splits_sub450/val.csv", val_sub, None),
        ("splits_sub450/test.csv", test_sub, None),
        ("splits_aug_sub450/train.csv", aug_sub, 30000),
        ("splits_aug_sub450/val.csv", val_sub, None),
        ("splits_aug_sub450/test.csv", test_sub, None),
    ]:
        g = {r["Gloss"].strip() for r in rows}
        flag = "" if want is None or len(rows) == want else f"  <<< EXPECTED {want}"
        print(f"  {name:<32} rows={len(rows):>6}  glosses={len(g):>4}"
              f"  {'OK' if len(g) == 450 else '<<< NOT 450'}{flag}")

    n_orig = sum(1 for r in aug_sub if split_aug_suffix(stem(r))[1] is None)
    print(f"\n  aug train composition: {n_orig} originals + "
          f"{len(aug_sub) - n_orig} augmented  (expect 6000 + 24000)")

    c = Counter(r["Gloss"].strip() for r in ce_train)
    print(f"  parents/gloss  min={min(c.values())} median="
          f"{statistics.median(c.values())} max={max(c.values())} "
          f"mean={len(ce_train) / len(c):.2f}")
    # per-gloss density of the FULL train set, for the "density matched" claim
    cf = Counter(r["Gloss"].strip() for r in plain_train)
    print(f"  full-set videos/gloss  min={min(cf.values())} median="
          f"{statistics.median(cf.values())} max={max(cf.values())} "
          f"mean={len(plain_train) / len(cf):.2f}")

    # gloss index space identical for both arms?
    idx_ce = {g: i for i, g in enumerate(sorted({r["Gloss"].strip() for r in ce_train}))}
    idx_aug = {g: i for i, g in enumerate(sorted({r["Gloss"].strip() for r in aug_sub}))}
    print(f"  gloss_to_idx(CE) == gloss_to_idx(AUG): {idx_ce == idx_aug} "
          f"(|V|={len(idx_ce)})")

    # ---- path existence on a 20-row sample of each file -------------------
    print("\n===== path existence (20-row random sample per file) =====")
    rng = random.Random(0)

    def check_orig(rows, label):
        miss = []
        for r in rng.sample(rows, min(20, len(rows))):
            b = stem(r)
            if not any(os.path.exists(os.path.join(args.video_dir, b + e))
                       for e in (".mp4", ".MP4", ".avi", ".mov", ".mkv", ".webm")):
                miss.append(b)
        print(f"  {label:<32} missing {len(miss)}/20 mp4  {miss[:3]}")

    check_orig(ce_train, "sub450 train (originals)")
    check_orig(val_sub, "sub450 val")
    check_orig(test_sub, "sub450 test")

    aug_only = [r for r in aug_sub if split_aug_suffix(stem(r))[1]]
    miss = []
    for r in rng.sample(aug_only, 20):
        vid, a = split_aug_suffix(stem(r))
        d = os.path.join(args.aug_frames_dir, vid, a)
        if not (os.path.isdir(d) and any(x.endswith(".jpg") for x in os.listdir(d))):
            miss.append(f"{vid}/{a}")
    print(f"  {'aug train (frame dirs)':<32} missing {len(miss)}/20 dirs  {miss[:3]}")
    print(f"    (aug_frames_dir = {args.aug_frames_dir})")
    # full check of parents against the tar inventory, cheap and decisive
    tars = {os.path.splitext(f)[0]
            for f in os.listdir("/scratch-shared/psobecki/ASL_Citizen/augmented_frames_512")
            if f.endswith(".tar")}
    print(f"  parents covered by a .tar: {len(parents & tars)}/{len(parents)}")


if __name__ == "__main__":
    main()
