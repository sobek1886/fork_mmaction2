"""Make a stratified ~50% subset of the ASL-Citizen train split.

This is a DATA-PREP step only: it writes a new split directory whose train.csv is a
deterministic per-gloss subsample of the original train.csv. It does NOT add a sampler
to the training code — the existing --train_csv machinery consumes the new CSV exactly
like splits/ or the smoke splits_tiny/. This mirrors how splits_tiny/ (1 row/gloss) was
produced, so it uses only existing split-CSV machinery.

Stratification: keep round(frac * n_i) rows of each gloss i (>=1 if the gloss has any),
sampled with a fixed seed. This preserves the 2,731-gloss vocabulary (train_logos builds
its label map from the ORIGINAL splits/train.csv, so vocab is unaffected regardless) and
keeps the signer/gloss balance roughly intact — unlike --max_videos, which just truncates
the first N rows (signer-biased).

val.csv and test.csv are copied UNCHANGED (eval must stay on the full held-out sets).

Usage (run on cluster login node, 0 SBU):
    python make_asl_citizen_train_subset.py \
        --train_csv $HOME/ASL_Citizen/splits/train.csv \
        --val_csv   $HOME/ASL_Citizen/splits/val.csv \
        --test_csv  $HOME/ASL_Citizen/splits/test.csv \
        --frac 0.5 --seed 0 \
        --output_dir $HOME/ASL_Citizen/splits_half

Then, for the aug reduced variant, feed splits_half/train.csv into the EXISTING
generate_asl_citizen_aug_splits.py (Fork_SignCLIP/examples/MMPT) to build
splits_half_aug/ — no new machinery:
    python generate_asl_citizen_aug_splits.py \
        --train_csv $HOME/ASL_Citizen/splits_half/train.csv \
        --val_csv   $HOME/ASL_Citizen/splits/val.csv \
        --test_csv  $HOME/ASL_Citizen/splits/test.csv \
        --aug_video_list $HOME/ASL_Citizen/aug_selection/aug_video_list.txt \
        --variants glasses shirt_1 signer_swap skin_mst_diffusion \
        --output_dir $HOME/ASL_Citizen/splits_half_aug

NOT SUBMITTED context: preparing files only; nothing is run here.
"""
import argparse
import csv
import os
import random
import shutil
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.train_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        by_gloss = defaultdict(list)
        for row in reader:
            by_gloss[row["Gloss"].strip()].append(row)

    rng = random.Random(args.seed)
    kept = []
    for gloss in sorted(by_gloss):
        rows = by_gloss[gloss]
        k = max(1, round(args.frac * len(rows)))
        kept.extend(rng.sample(rows, k))

    out_train = os.path.join(args.output_dir, "train.csv")
    with open(out_train, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    shutil.copy(args.val_csv, os.path.join(args.output_dir, "val.csv"))
    shutil.copy(args.test_csv, os.path.join(args.output_dir, "test.csv"))

    print(f"[subset] {len(kept)} / {sum(len(v) for v in by_gloss.values())} train rows "
          f"({len(by_gloss)} glosses preserved) -> {out_train}")


if __name__ == "__main__":
    main()
