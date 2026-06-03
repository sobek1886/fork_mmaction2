#!/bin/bash
# Master submission script for NGT feature re-extraction pipeline.
#
# Dependency graph:
#
#   crop_piotr_videos   ──►  extract_logos_features_piotranims
#   crop_bushuis_videos ──►  extract_logos_features_bushuis
#   extract_logos_features_ngt_aug  (crop is inline; no upstream dependency)
#
# SLURM handles the sequencing: extraction jobs are submitted with
# --dependency=afterok:<crop_job_id>, so they are only released by the
# scheduler once the corresponding crop job exits with code 0.
#
# Usage (from any directory on Snellius):
#   bash $HOME/fork_mmaction2/submit_ngt_jobs.sh

set -euo pipefail

SCRIPTS=$HOME/fork_mmaction2

# ── 1. Offline crop jobs ─────────────────────────────────────────────────────
echo "Submitting crop jobs..."

JOB_CROP_PIOTR=$(sbatch --parsable $SCRIPTS/crop_piotr_videos.job)
echo "  crop_piotr_videos    → job $JOB_CROP_PIOTR"

JOB_CROP_BUSHUIS=$(sbatch --parsable $SCRIPTS/crop_bushuis_videos.job)
echo "  crop_bushuis_videos  → job $JOB_CROP_BUSHUIS"

# ── 2. Feature extraction ─────────────────────────────────────────────────────
# Each extraction job carries --dependency=afterok so it is held in the queue
# until its crop job finishes successfully.  If the crop job fails the
# extraction job is automatically cancelled.
echo "Submitting extraction jobs (held until crop jobs succeed)..."

JOB_EXT_PIOTR=$(sbatch --parsable \
    --dependency=afterok:$JOB_CROP_PIOTR \
    $SCRIPTS/extract_logos_features_piotranims.job)
echo "  logos_piotranims     → job $JOB_EXT_PIOTR  (waits for $JOB_CROP_PIOTR)"

JOB_EXT_BUSHUIS=$(sbatch --parsable \
    --dependency=afterok:$JOB_CROP_BUSHUIS \
    $SCRIPTS/extract_logos_features_bushuis.job)
echo "  logos_bushuis        → job $JOB_EXT_BUSHUIS  (waits for $JOB_CROP_BUSHUIS)"

# NGT_Aug (palmer / digits PNG sequences): crop is handled inline inside
# extract_logos_features_pngseq.py, so this job has no upstream dependency.
JOB_EXT_NGT_AUG=$(sbatch --parsable $SCRIPTS/extract_logos_features_ngt_aug.job)
echo "  logos_ngt_aug        → job $JOB_EXT_NGT_AUG  (starts immediately)"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "Pipeline submitted:"
echo "  [$JOB_CROP_PIOTR]  crop_piotr_videos"
echo "     └─► [$JOB_EXT_PIOTR]  logos_piotranims"
echo "  [$JOB_CROP_BUSHUIS]  crop_bushuis_videos"
echo "     └─► [$JOB_EXT_BUSHUIS]  logos_bushuis"
echo "  [$JOB_EXT_NGT_AUG]  logos_ngt_aug  (independent)"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ~/fork_mmaction2/jobs/output/"
