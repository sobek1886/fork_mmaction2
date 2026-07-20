#!/usr/bin/env bash
# One command to check the ASL-Citizen re-extraction + backbone-FT campaign
# (spec: context/feature-reextraction-spec-2026-07-17.md) from the MacBook:
#   - queue state of all extraction/training jobs
#   - final states of recently finished jobs (sacct, last 3 days)
#   - .npy counts for every feature dir the experiments consume
#   - last lines of each job log -> pulled to Fork_Logos/status/logs/
set -euo pipefail

REMOTE=psobecki@snellius.surf.nl
DEST="$(cd "$(dirname "$0")" && pwd)/status"
mkdir -p "$DEST/logs"

echo "== queue =="
ssh "$REMOTE" 'squeue -u psobecki -o "%10i %22j %8T %10M %R"' || true

echo
echo "== finished in the last 3 days =="
ssh "$REMOTE" 'sacct -S $(date -d "3 days ago" +%F) -u psobecki -X \
    --format=JobID%10,JobName%24,State%12,Elapsed,End -n | grep -Ev "PENDING|RUNNING" | tail -20' || true

echo
echo "== feature dir .npy counts =="
ssh "$REMOTE" bash -s <<'EOF'
S=/scratch-shared/psobecki/ASL_Citizen
printf "%-38s %s\n" "dir" "#npy"
for d in logos_features_native logos_features_run2 logos_features_run3 \
         logos_features_run1 logos_features_run1_glasses logos_features_run1_shirt1 \
         logos_features_run1_signerswap logos_features_run1_skin logos_features_run_sdda \
         logos_features_run_full_ce logos_features_run_full_glosscon logos_features_run_full_sdda; do
    n=$(ls "$S/$d" 2>/dev/null | wc -l)
    printf "%-38s %s\n" "$d" "$n"
done
echo "variant features inside logos_features_native (E4; expect 6000 each when done):"
for v in glasses shirt_1 signer_swap skin_mst_diffusion; do
    n=$(ls "$S/logos_features_native" 2>/dev/null | grep -c "_${v}.npy" || true)
    printf "  %-22s %s\n" "$v" "$n"
done
echo "checkpoints:"
for t in run2 run3 run1 run1_glasses run1_shirt1 run1_signerswap run1_skin run_sdda run_full_ce run_full_glosscon run_full_sdda; do
    [ -f "/scratch-shared/psobecki/runs/asl_ft_$t/checkpoint_best.pt" ] && echo "  asl_ft_$t: checkpoint_best.pt OK"
done
EOF

echo
echo "== pulling job logs =="
rsync -az "$REMOTE:fork_mmaction2/jobs/output/" "$DEST/logs/" 2>/dev/null || echo "(no logs yet)"
echo "log tails:"
for f in "$DEST"/logs/slurm_*.txt; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f")"
    tail -2 "$f" | tr '\r' '\n' | tail -2
done

echo
echo "Logs in: $DEST/logs/"
