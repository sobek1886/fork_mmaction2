#!/usr/bin/env python3
"""Average NGT retrieval results over K independent training runs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

METRICS = ["R@1", "R@5", "R@10", "MRR"]
# Legacy eval_results.json (make_ngt_pair_manifest.py evals)
DIRECTIONS_LEGACY = {
    "b2s_avg": "B→S  avg(L+M+R)",
    "b2s_mid": "B→S  mid only ",
    "s2b_avg": "S→B  avg(L+M+R)",
    "s2b_mid": "S→B  mid only ",
}
# Take-list eval_results.json ("style": "front"; Flux-vs-Unreal ladder)
DIRECTIONS_FRONT = {
    "b2s_front": "B→S  front view",
    "s2b_front": "S→B  front view",
    "b2s_avg":   "B→S  avg views ",
    "s2b_avg":   "S→B  avg views ",
}


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate NGT retrieval results over K runs.")
    parser.add_argument("--experiment", required=True,
                        help="Experiment name (e.g. ngt_baseline)")
    parser.add_argument("--base_dir",   required=True,
                        help="Base directory containing run subdirs")
    parser.add_argument("--k",          type=int, default=5,
                        help="Number of runs (default: 5)")
    args = parser.parse_args()

    run_dirs = [
        Path(args.base_dir) / f"{args.experiment}_run{i}"
        for i in range(1, args.k + 1)
    ]

    results = []
    for d in run_dirs:
        p = d / "eval_results.json"
        if not p.exists():
            print(f"WARNING: missing {p}", file=sys.stderr)
            continue
        with open(p) as f:
            results.append(json.load(f))

    if not results:
        sys.exit("No results found.")

    K = len(results)
    front_style = results[0].get("style") == "front"
    DIRECTIONS = DIRECTIONS_FRONT if front_style else DIRECTIONS_LEGACY
    print(f"\nExperiment: {args.experiment}  ({K}/{args.k} runs found)")
    print("=" * 65)

    summary = {}
    for dir_key, dir_label in DIRECTIONS.items():
        print(f"\n  {dir_label}")
        summary[dir_key] = {}
        for metric in METRICS:
            vals = [r[dir_key][metric] for r in results]
            mean = float(np.mean(vals))
            std  = float(np.std(vals, ddof=1))
            summary[dir_key][metric] = {"mean": mean, "std": std, "runs": vals}
            runs_str = "  ".join(f"{v:5.1f}" for v in vals)
            print(f"    {metric:<6}  {mean:5.2f} ± {std:4.2f}   [{runs_str}]")

    if front_style:
        # Shortcut floor: retrieval by sentence length alone (same every run).
        lb = results[0].get("length_baseline", {})
        for key, label in [("len_b2s", "B→S"), ("len_s2b", "S→B")]:
            if key in lb:
                summary[key] = lb[key]
                print(f"\n  Length-only baseline {label}: "
                      + "  ".join(f"{m}={lb[key][m]:.1f}" for m in METRICS))

    out = Path(args.base_dir) / f"{args.experiment}_aggregated.json"
    with open(out, "w") as f:
        json.dump({"experiment": args.experiment, "k": K, "summary": summary},
                  f, indent=2)
    print(f"\nSaved to {out}\n")


if __name__ == "__main__":
    main()
