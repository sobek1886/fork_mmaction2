#!/usr/bin/env python3
"""Attach a file (the SLURM .txt log) as an artifact to an existing MLflow run.

Used at the end of the FT job scripts to upload the full job log (training +
re-extraction) to the run created by train_logos_asl_citizen.py.

Usage:
    python mlflow_log_slurm_log.py <run_id> <path_to_logfile>

Failure-tolerant: prints a warning and exits 0 if MLflow / the artifact store is
unreachable, so it never fails the SLURM job.
"""

import sys

_TRACKING_URI = "https://mlflow.ai.mytkhgroup.com/"


def main():
    if len(sys.argv) != 3:
        print("usage: mlflow_log_slurm_log.py <run_id> <logfile>")
        return
    run_id, logfile = sys.argv[1], sys.argv[2]
    try:
        import mlflow
        mlflow.set_tracking_uri(_TRACKING_URI)
        mlflow.tracking.MlflowClient().log_artifact(run_id, logfile)
        print(f"[MLflow] logged artifact {logfile} -> run {run_id}")
    except Exception as exc:
        print(f"[MLflow] WARNING: could not log artifact: {exc}")


if __name__ == "__main__":
    main()
