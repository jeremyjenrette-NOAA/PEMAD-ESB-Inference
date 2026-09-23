"""Resolve and poll the Cloud Batch job a DAG run submitted.

Uses the same `google.cloud.batch_v1` client the DAG itself uses
(confirmed against the actual nmfs-optics-pipeline-longrunning-dag
source), so job-state handling here matches the DAG's own terminal-state
logic (SUCCEEDED / FAILED / DELETION_IN_PROGRESS).

KNOWN LIMITATION: `find_job_by_prefix` finds a job by listing recent
Cloud Batch jobs and matching the `<model_type>-` name prefix, then
taking the most recently created match. That's a heuristic, not an exact
lookup -- if you (or a teammate) trigger two runs of the *same* model
concurrently, it can grab the wrong job. Fine for solo/serial testing;
worth tightening (e.g. by parsing the real run_id out of the
submit_cloud_batch_job task log) before relying on this for
parallel/team runs.
"""
from __future__ import annotations

import time
from typing import Optional

from google.cloud import batch_v1

from . import config


def _client() -> batch_v1.BatchServiceClient:
    return batch_v1.BatchServiceClient()


def find_job_by_prefix(model_type: str, region: str = config.REGION) -> Optional[str]:
    """Find the most recently created Cloud Batch job whose name starts
    with `<model_type>-` (the DAG's naming convention:
    `f"{model_type}-{run_id[:8]}"`). Returns the job's full resource
    name, or None if nothing matches.
    """
    parent = f"projects/{config.PROJECT_ID}/locations/{region}"
    client = _client()
    matches = [
        job for job in client.list_jobs(parent=parent)
        if job.name.rsplit("/", 1)[-1].startswith(f"{model_type}-")
    ]
    if not matches:
        return None
    matches.sort(key=lambda j: j.create_time)
    return matches[-1].name


def get_job_state(job_name: str) -> str:
    """`job_name` may be a bare job id (e.g. 'ultralytics-9fa24dfe') or a
    full resource name; bare ids are resolved against PROJECT_ID/REGION.
    """
    client = _client()
    if not job_name.startswith("projects/"):
        job_name = f"projects/{config.PROJECT_ID}/locations/{config.REGION}/jobs/{job_name}"
    job = client.get_job(name=job_name)
    return job.status.state.name


def poll_until_terminal(job_name: str, interval_seconds: int = 30) -> str:
    """Poll a Cloud Batch job until it reaches a terminal state, printing
    state transitions as they happen (mirrors the DAG's own
    wait_for_cloud_batch_job polling loop, just at a shorter interval
    since this is meant for interactive `--wait` use, not a days-long
    unattended job).
    """
    terminal = {"SUCCEEDED", "FAILED", "DELETION_IN_PROGRESS"}
    previous = None
    while True:
        state = get_job_state(job_name)
        if state != previous:
            print(f"[BATCH] {job_name}: {state}")
            previous = state
        if state in terminal:
            return state
        time.sleep(interval_seconds)
