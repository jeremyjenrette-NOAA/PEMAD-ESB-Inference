"""Trigger and monitor DAG runs via `gcloud composer environments run`.

This shells out to gcloud rather than calling the Cloud Composer API
client directly. That's a deliberate choice, not a shortcut: this exact
command sequence is what was actually verified end-to-end against
composer-env1 on 2026-09-17, right after the
`composer.environments.executeAirflowCommand` permission was granted via
the Platform Team JIRA request (see docs/permissions.md) -- so it's the
known-good path. Swapping this for the native
`google-cloud-orchestration-airflow` client's ExecuteAirflowCommand API
is a reasonable future enhancement (drops the gcloud/subprocess
dependency) but isn't required for this to work today.
"""
from __future__ import annotations

import json
import subprocess
from typing import Dict, List

from . import config


class AirflowCommandError(RuntimeError):
    pass


def _run_airflow_command(subcommand: List[str], cmd_args: List[str]) -> str:
    """Run `gcloud composer environments run` and return stdout.

    `gcloud composer environments run`'s own usage is:
        gcloud composer environments run (ENVIRONMENT : --location=LOCATION)
            SUBCOMMAND [SUBCOMMAND_NESTED] [optional flags] [-- CMD_ARGS ...]

    SUBCOMMAND/SUBCOMMAND_NESTED (e.g. "dags trigger", "dags list-runs",
    "tasks states-for-dag-run") are gcloud's own positional arguments and
    must come *before* `--`; only the arguments meant for the underlying
    airflow CLI command (dag_id, --conf, flags, ...) go *after* `--`.
    Putting the airflow subcommand name after `--` (as an earlier version
    of this function did) makes gcloud treat SUBCOMMAND as unset, failing
    with "argument SUBCOMMAND: Must be specified" before anything ever
    reaches Airflow.
    """
    cmd = [
        "gcloud", "composer", "environments", "run", config.COMPOSER_ENVIRONMENT,
        "--location", config.COMPOSER_LOCATION,
        *subcommand,
        "--",
        *cmd_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AirflowCommandError(
            f"Command failed ({' '.join(cmd)}):\n{result.stderr or result.stdout}"
        )
    return result.stdout


def trigger_dag(conf: Dict, dag_id: str = config.DAG_ID) -> str:
    """Trigger a DAG run with the given conf dict. Returns the raw CLI
    output (includes the dag_run_id / logical_date in Airflow's table
    format) -- follow up with `list_runs()` for a clean, structured read.
    """
    return _run_airflow_command(["dags", "trigger"], [dag_id, "--conf", json.dumps(conf)])


def list_runs(dag_id: str = config.DAG_ID) -> str:
    return _run_airflow_command(["dags", "list-runs"], ["-d", dag_id])


def task_states_for_run(logical_date: str, dag_id: str = config.DAG_ID) -> str:
    """`logical_date` must match the format Airflow prints, e.g.
    '2026-09-17T13:04:20+00:00'.
    """
    return _run_airflow_command(["tasks", "states-for-dag-run"], [dag_id, logical_date])
