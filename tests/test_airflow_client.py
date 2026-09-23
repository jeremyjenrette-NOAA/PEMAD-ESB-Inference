"""Confirms the `gcloud composer environments run` argv this package
builds matches gcloud's actual usage:

    gcloud composer environments run (ENVIRONMENT : --location=LOCATION)
        SUBCOMMAND [SUBCOMMAND_NESTED] [optional flags] [-- CMD_ARGS ...]

i.e. the airflow subcommand ("dags trigger", "dags list-runs", "tasks
states-for-dag-run") must appear *before* `--`, and only the arguments for
that subcommand itself go after it. An earlier version of this module put
the whole thing after `--`, which gcloud rejected with
"argument SUBCOMMAND: Must be specified" -- see the module docstring in
airflow_client.py.
"""
from unittest.mock import MagicMock

from pemad_esb_inference import airflow_client, config


def _mock_subprocess_run(monkeypatch, returncode=0, stdout="ok", stderr=""):
    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        result.stderr = stderr
        return result

    monkeypatch.setattr(airflow_client.subprocess, "run", fake_run)
    return captured


def test_trigger_dag_puts_subcommand_before_double_dash(monkeypatch):
    captured = _mock_subprocess_run(monkeypatch)
    airflow_client.trigger_dag({"model_type": "ultralytics"}, dag_id="my-dag")
    assert captured["cmd"] == [
        "gcloud", "composer", "environments", "run", config.COMPOSER_ENVIRONMENT,
        "--location", config.COMPOSER_LOCATION,
        "dags", "trigger",
        "--",
        "my-dag", "--conf", '{"model_type": "ultralytics"}',
    ]


def test_list_runs_puts_subcommand_before_double_dash(monkeypatch):
    captured = _mock_subprocess_run(monkeypatch)
    airflow_client.list_runs(dag_id="my-dag")
    assert captured["cmd"] == [
        "gcloud", "composer", "environments", "run", config.COMPOSER_ENVIRONMENT,
        "--location", config.COMPOSER_LOCATION,
        "dags", "list-runs",
        "--",
        "-d", "my-dag",
    ]


def test_task_states_for_run_puts_subcommand_before_double_dash(monkeypatch):
    captured = _mock_subprocess_run(monkeypatch)
    airflow_client.task_states_for_run("2026-09-17T13:04:20+00:00", dag_id="my-dag")
    assert captured["cmd"] == [
        "gcloud", "composer", "environments", "run", config.COMPOSER_ENVIRONMENT,
        "--location", config.COMPOSER_LOCATION,
        "tasks", "states-for-dag-run",
        "--",
        "my-dag", "2026-09-17T13:04:20+00:00",
    ]


def test_raises_airflow_command_error_on_nonzero_exit(monkeypatch):
    _mock_subprocess_run(monkeypatch, returncode=1, stdout="", stderr="boom")
    try:
        airflow_client.list_runs(dag_id="my-dag")
    except airflow_client.AirflowCommandError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected AirflowCommandError")
