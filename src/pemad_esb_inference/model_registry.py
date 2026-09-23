"""Load and validate entries from the live model_runtime_definitions.json.

This mirrors utils.runtime_loader.load_model_runtime_definitions, used by
the actual Airflow DAG (nmfs-optics-pipeline-longrunning-dag), so a model
key that resolves here is guaranteed to resolve the same way inside the
DAG. Running `resolve()` before `trigger` is the preflight check that
avoids the dev guide's "Model key not found" failure mode -- catching a
typo here takes a second; catching it inside a submitted Cloud Batch job
costs a full VM-provisioning cycle.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import config


@dataclass
class ModelDefinition:
    key: str
    region: str
    image: str
    cpu: int
    memory: str
    gpu: int
    gpu_type: Optional[str]
    machine_type: str
    timeout: int
    command: List[str]
    args: List[str]

    @property
    def family(self) -> str:
        """Infer the model family the same way the DAG itself does.

        model_runtime_definitions.json has no explicit "family" field.
        The DAG's submit_cloud_batch_job task instead treats any entry
        with an empty `command` list as a VIAME-style shell-args
        container (its `entrypoint_override_model_names` list), and
        everything else as an Ultralytics-style `python <script>`
        container. We reuse that exact heuristic rather than inventing
        a new one, so this never disagrees with what the DAG will do.
        """
        return "viame" if self.command == [] else "ultralytics"


def load_definitions(uri: str = config.MODEL_RUNTIME_DEFINITIONS_URI) -> Dict[str, ModelDefinition]:
    """Download and parse model_runtime_definitions.json from GCS."""
    # Deferred import: keeps ModelDefinition (a plain dataclass, useful in
    # tests and in run_card.py) importable without google-cloud-storage
    # installed -- only this function actually needs a live GCS call.
    from .gcs import download_text

    raw = json.loads(download_text(uri))
    return {
        key: ModelDefinition(
            key=key,
            region=entry.get("region", config.REGION),
            image=entry["image"],
            cpu=entry.get("cpu", 4),
            memory=entry.get("memory", "16Gi"),
            gpu=entry.get("gpu", 0),
            gpu_type=entry.get("gpu_type"),
            machine_type=entry.get("machine_type", "c2-standard-4"),
            timeout=entry.get("timeout", 360000),
            command=entry.get("command", []),
            args=entry.get("args", []),
        )
        for key, entry in raw.items()
    }


def resolve(model_type: str, definitions: Optional[Dict[str, ModelDefinition]] = None) -> ModelDefinition:
    """Look up a model key, raising a clear error (with the valid options
    listed) instead of letting a typo fail deep inside a Cloud Batch job.
    """
    if definitions is None:
        definitions = load_definitions()
    try:
        return definitions[model_type]
    except KeyError as exc:
        available = ", ".join(sorted(definitions)) or "(none found)"
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available model_type values "
            f"in {config.MODEL_RUNTIME_DEFINITIONS_URI}: {available}"
        ) from exc
