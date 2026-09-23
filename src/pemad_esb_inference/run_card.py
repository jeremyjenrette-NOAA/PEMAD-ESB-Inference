"""Build a "run card" packaging together everything that defines an
inference run -- model, its frozen resource definition, the resolved
input manifest, a frozen snapshot of the pipeline YAML's contents, and a
creation timestamp -- *before* the DAG is ever triggered.

This is deliberately independent of Airflow's own archived-runtime-YAML
mechanism (gs://.../configs/archive/<run_id>.yaml): that only exists
*after* a run is submitted, and only captures the pipeline YAML, not the
model's resource definition or how the input list was built (e.g. a
--survey-prefix sample and at what rate). A run card exists pre-flight,
so `pemad-infer trigger ... --dry-run` alone produces one.

`build_run_card` is pure (no I/O) and testable without credentials;
`upload_run_card` does the actual GCS write.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from . import config

if TYPE_CHECKING:
    from .model_registry import ModelDefinition


@dataclass
class RunCard:
    created_at: str
    model_type: str
    model_definition: Dict[str, Any]
    input_file: str
    input_count: Optional[int]
    yaml_config_path: str
    yaml_config_snapshot: str
    output_bucket: str
    output_folder: str
    gcs_prefix: str
    survey_sample: Optional[Dict[str, Any]] = None
    extra_weights: Optional[Dict[str, str]] = None
    staged_input: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def build_run_card(
    model_type: str,
    model_definition: "ModelDefinition",
    input_file: str,
    yaml_config_path: str,
    yaml_config_text: str,
    output_bucket: str,
    output_folder: str,
    gcs_prefix: str,
    input_count: Optional[int] = None,
    survey_sample: Optional[Dict[str, Any]] = None,
    extra_weights: Optional[Dict[str, str]] = None,
    staged_input: Optional[Dict[str, Any]] = None,
) -> RunCard:
    model_def_dict = dataclasses.asdict(model_definition)
    model_def_dict["family"] = model_definition.family  # a property, not a dataclass field -- add explicitly
    return RunCard(
        created_at=datetime.now(timezone.utc).isoformat(),
        model_type=model_type,
        model_definition=model_def_dict,
        input_file=input_file,
        input_count=input_count,
        yaml_config_path=yaml_config_path,
        yaml_config_snapshot=yaml_config_text,
        output_bucket=output_bucket,
        output_folder=output_folder,
        gcs_prefix=gcs_prefix,
        survey_sample=survey_sample,
        extra_weights=extra_weights,
        staged_input=staged_input,
    )


def upload_run_card(card: RunCard, bucket: str = config.OUTPUT_BUCKET) -> str:
    from . import gcs  # deferred: keeps build_run_card usable without google-cloud-storage

    timestamp_slug = card.created_at.replace(":", "").replace("+00:00", "Z")
    dest = f"gs://{bucket}/{card.gcs_prefix}/run_cards/{card.model_type}_{timestamp_slug}.json"
    gcs.upload_json(card.to_dict(), dest)
    return dest
