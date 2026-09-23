"""Stage local weights + a local pipeline-YAML template to GCS, wiring the
uploaded weights path into the config automatically.

Solves the manual "gcloud storage cp weights, then hand-edit the yaml's
`weights:` line" workflow: give a local .pt file and a local YAML
template, get back a ready-to-use gs://.../config.yaml URI (and the
weights URI it points at).

Scope: `rewrite_weights_field` only touches a top-level `weights:` key,
which is an Ultralytics-style pipeline-YAML thing (see
configs/ultralytics/24star.yaml). VIAME-family configs don't have a
`weights:` field at all -- their weights are baked into the Docker image
at build time (see docs/byom-docker-onboarding.md) -- so this module
doesn't apply there.

Multi-weight (cascade) models: `stage_config`'s `extra_weights` param and
the more general `rewrite_yaml_field` function extend the same
stage-and-rewrite trick to models that need more than one weights file --
e.g. a two-stage detector -> classifier cascade (see
docs/two-stage-cascade.md), which is built as a single container/model_type
whose `model.py` loads both a detector and a classifier checkpoint. The
actual field name(s) a cascade container's config expects (`weights` +
`classifier_weights`? something else?) are a property of that container's
`model.py`, not of this CLI -- confirm the real name(s) against the
container before relying on a specific one; `rewrite_yaml_field` works
with whatever name you pass it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import config

_YAML_FIELD_RE_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _field_re(field_name: str) -> "re.Pattern[str]":
    pattern = _YAML_FIELD_RE_CACHE.get(field_name)
    if pattern is None:
        pattern = re.compile(rf"(?m)^{re.escape(field_name)}:\s*.*$")
        _YAML_FIELD_RE_CACHE[field_name] = pattern
    return pattern


def rewrite_yaml_field(yaml_text: str, field_name: str, new_value: str) -> str:
    """Rewrite (or insert) a top-level `<field_name>: "<new_value>"` line
    in a pipeline YAML's raw text. Uses a targeted line-level substitution
    rather than a full YAML parse/dump round-trip, so existing comments
    and formatting (e.g. the alternate-weights lines commented out in
    24star.yaml) are preserved. Only touches an *uncommented*, top-level
    `<field_name>:` key -- a line starting with `#` never matches.

    General form of `rewrite_weights_field` -- use this directly for any
    field name other than `weights` (e.g. a cascade's second, classifier
    weights field).
    """
    replacement = f'{field_name}: "{new_value}"'
    pattern = _field_re(field_name)
    if pattern.search(yaml_text):
        return pattern.sub(replacement, yaml_text, count=1)
    # No existing top-level key with this name -- insert one at the top.
    return f"{replacement}\n{yaml_text}"


def rewrite_weights_field(yaml_text: str, new_weights_uri: str) -> str:
    """Rewrite (or insert) a top-level `weights:` line. Thin wrapper
    around `rewrite_yaml_field` kept for backwards compatibility and as
    the common case (a single-weights Ultralytics-style config).
    """
    return rewrite_yaml_field(yaml_text, "weights", new_weights_uri)


def stage_weights(
    local_weights_path: str,
    run_name: str,
    gcs_prefix: str,
    bucket: str = config.OUTPUT_BUCKET,
    weights_filename: Optional[str] = None,
) -> str:
    """Upload a local weights file to the standard layout:
    gs://<bucket>/<gcs_prefix>/weights/<run_name>/weights/<filename>
    (matches the convention already used by configs/ultralytics/24star.yaml).
    """
    from . import gcs  # deferred: keeps this module importable without google-cloud-storage

    filename = weights_filename or Path(local_weights_path).name
    dest = f"gs://{bucket}/{gcs_prefix}/weights/{run_name}/weights/{filename}"
    gcs.upload_file(local_weights_path, dest)
    return dest


def stage_config(
    local_yaml_path: str,
    local_weights_path: str,
    run_name: str,
    gcs_prefix: str,
    config_name: Optional[str] = None,
    bucket: str = config.OUTPUT_BUCKET,
    extra_weights: Optional[Dict[str, str]] = None,
) -> Tuple[str, str, str, Dict[str, str]]:
    """Upload weights, rewrite the local YAML template's `weights:` field
    to point at the uploaded path, and upload the resulting config.

    `extra_weights` is an optional {yaml_field_name: local_weights_path}
    map for multi-weight (cascade) models -- e.g. a two-stage detector ->
    classifier model whose config needs both a `weights:` field (handled
    as the primary/detector weights above) and a second field such as
    `classifier_weights:`. Each extra file is staged next to the primary
    weights (gs://<bucket>/<gcs_prefix>/weights/<run_name>/weights/<field_name>_<filename>)
    and that field is rewritten in the YAML the same way `weights:` is.
    The field name is whatever the target container's `model.py` actually
    reads from its config -- confirm it against the real container (see
    docs/two-stage-cascade.md) rather than assuming `classifier_weights`.

    Returns (yaml_config_uri, weights_uri, rewritten_yaml_text, extra_weight_uris)
    -- the text is returned too so callers (e.g. the CLI's run-card builder)
    don't have to re-derive or re-download it, and extra_weight_uris maps
    each extra_weights field name to where it ended up.
    """
    from . import gcs  # deferred, see stage_weights

    weights_uri = stage_weights(local_weights_path, run_name, gcs_prefix, bucket=bucket)
    yaml_text = Path(local_yaml_path).read_text(encoding="utf-8")
    new_yaml_text = rewrite_weights_field(yaml_text, weights_uri)

    extra_weight_uris: Dict[str, str] = {}
    for field_name, local_path in (extra_weights or {}).items():
        filename = f"{field_name}_{Path(local_path).name}"
        extra_uri = stage_weights(
            local_path, run_name, gcs_prefix, bucket=bucket, weights_filename=filename
        )
        new_yaml_text = rewrite_yaml_field(new_yaml_text, field_name, extra_uri)
        extra_weight_uris[field_name] = extra_uri

    name = config_name or f"{run_name}.yaml"
    dest = f"gs://{bucket}/{gcs_prefix}/configs/{name}"
    gcs.upload_text(new_yaml_text, dest, content_type="text/yaml")
    return dest, weights_uri, new_yaml_text, extra_weight_uris
