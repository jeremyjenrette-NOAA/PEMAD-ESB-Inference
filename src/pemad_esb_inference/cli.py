"""Command-line entry point: `pemad-infer`."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config, gcs, input_builder
from .input_staging import flatten_to_staging, spans_multiple_folders
from .airflow_client import list_runs, task_states_for_run, trigger_dag
from .batch_monitor import find_job_by_prefix, poll_until_terminal
from .config_builder import stage_config, stage_weights
from .habcam_paths import read_names_from_file, resolve_paths as resolve_habcam_paths
from .model_registry import load_definitions, resolve
from .run_card import build_run_card, upload_run_card
from .survey_sampler import sample_survey_folder


def _parse_extra_weights(pairs) -> dict:
    """Parse repeated `--extra-weights FIELD=PATH` values into a dict.

    Used for multi-weight (cascade) models -- see stage_config's
    `extra_weights` param and docs/two-stage-cascade.md.
    """
    result: dict = {}
    for item in pairs or []:
        field_name, sep, local_path = item.partition("=")
        field_name = field_name.strip()
        local_path = local_path.strip()
        if not sep or not field_name or not local_path:
            sys.exit(f"--extra-weights expects FIELD=PATH, got: {item!r}")
        result[field_name] = local_path
    return result


def _resolve_image_paths(args: argparse.Namespace):
    """Resolve the effective image list from whichever input-source flag
    was given. Exactly one of --survey-prefix, --habcam, --images is
    expected; if more than one is set this just uses the first match in
    that priority order.

    Returns (image_paths, survey_sample_dict_or_None) -- the second value
    is a JSON-serializable snapshot of the SampleReport (folder/image
    counts) when --survey-prefix was used, for embedding in a run card.
    """
    survey_prefix = getattr(args, "survey_prefix", None)
    if survey_prefix:
        sample_rate = getattr(args, "sample_rate", 1)
        include_nonconforming = getattr(args, "include_nonconforming_folders", False)
        sampled, report = sample_survey_folder(
            survey_prefix, sample_rate, include_nonconforming=include_nonconforming
        )
        print(report.summary())
        return sampled, dataclasses.asdict(report)
    if args.habcam:
        names = args.habcam
        if len(names) == 1 and names[0].endswith((".txt", ".csv")):
            names = read_names_from_file(names[0])
        return resolve_habcam_paths(names), None
    return list(args.images), None


def cmd_models(args: argparse.Namespace) -> None:
    definitions = load_definitions()
    for key in sorted(definitions):
        d = definitions[key]
        gpu = f"{d.gpu}x {d.gpu_type}" if d.gpu else "none"
        print(f"{key:45s} family={d.family:11s} machine={d.machine_type:16s} gpu={gpu}")


def cmd_build_input(args: argparse.Namespace) -> None:
    definition = resolve(args.model)
    image_paths, _survey_sample = _resolve_image_paths(args)

    if not image_paths:
        sys.exit("No image paths resolved -- pass --images, --habcam, or --survey-prefix.")

    manifest = input_builder.manifest_for_family(definition.family, image_paths)

    if args.upload:
        gcs.upload_json(manifest, args.upload)
        print(f"Uploaded manifest ({len(image_paths)} instances) to {args.upload}")
    else:
        print(json.dumps(manifest, indent=2))


def cmd_trigger(args: argparse.Namespace) -> None:
    definition = resolve(args.model)

    run_name = args.run_name or f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    survey_sample = None
    input_count = None

    input_file_uri = args.input_file
    staged_input_info = None
    if not input_file_uri:
        image_paths, survey_sample = _resolve_image_paths(args)
        if not image_paths:
            sys.exit("Provide --input-file, or --images/--habcam/--survey-prefix to build one.")

        if not args.no_flatten_inputs and spans_multiple_folders(image_paths):
            folder_count = len({p.rsplit("/", 1)[0] for p in image_paths})
            staging_prefix = f"gs://{config.OUTPUT_BUCKET}/{args.gcs_prefix}/staging/{run_name}/"
            print(
                f"Input spans {folder_count} GCS folders -- this is the condition that triggers "
                f"the inference wrapper's known chunking bug (see survey_sampler.py's module "
                f"docstring). Staging {len(image_paths)} file(s) into {staging_prefix} first "
                f"(server-side copy, no download/upload) so every batch the wrapper forms stays "
                f"single-folder. Pass --no-flatten-inputs to skip this."
            )
            image_paths = flatten_to_staging(image_paths, staging_prefix)
            staged_input_info = {"staging_prefix": staging_prefix, "file_count": len(image_paths)}
            print(f"Staged {len(image_paths)} file(s) -> {staging_prefix}")

        input_count = len(image_paths)
        manifest = input_builder.manifest_for_family(definition.family, image_paths)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        input_file_uri = (
            f"gs://{config.OUTPUT_BUCKET}/{args.gcs_prefix}/inputs/"
            f"{args.model}_{timestamp}.json"
        )
        gcs.upload_json(manifest, input_file_uri)
        print(f"Built and uploaded input manifest ({len(image_paths)} instances) -> {input_file_uri}")

    extra_weights = _parse_extra_weights(args.extra_weights)
    if extra_weights and not args.weights_file:
        sys.exit("--extra-weights requires --weights-file (it stages alongside the primary weights).")

    extra_weight_uris: dict = {}
    if args.weights_file:
        # Auto-stage: upload the local weights, rewrite the local YAML
        # template's `weights:` field to point at them, upload the result.
        # Replaces the old "upload weights by hand, then hand-edit the
        # YAML" workflow with one flag. Ultralytics-family configs only
        # (see config_builder.py's module docstring). --extra-weights adds
        # any further named weights files a multi-weight (cascade) model
        # needs -- see docs/two-stage-cascade.md.
        yaml_config_uri, weights_uri, yaml_config_text, extra_weight_uris = stage_config(
            local_yaml_path=args.yaml_config,
            local_weights_path=args.weights_file,
            run_name=run_name,
            gcs_prefix=args.gcs_prefix,
            extra_weights=extra_weights or None,
        )
        print(f"Staged weights {args.weights_file} -> {weights_uri}")
        for field_name, uri in extra_weight_uris.items():
            print(f"Staged {field_name} weights -> {uri}")
        print(f"Staged config {args.yaml_config} -> {yaml_config_uri}")
    else:
        yaml_config_uri = args.yaml_config
        if yaml_config_uri.startswith("gs://"):
            yaml_config_text = gcs.download_text(yaml_config_uri)
        else:
            yaml_config_text = Path(yaml_config_uri).read_text(encoding="utf-8")
            dest = f"gs://{config.OUTPUT_BUCKET}/{args.gcs_prefix}/configs/{args.model}.yaml"
            gcs.upload_file(yaml_config_uri, dest)
            print(f"Uploaded YAML config {yaml_config_uri} -> {dest}")
            yaml_config_uri = dest

    output_folder = args.output_folder or f"{args.gcs_prefix}/output/"

    conf = {
        "model_type": args.model,
        "yaml_config_path": yaml_config_uri,
        "input_file": input_file_uri,
        "output_bucket": config.OUTPUT_BUCKET,
        "output_folder": output_folder,
    }

    print("Triggering DAG with conf:")
    print(json.dumps(conf, indent=2))

    if not args.no_card:
        # Built and uploaded *before* the --dry-run check on purpose: the
        # whole point is a provenance record that exists pre-flight, so
        # `trigger ... --dry-run` alone still produces one.
        card = build_run_card(
            model_type=args.model,
            model_definition=definition,
            input_file=input_file_uri,
            yaml_config_path=yaml_config_uri,
            yaml_config_text=yaml_config_text,
            output_bucket=config.OUTPUT_BUCKET,
            output_folder=output_folder,
            gcs_prefix=args.gcs_prefix,
            input_count=input_count,
            survey_sample=survey_sample,
            extra_weights=extra_weight_uris or None,
            staged_input=staged_input_info,
        )
        card_uri = upload_run_card(card)
        print(f"Run card -> {card_uri}")

    if args.dry_run:
        print("--dry-run set: not submitting.")
        return

    print(trigger_dag(conf))
    print(list_runs())

    if args.wait:
        print(f"Waiting for a Cloud Batch job matching '{args.model}-*' to appear...")
        job_name = None
        for _ in range(20):
            job_name = find_job_by_prefix(args.model)
            if job_name:
                break
            time.sleep(15)
        if not job_name:
            sys.exit(
                "No Cloud Batch job found yet -- check `pemad-infer status` "
                "manually, submission may still be in progress."
            )
        print(f"Found job: {job_name}")
        final_state = poll_until_terminal(job_name)
        print(f"Final state: {final_state}")
        print(f"Expected output: gs://{config.OUTPUT_BUCKET}/{output_folder}")


def cmd_stage_weights(args: argparse.Namespace) -> None:
    dest = stage_weights(
        local_weights_path=args.weights_file,
        run_name=args.run_name,
        gcs_prefix=args.gcs_prefix,
        weights_filename=args.weights_filename,
    )
    print(f"Staged weights -> {dest}")


def cmd_stage_config(args: argparse.Namespace) -> None:
    extra_weights = _parse_extra_weights(args.extra_weights)
    dest, weights_uri, _yaml_text, extra_weight_uris = stage_config(
        local_yaml_path=args.yaml_config,
        local_weights_path=args.weights_file,
        run_name=args.run_name,
        gcs_prefix=args.gcs_prefix,
        config_name=args.config_name,
        extra_weights=extra_weights or None,
    )
    print(f"Staged weights -> {weights_uri}")
    for field_name, uri in extra_weight_uris.items():
        print(f"Staged {field_name} weights -> {uri}")
    print(f"Staged config -> {dest}")


def cmd_status(args: argparse.Namespace) -> None:
    if args.logical_date:
        print(task_states_for_run(args.logical_date))
    else:
        print(list_runs())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pemad-infer",
        description="Trigger and monitor NOAA Optics Cloud Batch / Airflow inference runs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_models = sub.add_parser("models", help="List models available in model_runtime_definitions.json")
    p_models.set_defaults(func=cmd_models)

    p_build = sub.add_parser("build-input", help="Build (and optionally upload) an input manifest JSON")
    p_build.add_argument("--model", required=True, help="model_type key (determines manifest schema)")
    p_build.add_argument("--images", nargs="*", default=[], help="gs:// image (or video) paths")
    p_build.add_argument(
        "--habcam", nargs="*", default=[],
        help="HabCam filenames, or a single .txt/.csv file of filenames (one per line)",
    )
    p_build.add_argument(
        "--survey-prefix",
        help="gs:// folder prefix to sample images from, e.g. "
             "'gs://nmfs-dev-uc1-landing-bucket/NEFSC/HabCam Survey/habcam/proc/Images/2023/' "
             "(sampled per leaf folder -- see --sample-rate)",
    )
    p_build.add_argument(
        "--sample-rate", type=int, default=1,
        help="With --survey-prefix: take 1 image out of every N within each leaf folder "
             "(e.g. 5 = 1-in-5, 500 = 1-in-500). Default 1 = every image.",
    )
    p_build.add_argument(
        "--include-nonconforming-folders", action="store_true",
        help="With --survey-prefix: also sample from leaf folders that don't match the standard "
             "<YYYYMMDD>_<HHMM> HabCam naming convention (e.g. a flat 'auv' folder). Excluded by "
             "default and reported instead -- see the printed SampleReport summary and "
             "survey_sampler.py's module docstring. Only pass this once you've confirmed a given "
             "anomalous folder's images are safe to include.",
    )
    p_build.add_argument("--upload", help="gs:// URI to upload the manifest to (otherwise prints to stdout)")
    p_build.set_defaults(func=cmd_build_input)

    p_trigger = sub.add_parser("trigger", help="Trigger a DAG run")
    p_trigger.add_argument("--model", required=True)
    p_trigger.add_argument("--input-file", help="Existing gs:// input manifest (skips building one)")
    p_trigger.add_argument("--images", nargs="*", default=[])
    p_trigger.add_argument("--habcam", nargs="*", default=[])
    p_trigger.add_argument(
        "--survey-prefix",
        help="gs:// folder prefix to sample images from (sampled per leaf folder -- see --sample-rate)",
    )
    p_trigger.add_argument(
        "--sample-rate", type=int, default=1,
        help="With --survey-prefix: take 1 image out of every N within each leaf folder. Default 1 = every image.",
    )
    p_trigger.add_argument(
        "--include-nonconforming-folders", action="store_true",
        help="With --survey-prefix: also sample from leaf folders that don't match the standard "
             "<YYYYMMDD>_<HHMM> HabCam naming convention (e.g. a flat 'auv' folder). Excluded by "
             "default and reported instead -- see the printed SampleReport summary. Only pass this "
             "once you've confirmed a given anomalous folder's images are safe to include.",
    )
    p_trigger.add_argument(
        "--yaml-config", required=True,
        help="gs:// URI or local path to the pipeline YAML. With --weights-file, this is treated as a "
             "local template whose `weights:` field gets rewritten automatically.",
    )
    p_trigger.add_argument(
        "--weights-file",
        help="Local weights file (e.g. best.pt) -- when given, --yaml-config is staged as a local "
             "template: weights are uploaded and the config's `weights:` field is rewritten to point "
             "at them automatically, instead of doing both by hand. Ultralytics-family models only.",
    )
    p_trigger.add_argument(
        "--extra-weights",
        action="append",
        metavar="FIELD=PATH",
        help="Additional local weights file to stage and wire into the YAML template, for "
             "multi-weight (cascade) models -- e.g. a two-stage detector+classifier model "
             "(see docs/two-stage-cascade.md). Repeatable, one FIELD=PATH per weights file, "
             "e.g. --extra-weights classifier_weights=~/models/crabdata_tax.pt stages that "
             "file and rewrites a top-level `classifier_weights:` field in the YAML, the same "
             "way --weights-file does for `weights:`. FIELD must match whatever the model's "
             "container config actually reads -- confirm against the container. Requires "
             "--weights-file.",
    )
    p_trigger.add_argument(
        "--run-name",
        help="Run name used for staged weights/config paths and the run-card filename "
             "(default: '<model>_<UTC timestamp>')",
    )
    p_trigger.add_argument(
        "--output-folder",
        help="Destination folder within the fixed output bucket (default: '<gcs-prefix>/output/')",
    )
    p_trigger.add_argument(
        "--gcs-prefix", required=True,
        help="Folder prefix used for auto-built inputs/configs/weights/output/run_cards, e.g. your username",
    )
    p_trigger.add_argument("--wait", action="store_true", help="Block and poll until the Cloud Batch job reaches a terminal state")
    p_trigger.add_argument("--dry-run", action="store_true", help="Print the assembled conf without submitting")
    p_trigger.add_argument(
        "--no-card", action="store_true",
        help="Skip building/uploading a run card. By default a run card is written before every "
             "trigger -- including under --dry-run -- so it exists pre-flight.",
    )
    p_trigger.add_argument(
        "--no-flatten-inputs", action="store_true",
        help="Skip the automatic input-flattening safeguard. By default, when the resolved image "
             "list spans more than one GCS folder, pemad-infer copies every image into one flat "
             "staging folder before triggering (see input_staging.py) -- this sidesteps a known "
             "bug in the inference wrapper where a batch spanning multiple folders crashes with "
             "FileNotFoundError. Pass this flag to submit the original (unstaged) paths instead.",
    )
    p_trigger.set_defaults(func=cmd_trigger)

    p_stage_weights = sub.add_parser(
        "stage-weights", help="Upload a local weights file to GCS using the standard layout"
    )
    p_stage_weights.add_argument("--weights-file", required=True, help="Local path to the weights file (e.g. best.pt)")
    p_stage_weights.add_argument("--run-name", required=True, help="Run name -- becomes part of the staged GCS path")
    p_stage_weights.add_argument("--gcs-prefix", required=True, help="Folder prefix, e.g. your username")
    p_stage_weights.add_argument("--weights-filename", help="Override the uploaded filename (default: local filename)")
    p_stage_weights.set_defaults(func=cmd_stage_weights)

    p_stage_config = sub.add_parser(
        "stage-config",
        help="Upload local weights + a local YAML template, with the template's `weights:` field "
             "rewritten to point at the uploaded weights automatically",
    )
    p_stage_config.add_argument("--yaml-config", required=True, help="Local path to the pipeline YAML template")
    p_stage_config.add_argument("--weights-file", required=True, help="Local path to the weights file")
    p_stage_config.add_argument("--run-name", required=True, help="Run name -- becomes part of the staged GCS paths")
    p_stage_config.add_argument("--gcs-prefix", required=True, help="Folder prefix, e.g. your username")
    p_stage_config.add_argument("--config-name", help="Override the uploaded config filename (default: '<run-name>.yaml')")
    p_stage_config.add_argument(
        "--extra-weights",
        action="append",
        metavar="FIELD=PATH",
        help="Additional local weights file to stage and wire into the YAML template, for "
             "multi-weight (cascade) models. Repeatable -- see `trigger --extra-weights` help "
             "and docs/two-stage-cascade.md.",
    )
    p_stage_config.set_defaults(func=cmd_stage_config)

    p_status = sub.add_parser("status", help="Check DAG run / task status")
    p_status.add_argument(
        "--logical-date",
        help="e.g. '2026-09-17T13:04:20+00:00' -- omit to list recent runs instead",
    )
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
