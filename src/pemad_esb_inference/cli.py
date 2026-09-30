"""Command-line entry point: `pemad-infer`."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

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

    if args.combined_image_contract:
        if not args.output_file:
            sys.exit("--combined-image-contract requires --output-file (a gs:// URI).")
        combined_config = {}
        if args.yaml_config:
            text = (
                gcs.download_text(args.yaml_config)
                if args.yaml_config.startswith("gs://")
                else Path(args.yaml_config).read_text(encoding="utf-8")
            )
            combined_config = yaml.safe_load(text) or {}
        combined_config["stereo_side"] = args.stereo_side
        manifest = input_builder.combined_contract_manifest(
            image_paths, args.output_file, combined_config
        )
    else:
        manifest = input_builder.manifest_for_family(definition.family, image_paths)

    if args.upload:
        gcs.upload_json(manifest, args.upload)
        print(f"Uploaded manifest ({len(image_paths)} instances) to {args.upload}")
    else:
        print(json.dumps(manifest, indent=2))


def cmd_trigger(args: argparse.Namespace) -> None:
    definition = resolve(args.model)

    run_name = args.run_name or f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    if args.stereo_side != "full" and not args.combined_image_contract:
        sys.exit(
            "--stereo-side left/right requires --combined-image-contract -- only that "
            "model.py contract reads the stereo_side config key. Pass --stereo-side full "
            "(the default) if this model doesn't need stereo splitting."
        )

    survey_sample = None
    input_count = None

    input_file_uri = args.input_file
    staged_input_info = None
    resolved_image_paths = None
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

        if args.combined_image_contract:
            # Built further below instead, once the (possibly staged/
            # rewritten) yaml config is known -- the combined-image
            # contract embeds that config inline per instance, so the
            # manifest can't be finalized until after the weights/yaml
            # staging block runs. See input_builder.combined_contract_manifest.
            resolved_image_paths = image_paths
        else:
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
            dedupe_weights=not args.no_dedupe_weights,
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
    output_file = None

    if args.combined_image_contract and resolved_image_paths is not None:
        combined_config = yaml.safe_load(yaml_config_text) or {}
        combined_config["stereo_side"] = args.stereo_side
        # Run-specific filename (nested under run_name) instead of a fixed
        # "annotations.json" -- the old fixed name meant every run using the
        # same --output-folder silently overwrote the previous run's output.
        output_file = (
            f"gs://{config.OUTPUT_BUCKET}/{output_folder.rstrip('/')}/"
            f"{run_name}/annotations.json"
        )
        manifest = input_builder.combined_contract_manifest(
            resolved_image_paths, output_file, combined_config
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        input_file_uri = (
            f"gs://{config.OUTPUT_BUCKET}/{args.gcs_prefix}/inputs/"
            f"{args.model}_{timestamp}.json"
        )
        gcs.upload_json(manifest, input_file_uri)
        print(
            f"Built and uploaded combined-image-contract input manifest "
            f"({len(resolved_image_paths)} files, 1 instance, output_file={output_file}) "
            f"-> {input_file_uri}"
        )

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
            output_file=output_file,
            stereo_side=args.stereo_side if args.combined_image_contract else None,
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
        if output_file:
            print(f"Expected output: {output_file}")
        else:
            print(f"Expected output: gs://{config.OUTPUT_BUCKET}/{output_folder}")


def cmd_stage_weights(args: argparse.Namespace) -> None:
    dest = stage_weights(
        local_weights_path=args.weights_file,
        run_name=args.run_name,
        gcs_prefix=args.gcs_prefix,
        weights_filename=args.weights_filename,
        dedupe=not args.no_dedupe_weights,
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
        dedupe_weights=not args.no_dedupe_weights,
    )
    print(f"Staged weights -> {weights_uri}")
    for field_name, uri in extra_weight_uris.items():
        print(f"Staged {field_name} weights -> {uri}")
    print(f"Staged config -> {dest}")


def cmd_stage_cleanup(args: argparse.Namespace) -> None:
    """List (default) or delete (--force) old staging/ run folders.

    Every trigger that spans multiple GCS folders copies its inputs
    into a fresh gs://<bucket>/<gcs_prefix>/staging/<run_name>/ folder
    (see input_staging.py); nothing ever removes those folders on its
    own, so they grow without bound as more runs are triggered. This
    groups objects under staging/ by run folder, reports total size,
    and -- only with --force -- deletes the folders whose newest file
    is older than --older-than-days. Recent folders are always left
    alone (by design: staged inputs double as a fixed candidate set
    for comparing model output across runs, so this only reclaims
    space from runs old enough that nobody is still comparing against
    them).
    """
    from . import gcs

    prefix = f"gs://{config.OUTPUT_BUCKET}/{args.gcs_prefix}/staging/"
    entries = gcs.list_prefix_with_metadata(prefix)
    if not entries:
        print(f"No staged files under {prefix}")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.older_than_days)

    runs: dict = {}
    for uri, size, created in entries:
        after = uri.split("/staging/", 1)[1]
        run_folder = after.split("/", 1)[0]
        entry = runs.setdefault(run_folder, {"size": 0, "count": 0, "newest": created})
        entry["size"] += size
        entry["count"] += 1
        if created and (entry["newest"] is None or created > entry["newest"]):
            entry["newest"] = created

    total_bytes = sum(info["size"] for info in runs.values())
    print(f"{len(runs)} staged run folder(s) under {prefix} ({total_bytes / 1e9:.2f} GB total)")

    stale = {
        run: info for run, info in runs.items()
        if info["newest"] is not None and info["newest"] < cutoff
    }
    stale_bytes = sum(info["size"] for info in stale.values())
    print(f"{len(stale)} folder(s) older than {args.older_than_days} day(s) ({stale_bytes / 1e9:.2f} GB):")
    for run, info in sorted(stale.items(), key=lambda kv: kv[1]["newest"]):
        print(
            f"  {run}  {info['count']} file(s)  {info['size'] / 1e6:.1f} MB  "
            f"staged {info['newest']:%Y-%m-%d}"
        )

    if not stale:
        return
    if not args.force:
        print("\nDry run -- pass --force to actually delete the folder(s) listed above.")
        return

    for run in stale:
        run_prefix = f"{prefix}{run}/"
        deleted = gcs.delete_prefix(run_prefix)
        print(f"Deleted {deleted} object(s) under {run_prefix}")



def cmd_export_viewer_assets(args: argparse.Namespace) -> None:
    """Download a completed run's original frames and write per-detection
    crop thumbnails + a manifest.json for the results viewer. See
    viewer_export.py's module docstring for why this needs the run card
    (not just annotations.json) and how the stereo crop is re-derived."""
    from .viewer_export import export

    result = export(
        run_card_uri=args.run_card,
        output_dir=args.output_dir,
        max_context_dim=args.max_context_dim,
        max_crop_dim=args.max_crop_dim,
        crop_padding_pct=args.crop_padding_pct,
        make_zip=not args.no_zip,
    )
    print(
        f"\nWrote {result['image_count']} context thumbnail(s) and "
        f"{result['annotation_count']} crop(s) to {result['output_dir']}"
    )
    print(f"Manifest: {result['manifest_path']}")
    if result["skipped_images"]:
        preview = ", ".join(result["skipped_images"][:10])
        more = " ..." if len(result["skipped_images"]) > 10 else ""
        print(
            f"WARNING: {len(result['skipped_images'])} image(s) referenced in annotations.json "
            f"had no matching entry in the input manifest and were skipped: {preview}{more}"
        )
    if result["zip_path"]:
        print(f"Zipped: {result['zip_path']} -- upload this to get the assets off this machine.")



def cmd_ingest_run(args: argparse.Namespace) -> None:
    """Load a completed export-viewer-assets export into Postgres for
    the results-auditor app. See db_ingest.py's module docstring and
    db/schema.sql."""
    import json
    from pathlib import Path as _Path
    from .db_ingest import build_rows, load_into_postgres, upload_assets

    assets_dir = _Path(args.assets_dir)
    manifest = json.loads((assets_dir / "manifest.json").read_text())

    if not args.skip_upload and not args.gcs_asset_prefix:
        sys.exit("ingest-run: --gcs-asset-prefix is required unless --skip-upload is set")

    asset_uri_prefix = None
    if not args.skip_upload:
        asset_uri_prefix = upload_assets(str(assets_dir), args.gcs_asset_prefix)
    else:
        print("[ingest-run] --skip-upload set: storing local relative thumb paths, not gs:// URIs.")

    rows = build_rows(manifest, asset_uri_prefix)
    run_id = load_into_postgres(rows, args.db_dsn)
    print(
        f"[ingest-run] Loaded run {run_id}: {len(rows['images'])} image(s), "
        f"{len(rows['detections'])} detection(s)."
    )


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
    p_build.add_argument(
        "--combined-image-contract", action="store_true",
        help="Build the manifest shape needed by models on the newer combined-image contract "
             "(app.py + model.py + inference_runner.py all in one image, e.g. the star-cascade "
             "container forked from optics-models-ultralytics-detection) instead of the "
             "family-based flat/VIAME shapes -- see input_builder.py's module docstring. "
             "Requires --output-file; --yaml-config optionally supplies the inline per-instance "
             "config (weights/classifier_weights/taxonomy_json/payload/...).",
    )
    p_build.add_argument(
        "--output-file",
        help="With --combined-image-contract: the exact gs:// URI the container will write its "
             "results to (this contract's app.py writes to one explicit file, not just a folder).",
    )
    p_build.add_argument(
        "--yaml-config",
        help="With --combined-image-contract: gs:// URI or local path to a YAML file parsed and "
             "embedded as each instance's inline 'config' dict.",
    )
    p_build.add_argument(
        "--stereo-side", choices=["full", "left", "right"], default="full",
        help="With --combined-image-contract: crop each image to one half of a spliced "
             "left+right stereo frame before it's embedded in the manifest's config -- the "
             "combined-image-contract model.py (e.g. star-cascade) reads this and splits "
             "before running the detector, so a real organism visible in both eyes' overlap "
             "zone isn't detected/classified twice. Default 'full' (no split, the original "
             "behavior) for backward compatibility.",
    )
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
    p_trigger.add_argument(
        "--no-dedupe-weights", action="store_true",
        help="Skip content-addressed weights staging. By default, staged weights "
             "(--weights-file / --extra-weights) are hashed and uploaded to a shared, "
             "run-independent gs://.../weights/by-hash/<sha256>/ path -- if this exact "
             "file was already staged by any prior run, the upload is skipped instead of "
             "creating another full copy. Pass this flag to always upload to a fresh, "
             "run-scoped path instead (the old behavior).",
    )
    p_trigger.add_argument(
        "--combined-image-contract", action="store_true",
        help="This model's container is built on the newer combined-image contract (app.py + "
             "model.py + inference_runner.py all in one image, e.g. star-cascade, forked from "
             "optics-models-ultralytics-detection) rather than the older YAML_CONFIG_PATH-plus-"
             "flat-instances-list wrapper contract. Changes the input manifest to one instance "
             "with input_files/output_file/config embedded inline (built from --yaml-config after "
             "weights staging) instead of a flat instances-list -- required for this model's "
             "inference_runner.py, which never reads YAML_CONFIG_PATH at all. Confirmed 2026-09-30 "
             "after every instance in a real run failed with \"'str' object has no attribute "
             "'get'\" using the old flat manifest shape. See input_builder.py's module docstring "
             "for the full contract difference.",
    )
    p_trigger.add_argument(
        "--stereo-side", choices=["full", "left", "right"], default="full",
        help="Crop each image to one half of a spliced left+right stereo frame before the "
             "combined-image-contract model.py splits it off to the detector -- fixes a real "
             "organism visible in both eyes' overlap zone being detected/classified twice "
             "(once per eye). Requires --combined-image-contract. Default 'full' (no split, "
             "the original behavior) for backward compatibility. Confirmed 2026-09-30 as the "
             "root fix for double-counted star-cascade detections on HabCam stereo frames.",
    )
    p_trigger.set_defaults(func=cmd_trigger)

    p_stage_weights = sub.add_parser(
        "stage-weights", help="Upload a local weights file to GCS using the standard layout"
    )
    p_stage_weights.add_argument("--weights-file", required=True, help="Local path to the weights file (e.g. best.pt)")
    p_stage_weights.add_argument("--run-name", required=True, help="Run name -- becomes part of the staged GCS path")
    p_stage_weights.add_argument("--gcs-prefix", required=True, help="Folder prefix, e.g. your username")
    p_stage_weights.add_argument("--weights-filename", help="Override the uploaded filename (default: local filename)")
    p_stage_weights.add_argument(
        "--no-dedupe-weights", action="store_true",
        help="Skip content-addressed staging and always upload to a fresh, run-scoped "
             "path, even if this exact file is already staged elsewhere. See `trigger "
             "--no-dedupe-weights` help.",
    )
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
    p_stage_config.add_argument(
        "--no-dedupe-weights", action="store_true",
        help="Skip content-addressed staging and always upload to a fresh, run-scoped "
             "path, even if this exact file is already staged elsewhere. See `trigger "
             "--no-dedupe-weights` help.",
    )
    p_stage_config.set_defaults(func=cmd_stage_config)

    p_stage_cleanup = sub.add_parser(
        "stage-cleanup",
        help="Report (and optionally delete) old run folders under the staging/ prefix",
    )
    p_stage_cleanup.add_argument("--gcs-prefix", required=True, help="Folder prefix, e.g. your username")
    p_stage_cleanup.add_argument(
        "--older-than-days", type=int, default=30,
        help="Only report/delete staging run folders whose newest file is older than this "
             "many days (default: 30). Folders newer than this are always left alone.",
    )
    p_stage_cleanup.add_argument(
        "--force", action="store_true",
        help="Actually delete the eligible folders. Without this flag, only reports what "
             "would be deleted (dry run, the default).",
    )
    p_stage_cleanup.set_defaults(func=cmd_stage_cleanup)

    p_export_viewer = sub.add_parser(
        "export-viewer-assets",
        help="Download a completed run's original frames and write per-detection crop "
             "thumbnails + a manifest.json for the results viewer",
    )
    p_export_viewer.add_argument(
        "--run-card", required=True,
        help="gs:// path to the run card uploaded at trigger time (list "
             "gs://<bucket>/<gcs-prefix>/run_cards/ to find it)",
    )
    p_export_viewer.add_argument(
        "--output-dir", required=True,
        help="Local directory to write context/, crops/, and manifest.json into",
    )
    p_export_viewer.add_argument(
        "--max-context-dim", type=int, default=900,
        help="Cap the longer side of each full-frame context thumbnail, in pixels (default: 900)",
    )
    p_export_viewer.add_argument(
        "--max-crop-dim", type=int, default=320,
        help="Cap the longer side of each per-detection crop thumbnail, in pixels (default: 320)",
    )
    p_export_viewer.add_argument(
        "--crop-padding-pct", type=float, default=15.0,
        help="Extra context padded around each bbox crop, as a percent of the box's own "
             "width/height (default: 15)",
    )
    p_export_viewer.add_argument(
        "--no-zip", action="store_true",
        help="Skip zipping output-dir into a single archive (zipped by default, for easy upload)",
    )
    p_export_viewer.set_defaults(func=cmd_export_viewer_assets)

    p_ingest_run = sub.add_parser(
        "ingest-run",
        help="Load a completed export-viewer-assets export into Postgres for the results-auditor app",
    )
    p_ingest_run.add_argument(
        "--assets-dir", required=True,
        help="The --output-dir from a prior export-viewer-assets run (contains manifest.json, context/, crops/)",
    )
    p_ingest_run.add_argument(
        "--db-dsn", required=True,
        help="Postgres connection string, e.g. postgresql://user:pass@localhost:5432/viewer_dev",
    )
    p_ingest_run.add_argument(
        "--gcs-asset-prefix", default=None,
        help="GCS folder prefix to upload context/crops thumbnails to (e.g. jeremy/viewer_assets/<run-name>); "
             "required unless --skip-upload is set",
    )
    p_ingest_run.add_argument(
        "--skip-upload", action="store_true",
        help="Don't upload thumbnails to GCS -- store the export's local relative paths in the DB instead. "
             "Only useful for inspecting the schema/ingestion locally before wiring up real asset hosting.",
    )
    p_ingest_run.set_defaults(func=cmd_ingest_run)

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
