# Architecture

Condensed from the NOAA Optics documentation set (Developer Guide, User Guide,
Model Zoo build doc, VIAME onboarding doc) plus what was actually confirmed
by testing against `composer-env1`. Full research notes live in the
"GCP administration" Claude Project, doc `airflow-gcp-inference-workflow-notes.md`.

## One mental model

Airflow (Cloud Composer) never touches image data directly -- it only
orchestrates. A DAG run walks four tasks:

```
generate_run_id -> load_pipeline_metadata -> submit_cloud_batch_job -> wait_for_cloud_batch_job
```

- **Airflow/Composer** (`composer-env1`): orchestration + monitoring only.
- **Cloud Batch**: provisions the VM/container and actually runs the model,
  under service account `batch-sa1@ggn-nmfs-osi-dev-1.iam.gserviceaccount.com`
  -- a different identity from your own user account. A failure inside the
  container (bad GCS path, model crash) is a `batch-sa1` permissions/config
  problem, not necessarily yours.
- **GCS**: config + data storage. Notably:
  - `gs://ggn-nmfs-osi-dev-1-data/configs/model_runtime_definitions.json` --
    the authoritative, decoupled-from-DAG-code registry of every model
    (`model_registry.py` reads this directly).
  - `gs://ggn-nmfs-osi-dev-1-data/configs/archive/<run_id>.yaml` -- every
    run's exact runtime config, archived after completion.
- **Artifact Registry** (`nmfs-dev-uc1-docker-repository`): stores inference
  container images.

## DAG params (ground truth, from the actual DAG source)

- `model_type` -- must be a key in `model_runtime_definitions.json`
- `yaml_config_path` -- full `gs://` path to a pipeline YAML
- `input_file` -- full `gs://` path to an input manifest JSON
- `output_bucket` -- **hard-coded to `ggn-nmfs-osi-dev-1-data` by the DAG's
  own Param definition** ("Fixed bucket due to permissions. Please do not
  change.") -- this is a DAG design constraint, not a permissions bug
- `output_folder` -- destination folder within that bucket

## Model family inference

`model_runtime_definitions.json` has no explicit "family" field. The DAG's
`submit_cloud_batch_job` task treats any entry with `"command": []` as a
VIAME-style shell-args container, and everything else as an
Ultralytics-style `python <script>` container. `model_registry.ModelDefinition.family`
reuses that exact heuristic.

## Debugging model

- Task-level failure at `submit_cloud_batch_job` -> your own IAM, or the
  Composer environment's own identity.
- Task-level failure at `wait_for_cloud_batch_job`, or a Cloud Batch job
  that reaches `FAILED` -- almost always `batch-sa1`'s permissions or the
  container/model itself. Check Cloud Batch job status events first, then
  Cloud Logging filtered by `job_name` (`<model_type>-<run_id[:8]>`) for
  the container's stack trace.
- `getIamPolicy` calls (on a bucket, a repo) are a *different* permission
  class than actually *using* that resource (`get`/`list`/`create` on
  objects). A `getIamPolicy` denial does not mean you can't use the
  resource -- test usage directly instead of chasing policy-read grants.

## Permission model

See `docs/permissions.md`.

## Multi-model cascades (e.g. detector -> taxonomic classifier)

See `docs/two-stage-cascade.md` for the full writeup. Short version:
cascades are built as a **single custom container** (both models loaded
and run internally, registered as one `model_type`) rather than a new
multi-task Airflow DAG -- decided 2026-09-17 on the basis that the
detector and classifier stages were confirmed to need similar compute, so
there's no efficiency reason to split them into separately-resourced
Cloud Batch jobs. This means cascades need zero changes to the DAG or
this CLI; only a new Docker image + `model_runtime_definitions.json`
entry. Revisit a real two-stage DAG only if a future cascade's stages
genuinely need different hardware. A separate org-wide effort (JIRA
OSI-129) is standardizing a similar single-image contract for *all*
future Ultralytics models -- worth building the cascade container against
that once it's finalized, rather than against `hello-world`'s current
ad hoc shape. See the Claude Project notes for the full OSI-129 writeup.

## Sampling a survey folder

`survey_sampler.py` (wired into `build-input`/`trigger` via
`--survey-prefix`/`--sample-rate`) takes 1-in-N images from a GCS folder
prefix, sampled independently within each leaf folder rather than across
one combined listing -- keeps the sample spread evenly across the whole
time range instead of being dominated by whichever folder has the most
images. It lists the *entire* prefix before sampling (GCS has no
server-side "every Nth object" listing), so always try `--dry-run` first
on a large prefix (e.g. a whole survey year) to see the folder/image
counts before committing to a real trigger. By default it also excludes
leaf folders that don't match the standard HabCam `<YYYYMMDD>_<HHMM>`
naming (e.g. a flat `auv` folder) -- pass `--include-nonconforming-folders`
once a specific one is confirmed safe. See "Known limitations" below for
the separate, more serious issue this sampling can trigger downstream.

## Weight staging, config auto-writing, and run cards

Three additions on top of the base trigger flow, all in service of the
same goal -- fewer manual GCS/YAML steps, and a pre-flight record of what
a run actually is before it's submitted:

- **`config_builder.py`** (`stage_weights`, `stage_config`,
  `rewrite_yaml_field`): given a local weights file and a local
  Ultralytics-style YAML template, uploads the weights to
  `gs://<bucket>/<gcs_prefix>/weights/<run_name>/weights/<filename>`,
  rewrites the template's `weights:` line to point at that path (a
  targeted line-level regex substitution, not a full YAML parse/dump --
  this preserves comments, like the alternate-weights lines commented out
  in `configs/ultralytics/24star.yaml`), and uploads the result to
  `gs://<bucket>/<gcs_prefix>/configs/<run_name>.yaml`. Wired into `cli.py`
  as `trigger --weights-file ... --run-name ...` and the standalone
  `stage-weights`/`stage-config` subcommands. A repeatable
  `--extra-weights FIELD=PATH` handles multi-weight/cascade models the
  same way. VIAME-family configs have no `weights:` field (weights are
  baked into the Docker image at build time), so this only applies to
  Ultralytics-style models.
- **`run_card.py`** (`build_run_card`, `upload_run_card`): every `trigger`
  call -- including under `--dry-run` -- writes a JSON run card to
  `gs://<bucket>/<gcs_prefix>/run_cards/<model_type>_<timestamp>.json`
  before anything is submitted. It bundles the resolved `ModelDefinition`
  (including the inferred `family`), the input manifest path and image
  count, the exact YAML config text used (post-rewrite, if
  `--weights-file` was used), the survey-sample report (including
  `skipped_folders` and `sampled_bytes`) if `--survey-prefix` was used,
  any staged `extra_weights`, and a UTC creation timestamp. Deliberately
  independent of Airflow's own archived-runtime-YAML
  (`gs://.../configs/archive/<run_id>.yaml`), which only exists *after* a
  run is submitted and only captures the YAML, not the model definition or
  how the input list was built. `--no-card` opts out for a given run.

## Airflow/Cloud Batch vs. running inference directly on a workstation

Worth being explicit about, since it's not obvious from the CLI output
alone (e.g. `pemad-infer models` showing every Ultralytics entry as
`gpu=none` today):

**Ultralytics GPU support is not structurally blocked** -- it's just that
none of the currently-registered Ultralytics entries in
`model_runtime_definitions.json` are configured for one yet. Enabling it
is two independent, per-entry/per-config choices, not a framework
limitation: (1) set `gpu`, `gpu_type`, and an appropriate `machine_type`
(e.g. `g2-standard-4`) on that model's entry in
`model_runtime_definitions.json`, and (2) set `payload.device` (e.g.
`cuda:0`) in the pipeline YAML instead of `cpu`. The existing entries are
simply provisioned CPU-only -- plausibly because `yolo12n` is light enough
to run acceptably on CPU, and/or conservative GPU-quota allocation -- not
because Ultralytics-in-this-pipeline can't use a GPU. The VIAME entries
already prove GPU scheduling works end-to-end on this same Cloud Batch
path.

As for the DAG's value over just running inference on a workstation
directly: Cloud Batch jobs can run far longer than anyone wants a laptop
tied up for (registered timeouts up to ~4 days), the VM is ephemeral --
paying only while a job actually runs, rather than an idle GPU workstation
sitting provisioned -- and every run now produces a run card plus
Airflow's own archived config, giving a reproducible, team-shared,
audit-trailable record of exactly what model/input/config produced a
given output. The real overhead is genuine too: Cloud Batch VM
provisioning adds a startup lag (order of minutes) that a local process
doesn't have, and the permission/JIRA path (see `docs/permissions.md`) is
more friction than running something locally. In practice: local
`docker run` (per `docs/byom-docker-onboarding.md`'s own recommended
workflow) is still the right tool for fast dev iteration on a single
image or two; the DAG is for scaled, shared, or reproducible production
runs, not a replacement for that local loop.

## Known limitations

- `find_job_by_prefix` (in `batch_monitor.py`) matches Cloud Batch jobs by
  name prefix + most-recent-created, not by the run's actual `run_id`. Two
  concurrent runs of the same model can confuse it. Fine for solo/serial
  testing.
- `airflow_client.py` shells out to `gcloud composer environments run`
  rather than using the native Cloud Composer API client. Deliberate --
  this is the exact command sequence verified live, and it keeps the
  dependency surface small. A native-client rewrite is a reasonable later
  cleanup, not a correctness fix.
- `habcam_paths.py`'s dated-subfolder layout hasn't been confirmed against
  a real production image path -- the one successful test so far used an
  image placed at the landing bucket's root, not under the computed
  subfolder structure. Confirm the real layout before trusting this for
  anything beyond another plumbing test.
- Only single-camera image-list manifests are wired into the CLI today.
  `input_builder.py` has VIAME video/stereo manifest builders already
  written, just not yet exposed as CLI flags.
- **Survey-prefix sampling excludes non-conforming leaf folders by default
  (added 2026-09-17) -- good hygiene, but NOT related to the chunking bug
  below.** A flat `auv` folder and `<date>_orig` variant folders (both
  structurally different from the documented `<YYYYMMDD>_<HHMM>` leaf
  naming) were seen in early sample reports. `sample_survey_folder` now
  partitions folders by `is_conforming_leaf_folder` and excludes
  non-conforming ones by default, reporting them in
  `SampleReport.skipped_folders`. Pass `--include-nonconforming-folders`
  once a specific folder is confirmed safe.
- **Known bug, root cause confirmed 2026-09-21: a multi-folder input list
  can crash the inference container partway through, even though the
  images downloaded successfully.** This is a bug inside
  `ultralytics-fish-segmentation-bw-wrapper` itself (specifically its
  `app_utils/gcs_utils.py:download_uri_list_to_local`), not in this repo
  -- documented here because it changes what input lists are safe to
  submit. The wrapper downloads images in ~500MB batches
  (`MAX_CHUNK_BYTES`, via `app_utils/process_dataset_pipeline.py`) and,
  for each batch, tries to compute a single shared folder that all of
  that batch's images live under (a "common prefix"). If a batch's images
  come from **more than one GCS folder**, that computation gives up and
  points the model at the download's top-level temp directory instead --
  but the images were actually saved nested under their original
  per-image subfolders, not flattened into that top level. So the model
  looks in a directory that (from its point of view) has no images in
  it, even though they're sitting one or more folders below, and crashes
  with `FileNotFoundError: No images or videos found in ...`. Confirmed
  directly from the wrapper's own source (pulled via `docker create` +
  `docker cp` from the live image, not guessed from logs).
  This supersedes an earlier, incorrect guess (a timing/race-condition
  theory, based on log timestamps alone) -- the real bug is a path
  computed from file *locations*, not a timing issue, and the fix on the
  wrapper's side would be straightforward: stop trying to compute a
  shared folder at all, and just download every batch's images flattened
  directly into one temp folder (no nested subfolders to lose track of).
  **Practical effect for this CLI**: a batch is safe if every image in it
  comes from the same GCS folder, *or* if the whole batch is small enough
  to land in a single ~500MB download batch on the wrapper's side AND
  still shares one folder. A `--survey-prefix` sample spanning many leaf
  folders is the highest-risk case, since the wrapper will eventually hit
  a batch boundary that crosses two folders no matter how the sample is
  sized. `sample_survey_folder`'s existing `sampled_bytes` warning (stay
  under ~500MB) is necessary but not sufficient on its own now -- see
  `survey_sampler.py`'s module docstring and the Claude Project notes for
  the fuller history and what's still open here.
- **Update 2026-09-21 -- client-side mitigation shipped.** `pemad-infer
  trigger` now automatically copies a multi-folder image list into one
  flat, run-specific GCS staging folder before submitting (a server-side
  `copy_blob`, no download/upload/egress cost) -- see
  `input_staging.py`. Since every file then lives in exactly one folder,
  the wrapper can never form a batch that crosses two folders, regardless
  of run size or `--sample-rate`. This removes the failure mode above for
  any run triggered through this CLI, without needing a fix from whoever
  owns the wrapper image. Pass `--no-flatten-inputs` to submit the
  original (unstaged) paths instead. This is a workaround, not a fix --
  the wrapper itself still has the bug, and any other consumer of that
  image is still exposed to it.
