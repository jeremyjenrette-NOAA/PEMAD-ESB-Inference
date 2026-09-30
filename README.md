# PEMAD-ESB-Inference

CLI for triggering and monitoring inference runs on the NOAA Optics Cloud
Batch / Airflow pipeline (`composer-env1`, project `ggn-nmfs-osi-dev-1`)
from any workstation. Given a model already registered in
`model_runtime_definitions.json` and a list of images, it builds the
correct input manifest, uploads what needs uploading, triggers the DAG
headlessly, and (optionally) polls the resulting Cloud Batch job to
completion.

**Scope:** this repo triggers and monitors **already-registered** models.
Building/registering a brand-new model container (Docker, Artifact
Registry, editing `model_runtime_definitions.json`) is a separate
workflow -- see `docs/byom-docker-onboarding.md` for reference notes on
that; it's not implemented here.

## Install

Requires `gcloud` already authenticated
(`gcloud auth application-default login`) and, for headless DAG
triggering, the `composer.environments.executeAirflowCommand` permission
on `composer-env1` -- see `docs/permissions.md` if you hit a
`PERMISSION_DENIED` here.

```bash
git clone <this-repo>
cd PEMAD-ESB-Inference
pip install -e .
```

## Quickstart

List models currently registered in the pipeline:
```bash
pemad-infer models
```

Trigger a run against a known model with an existing input manifest and
config (this reproduces the exact payload validated end-to-end against
`composer-env1` on 2026-09-17):
```bash
pemad-infer trigger \
  --model ultralytics \
  --yaml-config gs://ggn-nmfs-osi-dev-1-data/jeremy/configs/ultralytics_config.yaml \
  --input-file gs://ggn-nmfs-osi-dev-1-data/jeremy/inputs/starfish_20260806_182415.json \
  --output-folder "jeremy/output/test-$(date +%Y%m%d-%H%M%S)/" \
  --gcs-prefix jeremy \
  --wait
```
(also available as `examples/trigger_ultralytics_starfish.sh`)

Or build the input manifest from a raw image list and a local YAML config
in one shot:
```bash
pemad-infer trigger \
  --model ultralytics \
  --images gs://nmfs-dev-uc1-landing-bucket/202403.20240508.142352788.7482.png gs://nmfs-dev-uc1-landing-bucket/seal_test.jpg \
  --yaml-config configs/ultralytics/24star.yaml \
  --gcs-prefix jeremy \
  --wait
```

Or from a list of HabCam filenames (resolved to GCS paths automatically --
see the caveat in `docs/architecture.md` before trusting this beyond a
plumbing test):
```bash
pemad-infer build-input --model ultralytics --habcam scallop_filenames.txt --upload gs://ggn-nmfs-osi-dev-1-data/jeremy/inputs/scallop_test.json
```

Or sample 1-in-N images from an entire survey folder (sampled per leaf
folder, so the sample stays spread across the whole time range rather
than being dominated by whichever folder has the most images -- **always
try `--dry-run` first**, since this lists the *entire* prefix before
sampling and a whole-year folder can be large). Leaf folders that don't
match the standard `<YYYYMMDD>_<HHMM>` naming (e.g. a flat `auv` folder)
are excluded by default and called out in the printed summary -- pass
`--include-nonconforming-folders` once you've confirmed a specific one is
safe to sample from (see `survey_sampler.py` and `docs/architecture.md`'s
"Known limitations" for why):
```bash
pemad-infer trigger \
  --model ultralytics \
  --survey-prefix "gs://nmfs-dev-uc1-landing-bucket/NEFSC/HabCam Survey/habcam/proc/Images/2023/" \
  --sample-rate 50 \
  --yaml-config configs/ultralytics/24star.yaml \
  --gcs-prefix jeremy \
  --dry-run
```

Check on a run without `--wait`:
```bash
pemad-infer status                                               # recent runs
pemad-infer status --logical-date "2026-09-30T13:05:36+00:00"    # task-by-task state for one run
```

### Staging weights + auto-writing the config

Instead of uploading weights by hand and then hand-editing a local YAML's
`weights:` line, supply the local weights file directly and let `trigger`
stage both (Ultralytics-family models only -- VIAME configs bake weights
into the Docker image, they have no `weights:` field):
```bash
pemad-infer trigger \
  --model ultralytics \
  --weights-file ~/PEMAD-ESB-ScalTrain/train_arc/output/2426crabdata_yolo12n_gcp_20260910_205126/weights/best.pt \
  --yaml-config configs/ultralytics/24star.yaml \
  --run-name 2426crabdata_yolo12n_gcp_20260910_205126 \
  --survey-prefix "gs://nmfs-dev-uc1-landing-bucket/NEFSC/HabCam Survey/habcam/proc/Images/2023/" \
  --sample-rate 50 \
  --gcs-prefix jeremy \
  --dry-run
```
By default this is **content-addressed / deduplicated**: the local file
is hashed (sha256) and staged to
`gs://<bucket>/<gcs-prefix>/weights/by-hash/<sha256>/<filename>` --
a shared, run-independent path. If that exact byte-identical file was
already staged by any prior run (same model retriggered, same
`--weights-file`), the upload is **skipped** and the existing URI is
reused, instead of copying the same multi-hundred-MB file into a fresh
`<run-name>`-scoped path every single time. Which run used a given
weights file is still recorded in that run's run card (see below), so
nothing about provenance is lost by sharing the storage path across
runs. Pass `--no-dedupe-weights` (on `trigger`, `stage-weights`, or
`stage-config`) to restore the old behavior -- always upload to a
fresh `gs://<bucket>/<gcs-prefix>/weights/<run-name>/weights/<filename>`
path, even if identical content already exists elsewhere -- e.g. if
you deliberately want one run's copy fully isolated while debugging a
suspected bad upload.

The rewritten YAML config is always uploaded fresh to
`gs://<bucket>/<gcs-prefix>/configs/<run-name>.yaml` -- before the DAG
is ever triggered.

To stage weights and/or a config without triggering anything (e.g. to
pre-stage a batch of runs), use the standalone subcommands:
```bash
pemad-infer stage-weights --weights-file best.pt --run-name 24star_v2 --gcs-prefix jeremy
pemad-infer stage-config --yaml-config configs/ultralytics/24star.yaml --weights-file best.pt \
  --run-name 24star_v2 --gcs-prefix jeremy
```

### Multi-weight (cascade) models

A model that needs more than one local weights file -- e.g. a two-stage
detector -> classifier cascade, see `docs/two-stage-cascade.md` -- adds a
repeatable `--extra-weights FIELD=PATH` alongside `--weights-file` on
`trigger` or `stage-config`. Each extra file is staged next to the primary
weights and wired into a top-level `<FIELD>:` line in the YAML, the same
way `--weights-file` handles `weights:`:

(also available as `examples/trigger_cancer_crab_cascade.sh`). `FIELD`
must match whatever the container's `model.py` actually reads from its
config -- see `docs/two-stage-cascade.md` for the full walkthrough,
including building the cascade container itself.

### Combined-image-contract models (e.g. `star-cascade`)

Some containers (anything forked from `optics-models-ultralytics-detection`
-- app.py + model.py + inference_runner.py all in one image, e.g. the real
`star-cascade` model) use a different input contract than the
YAML_CONFIG_PATH-plus-flat-instances-list wrapper the other model families
use: `inference_runner.py` never reads `YAML_CONFIG_PATH` at all, and
expects each manifest instance to be a dict with `input_files`/
`output_file`/`config` keys rather than a bare `gs://` string. Confirmed
2026-09-30 after a real `star-cascade` Cloud Batch job failed every
instance with `'str' object has no attribute 'get'` using the older flat
manifest shape. Pass `--combined-image-contract` on `trigger` (or
`build-input`) for these models -- it builds the correct nested manifest
shape instead, embedding the staged/rewritten YAML config inline:
```bash
pemad-infer trigger \
  --model star-cascade \
  --weights-file models/24star_yolo12n_gcp_20260909_174845/weights/best.pt \
  --extra-weights classifier_weights=models/24star_yolo12n_gcp_20260909_174845/weights/star_tax.pt \
  --extra-weights taxonomy_json=models/24star_yolo12n_gcp_20260909_174845/weights/star_taxonomy.json \
  --yaml-config configs/two_stage/template.yaml \
  --combined-image-contract \
  --run-name star-cascade_20260930 \
  --survey-prefix "gs://nmfs-dev-uc1-landing-bucket/NEFSC/HabCam Survey/habcam/proc/Images/2023/" \
  --sample-rate 5 \
  --gcs-prefix jeremy \
  --wait
```
The output file this writes to is run-specific by default --
`gs://<bucket>/<output-folder>/<run-name>/annotations.json` -- so
consecutive runs against the same `--output-folder` no longer overwrite
each other's output. That resolved path is also printed at the end of a
`--wait` run and recorded on the run card.

#### Stereo splicing: `--stereo-side`

HabCam frames are spliced left+right stereo pairs in one wide image.
Running the detector on the full spliced frame means a real organism
visible in both eyes' overlap zone gets detected *and* classified
twice -- once per eye, at two different x-positions -- which
double-counts it (confirmed 2026-09-30 against a real survey-scale
star-cascade run). `--stereo-side {full,left,right}` (default `full`,
for backward compatibility) fixes this at the source rather than
post-hoc: it's threaded into the manifest's inline config, and the
combined-image-contract `model.py` crops each image to that half
*before* the stage-1 detector ever sees it, so bbox coordinates come
out already relative to the single eye -- no remapping needed
downstream. Only meaningful with `--combined-image-contract`; `trigger`
exits early with a clear message if you pass `--stereo-side left`/
`right` without it.
```bash
pemad-infer trigger \
  --model star-cascade \
  --weights-file best.pt \
  --extra-weights classifier_weights=star_tax.pt taxonomy_json=star_taxonomy.json \
  --yaml-config configs/two_stage/template.yaml \
  --combined-image-contract \
  --stereo-side left \
  --run-name star-cascade_20260930_left \
  --survey-prefix "gs://nmfs-dev-uc1-landing-bucket/NEFSC/HabCam Survey/habcam/proc/Images/2023/" \
  --sample-rate 5 \
  --gcs-prefix jeremy \
  --wait
```
To cover both eyes, trigger twice (once with `--stereo-side left`, once
`right`, ideally under different `--run-name`s so their outputs don't
collide) -- there's no single-run "both" mode yet. Each output image
entry also records `stereo_side` and `stereo_crop_x_offset` (how far
the crop's left edge sits from the original full frame, `0` for `full`/
`left` or half the frame width for `right`) -- useful if a later step
ever needs to map a detection back to full-frame coordinates (e.g. for
real stereo triangulation instead of just avoiding double-counting).
Video inputs aren't split yet -- `model.py` logs a warning and falls
back to the full frame rather than silently skipping the split; not
needed for the current HabCam-stills use case.

### Input staging (multi-folder inputs) and cleanup

When a resolved image list (from `--survey-prefix`, `--habcam`, or
`--images`) spans more than one GCS folder, `trigger` automatically
copies every image into one flat, run-specific staging folder first
(`gs://<bucket>/<gcs-prefix>/staging/<run-name>/`, a cheap server-side
copy, no download/upload/egress cost) before submitting the job -- this
sidesteps a known bug in the older chunking wrapper where a batch
spanning multiple folders crashes with `FileNotFoundError` (see
`docs/architecture.md`). Pass `--no-flatten-inputs` to skip this and
submit the original (unstaged) paths.

Nothing deletes these staging folders on its own, so they grow without
bound as more runs are triggered. Two ways to manage that:

1. **`pemad-infer stage-cleanup`** -- reports (dry run, the default) or
   deletes (`--force`) staging run folders whose newest file is older
   than `--older-than-days` (default 30). Recent folders are always left
   alone, so staged candidate images from recent runs stay available for
   comparing output across runs:
   ```bash
   pemad-infer stage-cleanup --gcs-prefix jeremy                       # dry run -- report only
   pemad-infer stage-cleanup --gcs-prefix jeremy --older-than-days 14 --force  # actually delete
   ```
2. **A GCS Object Lifecycle rule** (recommended as the durable,
   zero-maintenance fix -- set once by a bucket admin, needs no ongoing
   CLI/cron involvement):
   ```bash
   gcloud storage buckets update gs://ggn-nmfs-osi-dev-1-data \
     --lifecycle-file=/dev/stdin <<'EOF'
   {
     "rule": [
       {
         "action": {"type": "Delete"},
         "condition": {"age": 30, "matchesPrefix": ["jeremy/staging/"]}
       }
     ]
   }
   EOF
   ```
   The two are complementary -- `stage-cleanup` is useful for an
   on-demand check or a shorter/different retention window than the
   bucket-wide rule.

### Run cards

Every `trigger` (including under `--dry-run`) writes a **run card**: a
JSON snapshot -- model definition, resolved input manifest path and image
count, the exact YAML config text used, the survey-sample report if
`--survey-prefix` was used, and a creation timestamp -- uploaded to
`gs://<bucket>/<gcs-prefix>/run_cards/<model>_<timestamp>.json` *before*
anything is submitted. This is a pre-flight provenance record,
independent of Airflow's own post-hoc archived-config YAML. Skip it with
`--no-card` if you don't want one for a given run.

## Layout

- `src/pemad_esb_inference/` -- the package
  - `config.py` -- project/environment constants (env-var overridable)
  - `model_registry.py` -- reads the live `model_runtime_definitions.json`, resolves a model key, infers its family
  - `input_builder.py` -- builds the Ultralytics/VIAME flat/VIAME manifest shapes, plus the combined-image-contract manifest shape (`--combined-image-contract`, see above)
  - `input_staging.py` -- flattens a multi-folder input list into one run-specific `staging/` GCS folder via server-side copy (see "Input staging" above)
  - `habcam_paths.py` -- resolves HabCam filenames to their expected GCS paths
  - `survey_sampler.py` -- samples 1-in-N images from a survey folder, stratified per leaf folder, excluding non-conforming leaf-folder names (e.g. `auv`) by default
  - `config_builder.py` -- stages local weights (content-addressed/deduplicated by default, see above) + rewrites a local YAML template's `weights:` field automatically (and, via `--extra-weights`, any number of additional named weight fields for multi-weight/cascade models)
  - `run_card.py` -- builds and uploads a pre-flight provenance "run card" for a trigger
  - `gcs.py` -- small google-cloud-storage wrappers (upload/download/list/exists/delete)
  - `airflow_client.py` -- triggers/queries the DAG via `gcloud composer environments run`
  - `batch_monitor.py` -- resolves and polls the resulting Cloud Batch job
  - `cli.py` -- the `pemad-infer` command (`models|build-input|trigger|status|stage-weights|stage-config|stage-cleanup`)
- `configs/` -- known-good pipeline YAML configs, one per model/weight-set
- `docs/` -- architecture notes, the permissions/troubleshooting model, the BYOM/Docker reference, and the two-stage cascade model guide
- `examples/` -- copy-paste shell examples, plus a structural (untested) template for a cascade model's `model.py`
- `tests/` -- unit tests for the pure-Python pieces (manifest building, HabCam path resolution, config rewriting, run cards) -- no GCP credentials required to run these

## Known limitations / next steps

See `docs/architecture.md`'s "Known limitations" section -- summary: the
Cloud Batch job lookup used by `--wait` isn't safe for concurrent runs of
the same model yet; `airflow_client.py` shells out to `gcloud` rather than
using the native Composer API client; the HabCam folder-layout assumption
hasn't been confirmed against a real (non-test) image path; and VIAME
video/stereo manifests exist in code but aren't wired into the CLI flags
yet.
