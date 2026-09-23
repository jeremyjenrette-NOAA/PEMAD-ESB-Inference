# Two-stage cascade models (e.g. cancer_crab detector -> taxonomic classifier)

Decision (2026-09-17): build cascades as a **single custom container**, not
a new multi-task Airflow DAG. Rationale: the detector and classifier were
confirmed to need similar compute, so there's no resource-efficiency reason
to split them into separate Cloud Batch jobs, and a single container needs
zero changes to `nmfs-optics-pipeline-longrunning-dag` or to this repo's
CLI -- it's just a new `model_type` entry, same as any other model.

If a future cascade's stages genuinely need different hardware (e.g. a
GPU-hungry detector feeding a heavier downstream model), revisit a real
two-stage DAG then -- that's a bigger lift (new DAG file, dropped into the
shared Composer bucket with careful naming, a GCS handoff between stages)
and isn't built here.

## How it fits the existing pipeline

Nothing about `nmfs-optics-pipeline-longrunning-dag`, `model_registry.py`,
or `input_builder.py` needs to change. A cascade model:

- Takes the same flat `{"instances": ["gs://...", ...]}` input manifest as
  any other Ultralytics-shaped model (the `family` heuristic in
  `model_registry.py` -- "anything with a non-empty `command`" -- already
  covers this correctly, since the cascade's `model_runtime_definitions.json`
  entry uses `"command": ["python"]`, same as the hello-world template).
- Is triggered and monitored with the exact same `pemad-infer trigger
  --model <cascade-name> ...` command as `ultralytics` or any VIAME model.
- Is registered by adding one entry to `model_runtime_definitions.json`,
  same process as any other new model (see `docs/byom-docker-onboarding.md`).

All the cascading (detect -> crop -> classify -> combine) happens *inside*
the container, invisible to Airflow/Cloud Batch/this CLI.

## Building the container

1. Fork `optics-models-hello-world` (the template referenced in
   `docs/byom-docker-onboarding.md`) as the starting point -- it already
   handles the GCS download/upload plumbing, HTTP server, and KWCOCO
   output convention; you only need to change `model.py` (and
   `requirements.txt`/`Dockerfile` for new dependencies: `ultralytics` for
   the detector, plus whatever `crabdata_tax.pt` needs to load).
2. Copy both weight files (`best.pt`, `crabdata_tax.pt`) into the image
   (or download them from GCS at container startup -- check how the
   hello-world template's `model.py` receives its `config` dict, since
   weight paths likely belong there rather than hard-coded).
3. Replace `model.py`'s dummy logic with the two-stage version. A
   structural starting point is at `examples/cascade_model_template.py`
   in this repo -- **it has not been run against the real hello-world
   template or the real weight files**, so treat it as a skeleton to
   adapt, not tested code. In particular:
   - Confirm the actual `model.py` function signature/contract against
     the real template (the README describes it as reading `input_dir`,
     a `config` dict, and writing to `output_file_path`, but confirm
     exact names).
   - Confirm how `crabdata_tax.pt` should be loaded (`torch.load` of a
     full model object? a `state_dict` needing a model class? TorchScript
     via `torch.jit.load`?) and its expected input size/normalization.
   - Confirm the hierarchical taxonomy's actual output shape (single flat
     species softmax vs. multiple prediction heads, e.g. genus then
     species) and adjust `_classify_crop` / the output schema accordingly
     -- the template assumes a single flat label for simplicity.
   - Decide the final output schema: at minimum, each detection's bbox +
     detector confidence + predicted species label + classifier
     confidence. Consider also writing the crop images themselves to the
     output folder (even though this is a single-container cascade) if
     you'll want to QA misclassifications without re-running detection.
4. Test locally first with `docker run` + `curl`, exactly as the
   hello-world README describes, using a handful of real crab images
   before ever submitting a real Cloud Batch job.
5. Push to `nmfs-dev-uc1-docker-repository` and register in
   `model_runtime_definitions.json`, e.g.:
   ```json
   "cancer-crab-cascade": {
       "region": "us-central1",
       "image": "us-central1-docker.pkg.dev/ggn-nmfs-osi-dev-1/nmfs-dev-uc1-docker-repository/cancer-crab-cascade:latest",
       "cpu": 4,
       "memory": "16Gi",
       "gpu": 0,
       "gpu_type": null,
       "machine_type": "c2-standard-4",
       "timeout": 360000,
       "command": ["python"],
       "args": ["/workspace/inference_runner.py"]
   }
   ```
   (Defaults here match the existing CPU-only `ultralytics` entry, since
   the existing `24star` Ultralytics config already runs on `device: cpu`
   -- a reasonable starting profile given the detector/classifier were
   confirmed to have similar, presumably modest, compute needs. Bump to
   `gpu: 1` / `g2-standard-4` if real testing shows CPU is too slow.)
6. Run a small test via `pemad-infer trigger --model cancer-crab-cascade
   --survey-prefix <a small folder> --sample-rate <high N> --dry-run`
   first, then drop `--dry-run` once the assembled conf looks right.

## Staging both weight files from the CLI (added 2026-09-17)

A cascade container needs two local weight files staged, not one --
`config_builder.py`'s `--weights-file` auto-stage flow originally only
handled a single top-level `weights:` field. `trigger` and `stage-config`
now also take a repeatable `--extra-weights FIELD=PATH` flag: it stages
each extra local file next to the primary weights
(`gs://<bucket>/<gcs-prefix>/weights/<run-name>/weights/<field>_<filename>`)
and rewrites a top-level `<field>:` line in the YAML the same way
`--weights-file` does for `weights:`. `FIELD` must be whatever the
cascade container's `model.py` actually reads out of its config -- see the
"confirm" caveats in step 3 above; `classifier_weights` here is a
placeholder convention (matching `examples/cascade_model_template.py` and
`configs/cancer-crab-cascade/template.yaml`), not a confirmed contract.

```bash
pemad-infer trigger \
  --model cancer-crab-cascade \
  --weights-file ~/train_arc/output/.../weights/best.pt \
  --extra-weights classifier_weights=~/train_arc/output/crabdata_tax/weights/crabdata_tax.pt \
  --yaml-config configs/cancer-crab-cascade/template.yaml \
  --run-name cancer_crab_cascade_20260917 \
  --survey-prefix "gs://nmfs-dev-uc1-landing-bucket/NEFSC/HabCam Survey/habcam/proc/Images/2023/" \
  --sample-rate 50 \
  --gcs-prefix jeremy \
  --dry-run
```
(also available as `examples/trigger_cancer_crab_cascade.sh`). Both staged
URIs, and the config text with both fields rewritten, are captured in the
run card (`run_card.py`'s `extra_weights` field) alongside everything else
a single-weights run captures. `--extra-weights` requires `--weights-file`
(it stages alongside the primary weights, not standalone) and works
identically on `stage-config` for pre-staging without triggering.

## Airflow-trigger bug fixed 2026-09-17

Separately: the first real (non-`--dry-run`) `trigger` against any model --
cascade or otherwise -- failed with gcloud's
`argument SUBCOMMAND: Must be specified`. `airflow_client.py` was putting
the whole airflow subcommand (`dags trigger ...`) after `gcloud`'s `--`;
gcloud's own usage requires `SUBCOMMAND [SUBCOMMAND_NESTED]` (`dags`,
`trigger`) *before* `--`, with only the arguments meant for Airflow itself
(the dag_id, `--conf`, ...) after it. Fixed in `_run_airflow_command`; see
that function's docstring for the exact before/after argv.
