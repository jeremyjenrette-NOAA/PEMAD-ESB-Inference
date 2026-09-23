# Bring-your-own-model (Docker) onboarding -- reference only

**Not implemented by this CLI.** `pemad-infer` triggers and monitors models
that are *already* registered in `model_runtime_definitions.json`. Building
and registering a brand-new model container is a separate, Docker-heavy
workflow. This doc is kept here for future reference (per the "keep
everything on record" approach used for the rest of this project's source
docs).

If/when this becomes a real priority, the natural shape is a second CLI
subcommand (e.g. `pemad-infer onboard`) wrapping the `docker build` /
`docker push` / `model_runtime_definitions.json`-edit steps below -- not
started yet.

## Two-phase pattern (from the NOAA "Bring Your Self-Trained VIAME Model to OSI" guide)

Applies to VIAME-family models specifically:

- **Phase 1 (build, on a GPU Cloud Workstation -- the `viame-gpu` base
  image is ~14GB, too large for most laptops' Docker):** zip model
  artifacts as `configs/pipelines/<detector>.pipe` + weights at the zip's
  top level; build the model image via VIAME's official
  `viame_gpu_vertex_ai_with_addon.sh` add-on script; push to
  `nmfs-dev-uc1-docker-repository`; then build a wrapper image from the
  `optics-inference-runner` repo (`BASE_IMAGE` build arg) and push that
  too. The workstation is only needed for building, not for running
  inference afterward.
- **Phase 2 (register):** download, edit, and re-upload
  `gs://ggn-nmfs-osi-dev-1-data/configs/model_runtime_definitions.json`
  with a new entry modeled on an existing one (e.g. `sefsc-class-fish`).
  Standard vs. stereo VIAME models use slightly different `args` (stereo
  adds an `input_list_0.txt` symlink step).

## Reference: `optics-models-hello-world` README

Source: a colleague's (csbrown-noaa) `optics-models-hello-world` template
repo -- a general Docker-based path for onboarding a model that is
*neither* Ultralytics- nor VIAME-family (those have their own, easier
paths -- see above). Reproduced here as supplied, for reference:

> ### NOAA NMFS Optics Model Deployment Template ("Hello World")
>
> Welcome! This repository is a starting template for deploying your custom
> Computer Vision models into the Optics SI Airflow ecosystem. If you have
> an Ultralytics-family model or a VIAME-family model **STOP!** -- there
> are existing frameworks for that, please use those, it's easier that way.
>
> Our infrastructure requires models to run inside isolated Docker
> containers, expose an HTTP endpoint, and communicate with Google Cloud
> Storage (GCS). Most of this infrastructure is pre-written; you mainly
> need to import your model and get predictions into the expected shape.
>
> **Phase 1: Deploy the "Hello World" baseline.** Fork/clone the repo. It
> ships with a dummy model that draws random bounding boxes -- deploy it
> exactly as-is first to verify your Docker setup, GCP permissions, and
> Airflow config all work before writing any real CV code.
>
> - `gcloud auth application-default login`, then
>   `docker build -t optics-hello-world:latest .`
> - Run it locally: `docker run -p 8080:8080 -v ~/.config/gcloud:/tmp/.config/gcloud -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/.config/gcloud/application_default_credentials.json -e GOOGLE_CLOUD_PROJECT=ggn-nmfs-osi-dev-1 optics-hello-world:latest`
> - Upload the provided `test_payloads/` JSON + sample media to a GCS
>   folder you control, then `curl -X POST http://localhost:8080/predict -d @test_payloads/test_payload_images.json` (and the video/manifest variants) to confirm it works end to end -- new KWCOCO files should appear in GCS.
> - Push to the registry: `gcloud auth login`, then tag and
>   `docker push us-central1-docker.pkg.dev/ggn-nmfs-osi-dev-1/nmfs-dev-uc1-docker-repository/optics-hello-world:latest`.
>   **Warning: pushing to this shared registry replaces the existing image
>   at that tag for everyone** -- be careful with naming.
> - Register it: download
>   `gs://ggn-nmfs-osi-dev-1-data/configs/model_runtime_definitions.json`,
>   add an entry modeled on:
>   ```json
>   "optics-hello-world": {
>       "region": "us-central1",
>       "image": "us-central1-docker.pkg.dev/ggn-nmfs-osi-dev-1/nmfs-dev-uc1-docker-repository/optics-hello-world:latest",
>       "cpu": 4, "memory": "16Gi",
>       "gpu": 0, "gpu_type": null,
>       "machine_type": "c2-standard-4",
>       "timeout": 360000,
>       "command": ["python"],
>       "args": ["/workspace/inference_runner.py"]
>   }
>   ```
>   (flip `gpu`/`gpu_type`/`machine_type` for a GPU model; never change
>   `args`), save, and re-upload.
> - Trigger it: since a headless Cloud Batch VM can't accept a local
>   `curl`, upload your test payload JSON to GCS first, then trigger the
>   DAG (UI or headless) with `model_type=optics-hello-world` and
>   `input_file=<that GCS URI>`.
>
> **Phase 2: Bring Your Own Model (BYOM).** Replace the dummy logic in
> `model.py` with real inference code (read `input_dir`, load weights/
> config, run your framework, write output -- KWCOCO recommended for
> interoperability with the rest of the pipeline). Add real dependencies
> to `requirements.txt` (OpenCV is already included); add system-level
> packages to the `Dockerfile` if needed. Then repeat: test locally with
> `curl`, pick a real image name (increment the version tag, or keep
> `:latest` so you don't have to keep editing the DAG registry entry),
> rebuild, push, update `model_runtime_definitions.json` if the image name
> changed, and trigger.

(Condensed for length in a couple of places above -- see the
`optics-models-hello-world` GitHub repo directly for the full, current
version, since this snapshot can drift out of date.)
