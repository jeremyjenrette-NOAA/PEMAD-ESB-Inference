"""Central configuration for the PEMAD-ESB-Inference CLI.

All values default to the known-good settings for the NOAA Optics /
NEFSC Cloud Batch + Airflow pipeline (project `ggn-nmfs-osi-dev-1`), but
every value can be overridden with an environment variable so this
package stays portable if the pipeline ever moves projects/environments.
"""
import os

PROJECT_ID = os.environ.get("PEMAD_PROJECT_ID", "ggn-nmfs-osi-dev-1")
REGION = os.environ.get("PEMAD_REGION", "us-central1")

# Cloud Composer / Airflow
COMPOSER_ENVIRONMENT = os.environ.get("PEMAD_COMPOSER_ENV", "composer-env1")
COMPOSER_LOCATION = os.environ.get("PEMAD_COMPOSER_LOCATION", "us-central1")
DAG_ID = os.environ.get("PEMAD_DAG_ID", "nmfs-optics-pipeline-longrunning-dag")

# GCS
# The DAG's `output_bucket` param is hard-coded server-side ("Fixed bucket
# due to permissions. Please do not change." per the DAG's own Param
# description) -- this default matches that. Don't point it elsewhere
# unless the DAG itself changes.
OUTPUT_BUCKET = os.environ.get("PEMAD_OUTPUT_BUCKET", "ggn-nmfs-osi-dev-1-data")
MODEL_RUNTIME_DEFINITIONS_URI = os.environ.get(
    "PEMAD_MODEL_DEFINITIONS_URI",
    "gs://ggn-nmfs-osi-dev-1-data/configs/model_runtime_definitions.json",
)

# HabCam landing bucket path used by habcam_paths.py.
# NOTE: not yet confirmed against a real, non-test image -- see
# docs/architecture.md "Known limitations".
HABCAM_BASE_PATH = os.environ.get(
    "PEMAD_HABCAM_BASE_PATH",
    "gs://nmfs-dev-uc1-landing-bucket/NEFSC/HabCam Survey/habcam/proc/Images",
)
