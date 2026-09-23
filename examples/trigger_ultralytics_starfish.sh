#!/usr/bin/env bash
# Reproduces the first successful end-to-end headless test
# (composer-env1, 2026-09-17): generate_run_id -> load_pipeline_metadata
# -> submit_cloud_batch_job -> wait_for_cloud_batch_job, all "success",
# output landed at gs://ggn-nmfs-osi-dev-1-data/jeremy/output/.
#
# The original raw payload lived at ~/airflow_admin/dag_config.json on
# the dev workstation; this is the same run driven through the CLI.
set -euo pipefail

pemad-infer trigger \
  --model ultralytics \
  --yaml-config gs://ggn-nmfs-osi-dev-1-data/jeremy/configs/ultralytics_config.yaml \
  --input-file gs://ggn-nmfs-osi-dev-1-data/jeremy/inputs/starfish_20260806_182415.json \
  --output-folder "jeremy/output/test-$(date +%Y%m%d-%H%M%S)/" \
  --gcs-prefix jeremy \
  --wait
