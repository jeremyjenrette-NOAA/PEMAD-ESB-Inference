#!/usr/bin/env bash
# Two-stage cascade (detector -> taxonomic classifier) trigger example.
# See docs/two-stage-cascade.md for the full picture: the cascade runs as
# ONE model_type/container (registered in model_runtime_definitions.json,
# e.g. "cancer-crab-cascade") -- Airflow/Cloud Batch/this CLI never see two
# separate stages. `--weights-file` stages the detector weights (rewrites
# the YAML's `weights:` field, same as any single-model Ultralytics
# config); `--extra-weights classifier_weights=...` stages the classifier
# weights alongside it and rewrites a second `classifier_weights:` field.
#
# Prerequisites this script does NOT do for you:
#   1. The cancer-crab-cascade container (fork of optics-models-hello-world
#      with a two-stage model.py -- see examples/cascade_model_template.py)
#      must be built, pushed to nmfs-dev-uc1-docker-repository, and
#      registered in model_runtime_definitions.json.
#   2. configs/cancer-crab-cascade/template.yaml's field names
#      (classifier_weights, classifier_payload) are placeholders -- confirm
#      them against the real container's model.py and adjust both the YAML
#      and this command's --extra-weights field name together.
set -euo pipefail

pemad-infer trigger \
  --model cancer-crab-cascade \
  --weights-file ~/PEMAD-ESB-ScalTrain/train_arc/output/2426crabdata_yolo12n_gcp_20260910_205126/weights/best.pt \
  --extra-weights classifier_weights=~/PEMAD-ESB-ScalTrain/train_arc/output/crabdata_tax/weights/crabdata_tax.pt \
  --yaml-config configs/cancer-crab-cascade/template.yaml \
  --run-name cancer_crab_cascade_$(date +%Y%m%d_%H%M%S) \
  --survey-prefix "gs://nmfs-dev-uc1-landing-bucket/NEFSC/HabCam Survey/habcam/proc/Images/2023/" \
  --sample-rate 50 \
  --gcs-prefix jeremy \
  --dry-run
# Drop --dry-run once the printed conf, staged weights paths, and run card
# all look right.
