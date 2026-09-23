# Pipeline YAML configs

One file per model/weight-set, named `<model_type>/<short-label>.yaml`. These
are the `yaml_config_path` inputs the DAG downloads and validates in its
`load_pipeline_metadata` task -- keep them here so a known-good config is
never just sitting on one person's workstation.

`ultralytics/24star.yaml` is the config validated end-to-end against
`composer-env1` on 2026-09-17 (weights: `24star_yolo12n_gcp_20260730_202404`).
Add new files alongside it as new weight sets are trained; don't overwrite
a config that's referenced from a completed run's archived runtime YAML
(`gs://ggn-nmfs-osi-dev-1-data/configs/archive/<run_id>.yaml`) if you want
that run to stay reproducible.
