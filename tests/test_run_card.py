from pemad_esb_inference.model_registry import ModelDefinition
from pemad_esb_inference.run_card import build_run_card


def _sample_definition(command=None):
    return ModelDefinition(
        key="24star_yolo12n",
        region="us-central1",
        image="us-central1-docker.pkg.dev/proj/repo/ultralytics:latest",
        cpu=4,
        memory="16Gi",
        gpu=0,
        gpu_type=None,
        machine_type="c2-standard-4",
        timeout=360000,
        command=command if command is not None else ["python", "infer.py"],
        args=[],
    )


def test_build_run_card_captures_inputs_and_family():
    definition = _sample_definition()
    card = build_run_card(
        model_type="24star_yolo12n",
        model_definition=definition,
        input_file="gs://bucket/prefix/inputs/24star_yolo12n_20260917T000000Z.json",
        yaml_config_path="gs://bucket/prefix/configs/run.yaml",
        yaml_config_text="weights: \"gs://bucket/prefix/weights/run/weights/best.pt\"\n",
        output_bucket="ggn-nmfs-osi-dev-1-data",
        output_folder="prefix/output/",
        gcs_prefix="jeremy",
        input_count=42,
    )
    assert card.model_type == "24star_yolo12n"
    assert card.input_count == 42
    assert card.model_definition["key"] == "24star_yolo12n"
    # `family` is a ModelDefinition property, not a dataclass field --
    # build_run_card must add it explicitly so it survives into the card.
    assert card.model_definition["family"] == "ultralytics"
    assert card.survey_sample is None
    assert card.created_at  # non-empty ISO timestamp


def test_build_run_card_captures_viame_family_and_survey_sample():
    definition = _sample_definition(command=[])
    survey_sample = {
        "prefix": "gs://landing/2023/",
        "sample_rate": 50,
        "folders_seen": 3,
        "total_images": 1500,
        "sampled_images": 30,
        "per_folder": {},
    }
    card = build_run_card(
        model_type="viame_detector",
        model_definition=definition,
        input_file="gs://bucket/prefix/inputs/viame_detector_20260917T000000Z.json",
        yaml_config_path="gs://bucket/prefix/configs/viame.yaml",
        yaml_config_text="",
        output_bucket="ggn-nmfs-osi-dev-1-data",
        output_folder="prefix/output/",
        gcs_prefix="jeremy",
        input_count=30,
        survey_sample=survey_sample,
    )
    assert card.model_definition["family"] == "viame"
    assert card.survey_sample == survey_sample


def test_run_card_to_dict_round_trips_all_fields():
    definition = _sample_definition()
    card = build_run_card(
        model_type="24star_yolo12n",
        model_definition=definition,
        input_file="gs://bucket/prefix/inputs/x.json",
        yaml_config_path="gs://bucket/prefix/configs/run.yaml",
        yaml_config_text="weights: \"gs://bucket/prefix/weights/run/weights/best.pt\"\n",
        output_bucket="ggn-nmfs-osi-dev-1-data",
        output_folder="prefix/output/",
        gcs_prefix="jeremy",
    )
    as_dict = card.to_dict()
    assert as_dict["model_type"] == "24star_yolo12n"
    assert as_dict["output_bucket"] == "ggn-nmfs-osi-dev-1-data"
    assert isinstance(as_dict["model_definition"], dict)
