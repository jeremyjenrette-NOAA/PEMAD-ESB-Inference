import pytest

from pemad_esb_inference.cli import _parse_extra_weights, build_parser


def test_parse_extra_weights_empty():
    assert _parse_extra_weights(None) == {}
    assert _parse_extra_weights([]) == {}


def test_parse_extra_weights_single_pair():
    assert _parse_extra_weights(["classifier_weights=/models/crabdata_tax.pt"]) == {
        "classifier_weights": "/models/crabdata_tax.pt"
    }


def test_parse_extra_weights_multiple_pairs():
    result = _parse_extra_weights(
        [
            "classifier_weights=/models/crabdata_tax.pt",
            "secondary_weights=/models/other.pt",
        ]
    )
    assert result == {
        "classifier_weights": "/models/crabdata_tax.pt",
        "secondary_weights": "/models/other.pt",
    }


def test_parse_extra_weights_rejects_missing_equals():
    with pytest.raises(SystemExit):
        _parse_extra_weights(["no-equals-sign"])


def test_parse_extra_weights_rejects_empty_field_or_path():
    with pytest.raises(SystemExit):
        _parse_extra_weights(["=/models/x.pt"])
    with pytest.raises(SystemExit):
        _parse_extra_weights(["classifier_weights="])


# --- --include-nonconforming-folders: opts back into sampling from leaf
# folders that don't match the standard HabCam naming convention (see
# survey_sampler.py and tests/test_survey_sampler.py for why this defaults
# to False). Off by default on both build-input and trigger; settable on
# either.

def test_include_nonconforming_folders_defaults_false_on_trigger():
    parser = build_parser()
    args = parser.parse_args(
        [
            "trigger",
            "--model", "ultralytics",
            "--yaml-config", "configs/ultralytics/24star.yaml",
            "--survey-prefix", "gs://b/2023/",
            "--sample-rate", "500",
            "--gcs-prefix", "jeremy",
            "--dry-run",
        ]
    )
    assert args.include_nonconforming_folders is False


def test_include_nonconforming_folders_flag_on_trigger():
    parser = build_parser()
    args = parser.parse_args(
        [
            "trigger",
            "--model", "ultralytics",
            "--yaml-config", "configs/ultralytics/24star.yaml",
            "--survey-prefix", "gs://b/2023/",
            "--sample-rate", "500",
            "--include-nonconforming-folders",
            "--gcs-prefix", "jeremy",
            "--dry-run",
        ]
    )
    assert args.include_nonconforming_folders is True


def test_include_nonconforming_folders_flag_on_build_input():
    parser = build_parser()
    args = parser.parse_args(
        [
            "build-input",
            "--model", "ultralytics",
            "--survey-prefix", "gs://b/2023/",
            "--include-nonconforming-folders",
        ]
    )
    assert args.include_nonconforming_folders is True
