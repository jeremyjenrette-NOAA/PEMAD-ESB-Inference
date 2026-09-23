from pemad_esb_inference.config_builder import (
    rewrite_weights_field,
    rewrite_yaml_field,
    stage_config,
)


def test_rewrite_weights_field_replaces_existing_uncommented_line():
    yaml_text = (
        'weights: "gs://old-bucket/old/path/best.pt"\n'
        "payload:\n"
        "  conf: 0.25\n"
    )
    result = rewrite_weights_field(yaml_text, "gs://new-bucket/new/path/best.pt")
    assert 'weights: "gs://new-bucket/new/path/best.pt"' in result
    assert "gs://old-bucket" not in result
    # Rest of the file is untouched.
    assert "payload:\n  conf: 0.25\n" in result


def test_rewrite_weights_field_ignores_commented_alternates():
    yaml_text = (
        'weights: "gs://old-bucket/a/best.pt"\n'
        '# weights: "gs://old-bucket/b/alt.pt"\n'
    )
    result = rewrite_weights_field(yaml_text, "gs://new-bucket/best.pt")
    # Only the uncommented line is replaced.
    assert 'weights: "gs://new-bucket/best.pt"' in result
    assert '# weights: "gs://old-bucket/b/alt.pt"' in result
    assert result.count("gs://new-bucket") == 1


def test_rewrite_weights_field_inserts_when_missing():
    yaml_text = "payload:\n  conf: 0.25\n"
    result = rewrite_weights_field(yaml_text, "gs://new-bucket/best.pt")
    assert result.startswith('weights: "gs://new-bucket/best.pt"\n')
    assert "payload:\n  conf: 0.25\n" in result


def test_rewrite_weights_field_only_replaces_first_match():
    # Malformed input with two uncommented `weights:` lines -- shouldn't
    # happen in practice, but confirms we don't touch anything past the
    # first match (count=1 in the underlying re.sub).
    yaml_text = 'weights: "gs://a/best.pt"\nweights: "gs://b/best.pt"\n'
    result = rewrite_weights_field(yaml_text, "gs://c/best.pt")
    assert result == 'weights: "gs://c/best.pt"\nweights: "gs://b/best.pt"\n'


# --- rewrite_yaml_field: the general form used for cascade (multi-weight)
# models, e.g. a second `classifier_weights:` field alongside `weights:`
# (see docs/two-stage-cascade.md). --------------------------------------

def test_rewrite_yaml_field_inserts_new_field():
    yaml_text = 'weights: "gs://bucket/best.pt"\npayload:\n  conf: 0.25\n'
    result = rewrite_yaml_field(yaml_text, "classifier_weights", "gs://bucket/tax.pt")
    assert result.startswith('classifier_weights: "gs://bucket/tax.pt"\n')
    assert 'weights: "gs://bucket/best.pt"' in result


def test_rewrite_yaml_field_replaces_existing_field_only():
    yaml_text = (
        'weights: "gs://bucket/old-best.pt"\n'
        'classifier_weights: "gs://bucket/old-tax.pt"\n'
    )
    result = rewrite_yaml_field(yaml_text, "classifier_weights", "gs://bucket/new-tax.pt")
    assert 'classifier_weights: "gs://bucket/new-tax.pt"' in result
    assert "old-tax.pt" not in result
    # The unrelated `weights:` field is untouched.
    assert 'weights: "gs://bucket/old-best.pt"' in result


def test_rewrite_weights_field_is_rewrite_yaml_field_for_weights():
    yaml_text = 'weights: "gs://a/best.pt"\n'
    assert rewrite_weights_field(yaml_text, "gs://b/best.pt") == rewrite_yaml_field(
        yaml_text, "weights", "gs://b/best.pt"
    )


# --- stage_config with extra_weights: staging a second (or Nth) weights
# file for a cascade model in the same call that stages the primary one. --

def test_stage_config_with_extra_weights_stages_and_rewrites_each_field(tmp_path, monkeypatch):
    uploaded_files = []
    uploaded_text = {}

    def fake_upload_file(local_path, dest):
        uploaded_files.append((local_path, dest))

    def fake_upload_text(text, dest, content_type=None):
        uploaded_text["dest"] = dest
        uploaded_text["text"] = text

    import pemad_esb_inference.gcs as gcs_module

    monkeypatch.setattr(gcs_module, "upload_file", fake_upload_file)
    monkeypatch.setattr(gcs_module, "upload_text", fake_upload_text)

    yaml_path = tmp_path / "template.yaml"
    yaml_path.write_text("model_family: cascade\npayload:\n  conf: 0.25\n", encoding="utf-8")
    detector_path = tmp_path / "best.pt"
    detector_path.write_bytes(b"")
    classifier_path = tmp_path / "crabdata_tax.pt"
    classifier_path.write_bytes(b"")

    yaml_uri, weights_uri, new_yaml_text, extra_weight_uris = stage_config(
        local_yaml_path=str(yaml_path),
        local_weights_path=str(detector_path),
        run_name="cascade_run",
        gcs_prefix="jeremy",
        bucket="ggn-nmfs-osi-dev-1-data",
        extra_weights={"classifier_weights": str(classifier_path)},
    )

    assert weights_uri == "gs://ggn-nmfs-osi-dev-1-data/jeremy/weights/cascade_run/weights/best.pt"
    assert extra_weight_uris == {
        "classifier_weights": (
            "gs://ggn-nmfs-osi-dev-1-data/jeremy/weights/cascade_run/weights/"
            "classifier_weights_crabdata_tax.pt"
        )
    }
    assert f'weights: "{weights_uri}"' in new_yaml_text
    assert f'classifier_weights: "{extra_weight_uris["classifier_weights"]}"' in new_yaml_text
    assert yaml_uri == "gs://ggn-nmfs-osi-dev-1-data/jeremy/configs/cascade_run.yaml"
    # Both weight files were uploaded, plus the rewritten config text.
    assert len(uploaded_files) == 2
    assert uploaded_text["text"] == new_yaml_text


def test_stage_config_without_extra_weights_behaves_as_before(tmp_path, monkeypatch):
    import pemad_esb_inference.gcs as gcs_module

    monkeypatch.setattr(gcs_module, "upload_file", lambda *a, **k: None)
    monkeypatch.setattr(gcs_module, "upload_text", lambda *a, **k: None)

    yaml_path = tmp_path / "template.yaml"
    yaml_path.write_text('weights: "gs://old/best.pt"\n', encoding="utf-8")
    weights_path = tmp_path / "best.pt"
    weights_path.write_bytes(b"")

    _yaml_uri, _weights_uri, _new_yaml_text, extra_weight_uris = stage_config(
        local_yaml_path=str(yaml_path),
        local_weights_path=str(weights_path),
        run_name="r1",
        gcs_prefix="jeremy",
    )
    assert extra_weight_uris == {}
