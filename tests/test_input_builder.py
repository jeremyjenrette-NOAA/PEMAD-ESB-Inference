import pytest

from pemad_esb_inference import input_builder


def test_ultralytics_manifest_is_flat_list():
    manifest = input_builder.ultralytics_manifest(["gs://b/a.png", "gs://b/b.png"])
    assert manifest == {"instances": ["gs://b/a.png", "gs://b/b.png"]}


def test_viame_images_manifest_wraps_in_instance_object():
    manifest = input_builder.viame_images_manifest(["gs://b/a.png"])
    assert manifest == {"instances": [{"input_images": ["gs://b/a.png"]}]}


def test_viame_video_manifest():
    manifest = input_builder.viame_video_manifest("gs://b/vid.mp4")
    assert manifest == {"instances": [{"input_path": "gs://b/vid.mp4"}]}


def test_viame_stereo_images_manifest():
    manifest = input_builder.viame_stereo_images_manifest(["gs://b/l1.png"], ["gs://b/r1.png"])
    assert manifest == {"instances": [{"input_images": [["gs://b/l1.png"], ["gs://b/r1.png"]]}]}


def test_manifest_for_family_dispatches_correctly():
    assert input_builder.manifest_for_family("ultralytics", ["x"]) == {"instances": ["x"]}
    assert input_builder.manifest_for_family("viame", ["x"]) == {"instances": [{"input_images": ["x"]}]}


def test_manifest_for_family_rejects_unknown_family():
    with pytest.raises(ValueError):
        input_builder.manifest_for_family("vertex", ["x"])
