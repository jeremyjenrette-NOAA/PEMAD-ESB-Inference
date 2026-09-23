from pemad_esb_inference.input_staging import (
    plan_flat_names,
    spans_multiple_folders,
)


def test_spans_multiple_folders_true_when_paths_differ():
    paths = [
        "gs://b/2023/20230101_10/20230101_1000/img1.png",
        "gs://b/2023/20230101_10/20230101_1010/img2.png",
    ]
    assert spans_multiple_folders(paths)


def test_spans_multiple_folders_false_for_one_folder():
    paths = [
        "gs://b/2023/20230101_10/20230101_1000/img1.png",
        "gs://b/2023/20230101_10/20230101_1000/img2.png",
    ]
    assert not spans_multiple_folders(paths)


def test_spans_multiple_folders_false_for_single_path():
    assert not spans_multiple_folders(["gs://b/2023/x/img1.png"])


def test_plan_flat_names_keeps_original_basename_when_unique():
    paths = [
        "gs://b/folderA/img1.png",
        "gs://b/folderB/img2.png",
    ]
    plan = plan_flat_names(paths)
    assert plan == [
        ("gs://b/folderA/img1.png", "img1.png"),
        ("gs://b/folderB/img2.png", "img2.png"),
    ]


def test_plan_flat_names_dedupes_on_real_collision():
    # Same filename, two different source folders -- must not collide
    # once flattened into one destination folder.
    paths = [
        "gs://b/folderA/img.png",
        "gs://b/folderB/img.png",
        "gs://b/folderC/img.png",
    ]
    plan = plan_flat_names(paths)
    dest_names = [dest for _src, dest in plan]
    assert dest_names == ["img.png", "img_1.png", "img_2.png"]
    # Every destination name is unique.
    assert len(set(dest_names)) == len(dest_names)


def test_plan_flat_names_preserves_order_and_length():
    paths = [f"gs://b/f{i}/img.png" for i in range(5)]
    plan = plan_flat_names(paths)
    assert [src for src, _dest in plan] == paths
    assert len(plan) == len(paths)


def test_plan_flat_names_dedupe_handles_no_extension():
    paths = ["gs://b/folderA/readme", "gs://b/folderB/readme"]
    plan = plan_flat_names(paths)
    dest_names = [dest for _src, dest in plan]
    assert dest_names == ["readme", "readme_1"]


def test_flatten_to_staging_copies_and_preserves_order(monkeypatch):
    from pemad_esb_inference import input_staging

    copied = []

    class FakeGcs:
        @staticmethod
        def copy_blob(source_uri, dest_uri):
            copied.append((source_uri, dest_uri))
            return dest_uri

    monkeypatch.setattr(input_staging, "gcs", FakeGcs, raising=False)
    # flatten_to_staging does `from . import gcs` internally, so patch the
    # real module it will import instead of the (not-yet-imported) local name.
    import pemad_esb_inference.gcs as real_gcs_module
    monkeypatch.setattr(real_gcs_module, "copy_blob", FakeGcs.copy_blob)

    paths = [
        "gs://b/folderA/img.png",
        "gs://b/folderB/img.png",
    ]
    result = input_staging.flatten_to_staging(paths, "gs://b/staging/run1")

    assert result == [
        "gs://b/staging/run1/img.png",
        "gs://b/staging/run1/img_1.png",
    ]
    assert sorted(copied) == sorted(
        [
            ("gs://b/folderA/img.png", "gs://b/staging/run1/img.png"),
            ("gs://b/folderB/img.png", "gs://b/staging/run1/img_1.png"),
        ]
    )
