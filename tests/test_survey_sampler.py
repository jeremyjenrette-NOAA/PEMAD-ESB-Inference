from pemad_esb_inference.survey_sampler import (
    group_by_folder,
    is_conforming_leaf_folder,
    sample_groups,
    sample_survey_folder,
)


def test_group_by_folder_buckets_by_parent_and_filters_extensions():
    paths = [
        "gs://b/2023/20230101_10/20230101_1000/img1.png",
        "gs://b/2023/20230101_10/20230101_1000/img2.png",
        "gs://b/2023/20230101_10/20230101_1010/img3.png",
        "gs://b/2023/20230101_10/20230101_1000/readme.txt",
    ]
    groups = group_by_folder(paths)
    assert len(groups) == 2
    assert len(groups["gs://b/2023/20230101_10/20230101_1000"]) == 2
    assert len(groups["gs://b/2023/20230101_10/20230101_1010"]) == 1


def test_sample_groups_takes_every_nth_per_folder():
    groups = {
        "folderA": [f"img{i}.png" for i in range(10)],
        "folderB": [f"img{i}.png" for i in range(3)],
    }
    sampled, per_folder = sample_groups(groups, sample_rate=5)
    assert per_folder["folderA"] == (10, 2)  # indices 0, 5
    assert per_folder["folderB"] == (3, 1)  # index 0
    assert len(sampled) == 3


def test_sample_groups_rate_1_keeps_everything():
    groups = {"f": ["a.png", "b.png"]}
    sampled, per_folder = sample_groups(groups, sample_rate=1)
    assert sampled == ["a.png", "b.png"]
    assert per_folder["f"] == (2, 2)


def test_sample_groups_rejects_rate_below_one():
    import pytest

    with pytest.raises(ValueError):
        sample_groups({"f": ["a.png"]}, sample_rate=0)


def test_sample_groups_stratifies_evenly_across_uneven_folders():
    # A dense folder (100 images) shouldn't drown out a sparse one (5
    # images) the way a single global sort would.
    groups = {
        "dense": [f"img{i:03d}.png" for i in range(100)],
        "sparse": [f"img{i}.png" for i in range(5)],
    }
    sampled, per_folder = sample_groups(groups, sample_rate=10)
    assert per_folder["dense"] == (100, 10)
    assert per_folder["sparse"] == (5, 1)  # still gets a representative, not zero


# --- is_conforming_leaf_folder / sample_survey_folder's non-conforming-folder
# exclusion. Added after a `pemad-infer trigger --survey-prefix ...
# --sample-rate 500` run processed 53/73 sampled images successfully, then
# died with `FileNotFoundError: No images or videos found in
# /tmp/gcs_downloads_...` partway through the second download chunk -- a
# flat `auv` folder (and other folders that don't match the standard
# <YYYYMMDD>_<HHMM> leaf-folder naming convention) was the leading suspect
# at the time. A follow-up run with this exclusion already applied --
# 71 images, confirmed zero non-conforming folders in the actual uploaded
# manifest -- failed IDENTICALLY (53/71, then an empty second chunk),
# ruling folder naming out as the cause (see survey_sampler.py's module
# docstring and the KNOWN_WRAPPER_CHUNK_BYTES tests below for what's
# actually going on). The exclusion is kept anyway -- these folders are
# still structurally different from the documented layout -- but it's not
# a fix for the chunking failure.

def _fake_listing(paths, size=1024):
    """Build a `list_prefix_with_sizes`-shaped return value from plain
    paths, all the same small size unless overridden.
    """
    return [(p, size) for p in paths]


def test_is_conforming_leaf_folder_accepts_standard_naming():
    assert is_conforming_leaf_folder(
        "gs://b/2023/202306/20230615/20230615_05/20230615_0500"
    )
    assert is_conforming_leaf_folder("20230615_0500")


def test_is_conforming_leaf_folder_rejects_known_anomalies():
    # Flat folder with no date/hour nesting at all.
    assert not is_conforming_leaf_folder("gs://b/2023/auv")
    # A "_orig" variant alongside the normal per-minute buckets.
    assert not is_conforming_leaf_folder(
        "gs://b/2023/202304/20230406/20230406_21/20230406_orig"
    )


def test_sample_survey_folder_excludes_nonconforming_folders_by_default(monkeypatch):
    fake_paths = (
        [
            f"gs://b/2023/202306/20230615/20230615_05/20230615_0500/img{i:03d}.png"
            for i in range(10)
        ]
        + [
            f"gs://b/2023/202304/20230406/20230406_21/20230406_orig/img{i}.png"
            for i in range(4)
        ]
        + ["gs://b/2023/auv/weird1.png"]
    )

    import pemad_esb_inference.gcs as gcs_module

    monkeypatch.setattr(
        gcs_module, "list_prefix_with_sizes", lambda prefix: _fake_listing(fake_paths)
    )

    sampled, report = sample_survey_folder("gs://b/2023/", sample_rate=5)

    assert report.skipped_folders == {
        "gs://b/2023/202304/20230406/20230406_21/20230406_orig": 4,
        "gs://b/2023/auv": 1,
    }
    assert "gs://b/2023/202306/20230615/20230615_05/20230615_0500" in report.per_folder
    assert all("auv" not in p and "orig" not in p for p in sampled)
    # total_images counts everything found (for transparency), even though
    # only the conforming folders were actually sampled from.
    assert report.total_images == 15
    assert report.folders_seen == 3
    assert "Skipped 2 non-conforming folder(s)" in report.summary()


def test_sample_survey_folder_include_nonconforming_opts_back_in(monkeypatch):
    fake_paths = [
        f"gs://b/2023/202306/20230615/20230615_05/20230615_0500/img{i:03d}.png"
        for i in range(3)
    ] + ["gs://b/2023/auv/weird1.png"]

    import pemad_esb_inference.gcs as gcs_module

    monkeypatch.setattr(
        gcs_module, "list_prefix_with_sizes", lambda prefix: _fake_listing(fake_paths)
    )

    sampled, report = sample_survey_folder(
        "gs://b/2023/", sample_rate=5, include_nonconforming=True
    )
    assert report.skipped_folders == {}
    assert any("auv" in p for p in sampled)
    assert "Skipped" not in report.summary()


# --- sampled_bytes / the chunk-boundary warning. This is the actual
# mitigation for the real Cloud Batch failure above: since it's not caused
# by folder naming, and the fix lives inside a container this repo doesn't
# control, the best this CLI can do is warn -- using real GCS object sizes,
# which are free from the same listing call -- before a known-bad-sized
# run gets submitted. ---

def test_sample_survey_folder_warns_when_sample_exceeds_known_chunk_size(monkeypatch):
    from pemad_esb_inference.survey_sampler import KNOWN_WRAPPER_CHUNK_BYTES

    import pemad_esb_inference.gcs as gcs_module

    # One folder, 60 images at 10MB each = 600MB > the ~500MB known chunk size.
    paths = [
        f"gs://b/2023/202306/20230615/20230615_05/20230615_0500/img{i:03d}.png"
        for i in range(60)
    ]
    ten_mb = 10 * 1024 * 1024
    monkeypatch.setattr(
        gcs_module, "list_prefix_with_sizes", lambda prefix: _fake_listing(paths, size=ten_mb)
    )

    sampled, report = sample_survey_folder("gs://b/2023/", sample_rate=1)

    assert report.sampled_bytes == 60 * ten_mb
    assert report.sampled_bytes > KNOWN_WRAPPER_CHUNK_BYTES
    assert "WARNING" in report.summary()
    assert "download chunk" in report.summary()


def test_sample_survey_folder_no_warning_under_known_chunk_size(monkeypatch):
    import pemad_esb_inference.gcs as gcs_module

    # 10 images at 10MB each = 100MB, comfortably under the ~500MB boundary.
    paths = [
        f"gs://b/2023/202306/20230615/20230615_05/20230615_0500/img{i:03d}.png"
        for i in range(10)
    ]
    ten_mb = 10 * 1024 * 1024
    monkeypatch.setattr(
        gcs_module, "list_prefix_with_sizes", lambda prefix: _fake_listing(paths, size=ten_mb)
    )

    sampled, report = sample_survey_folder("gs://b/2023/", sample_rate=1)

    assert report.sampled_bytes == 10 * ten_mb
    assert "WARNING" not in report.summary()
