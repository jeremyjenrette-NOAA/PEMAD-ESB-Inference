from pemad_esb_inference import habcam_paths

BASE = "gs://nmfs-dev-uc1-landing-bucket/NEFSC/HabCam Survey/habcam/proc/Images"


def test_parse_imagename_extracts_date_and_time():
    parsed = habcam_paths.parse_imagename("202403.20240508.142352788.7482.png")
    assert parsed["date_yyyymmdd"] == "20240508"
    assert parsed["hour"] == "14"
    assert parsed["minute"] == "23"
    assert parsed["tenmin"] == "20"


def test_build_expected_path_matches_known_layout():
    path = habcam_paths.build_expected_path("202403.20240508.142352788.7482.png", base_path=BASE)
    assert path == (
        f"{BASE}/2024/202405/20240508/20240508_14/20240508_1420/"
        "202403.20240508.142352788.7482.png"
    )


def test_unparseable_name_returns_none():
    assert habcam_paths.build_expected_path("not-a-habcam-name.png") is None


def test_resolve_paths_skips_unparseable_and_reports(capsys):
    resolved = habcam_paths.resolve_paths(
        ["202403.20240508.142352788.7482.png", "bad-name.png"], base_path=BASE
    )
    assert len(resolved) == 1
    assert resolved[0].endswith("202403.20240508.142352788.7482.png")
    assert "Warning" in capsys.readouterr().out
