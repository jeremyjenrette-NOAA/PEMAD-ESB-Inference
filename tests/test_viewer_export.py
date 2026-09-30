import io
import json
import zipfile

import pytest
from PIL import Image

import pemad_esb_inference.gcs as gcs_module
from pemad_esb_inference.viewer_export import _padded_crop_box, _resize_capped, export


def _make_stereo_jpeg(width=200, height=100) -> bytes:
    # Left half red, right half blue -- lets us confirm a "left"
    # crop actually only samples the red half, not a blend, and a
    # "right" crop only samples the blue half.
    img = Image.new("RGB", (width, height), (0, 0, 255))
    for x in range(width // 2):
        for y in range(height):
            img.putpixel((x, y), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


def test_export_end_to_end(tmp_path, monkeypatch):
    run_card = {
        "output_file": "gs://bucket/jeremy/output/run1/annotations.json",
        "input_file": "gs://bucket/jeremy/inputs/run1.json",
        "model_type": "star-cascade",
        "created_at": "2026-09-30T18:29:52.802318+00:00",
    }
    annotations = {
        "categories": [{"id": 0, "name": "cancer_crab"}],
        "images": [
            {
                "id": 1,
                "file_name": "frame_a.jpg",
                "width": 100,
                "height": 100,
                "stereo_side": "left",
                "stereo_crop_x_offset": 0,
            },
            {
                # References a file with no match in the input manifest --
                # must be skipped, not crash the whole export.
                "id": 2,
                "file_name": "frame_missing.jpg",
                "width": 100,
                "height": 100,
                "stereo_side": "left",
                "stereo_crop_x_offset": 0,
            },
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "bbox": [5, 5, 20, 20],
                "category_id": 0,
                "score": 0.9,
                "genus": "Henricia",
                "genus_confidence": 0.99,
                "species": "henricia_sp",
                "species_confidence": 0.98,
            },
            {
                # Sits right on the image edge -- padding must clamp,
                # not throw or produce a zero-area crop.
                "id": 11,
                "image_id": 1,
                "bbox": [0, 90, 10, 10],
                "category_id": 0,
                "score": 0.2,
                "genus": "Asterias",
                "genus_confidence": 0.5,
                "species": "vulgaris",
                "species_confidence": 0.5,
            },
        ],
    }
    input_manifest = {
        "instances": [
            {
                "input_files": [
                    "gs://bucket/landing/2023/original/frame_a.jpg",
                    "gs://bucket/landing/2023/original/frame_b.jpg",
                ]
            }
        ]
    }

    texts = {
        "gs://bucket/jeremy/run_cards/run1.json": json.dumps(run_card),
        "gs://bucket/jeremy/output/run1/annotations.json": json.dumps(annotations),
        "gs://bucket/jeremy/inputs/run1.json": json.dumps(input_manifest),
    }
    frame_bytes = _make_stereo_jpeg(200, 100)

    def fake_download_bytes(uri):
        assert uri == "gs://bucket/landing/2023/original/frame_a.jpg", f"unexpected download {uri}"
        return frame_bytes

    monkeypatch.setattr(gcs_module, "download_text", lambda uri: texts[uri])
    monkeypatch.setattr(gcs_module, "download_bytes", fake_download_bytes)

    out_dir = tmp_path / "viewer_assets"
    result = export(
        "gs://bucket/jeremy/run_cards/run1.json", str(out_dir), max_context_dim=500, max_crop_dim=100
    )

    assert result["image_count"] == 1  # frame_missing.jpg skipped
    assert result["skipped_images"] == ["frame_missing.jpg"]
    assert result["annotation_count"] == 2

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["images"][0]["file_name"] == "frame_a.jpg"
    assert manifest["images"][0]["annotation_count"] == 2
    assert manifest["annotations"][0]["category_name"] == "cancer_crab"
    assert manifest["annotations"][0]["genus"] == "Henricia"

    context_img = Image.open(out_dir / "context" / "1.jpg")
    assert context_img.size[0] <= 500

    for ann_id in (10, 11):
        crop_img = Image.open(out_dir / "crops" / f"{ann_id}.jpg")
        assert crop_img.size[0] >= 1 and crop_img.size[1] >= 1

    assert result["zip_path"] == str(out_dir) + ".zip"
    with zipfile.ZipFile(result["zip_path"]) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        assert "context/1.jpg" in names
        assert "crops/10.jpg" in names


@pytest.mark.parametrize(
    "side,offset,expected_rgb",
    [
        ("left", 0, (255, 0, 0)),
        ("right", 100, (0, 0, 255)),
    ],
)
def test_export_stereo_crop_samples_correct_eye(tmp_path, monkeypatch, side, offset, expected_rgb):
    """The exact bug this whole feature exists to catch: re-deriving the
    stereo crop from stereo_crop_x_offset/width must actually land on the
    recorded eye, not the full frame or the other eye."""
    run_card = {"output_file": "gs://b/out.json", "input_file": "gs://b/in.json"}
    annotations = {
        "categories": [],
        "images": [
            {
                "id": 1,
                "file_name": "f.jpg",
                "width": 100,
                "height": 100,
                "stereo_side": side,
                "stereo_crop_x_offset": offset,
            }
        ],
        "annotations": [],
    }
    input_manifest = {"instances": [{"input_files": ["gs://b/orig/f.jpg"]}]}
    texts = {
        "gs://b/rc.json": json.dumps(run_card),
        "gs://b/out.json": json.dumps(annotations),
        "gs://b/in.json": json.dumps(input_manifest),
    }
    frame_bytes = _make_stereo_jpeg(200, 100)
    monkeypatch.setattr(gcs_module, "download_text", lambda uri: texts[uri])
    monkeypatch.setattr(gcs_module, "download_bytes", lambda uri: frame_bytes)

    out_dir = tmp_path / "out"
    export("gs://b/rc.json", str(out_dir), make_zip=False)
    img = Image.open(out_dir / "context" / "1.jpg")
    center_pixel = img.getpixel((img.size[0] // 2, img.size[1] // 2))
    # JPEG re-encoding can shift values by a few levels -- compare loosely.
    assert all(abs(a - b) < 10 for a, b in zip(center_pixel, expected_rgb))


def test_export_raises_without_output_file(tmp_path, monkeypatch):
    texts = {"gs://bucket/rc.json": json.dumps({"input_file": "gs://bucket/in.json"})}
    monkeypatch.setattr(gcs_module, "download_text", lambda uri: texts[uri])
    with pytest.raises(ValueError, match="output_file"):
        export("gs://bucket/rc.json", str(tmp_path / "out"))


def test_export_raises_without_input_file(tmp_path, monkeypatch):
    texts = {"gs://bucket/rc.json": json.dumps({"output_file": "gs://bucket/out.json"})}
    monkeypatch.setattr(gcs_module, "download_text", lambda uri: texts[uri])
    with pytest.raises(ValueError, match="input_file"):
        export("gs://bucket/rc.json", str(tmp_path / "out"))


def test_padded_crop_box_clamps_to_image_bounds():
    left, top, right, bottom = _padded_crop_box(
        (90, 90, 10, 10), image_width=100, image_height=100, padding_pct=50
    )
    assert 0 <= left < right <= 100
    assert 0 <= top < bottom <= 100


def test_padded_crop_box_degenerate_edge_box_still_nonzero_area():
    left, top, right, bottom = _padded_crop_box(
        (0, 0, 0.4, 0.4), image_width=100, image_height=100, padding_pct=0
    )
    assert right > left
    assert bottom > top


def test_resize_capped_noop_when_already_small():
    img = Image.new("RGB", (50, 30))
    assert _resize_capped(img, 100).size == (50, 30)


def test_resize_capped_scales_longer_side_down():
    img = Image.new("RGB", (1000, 500))
    assert _resize_capped(img, 200).size == (200, 100)
