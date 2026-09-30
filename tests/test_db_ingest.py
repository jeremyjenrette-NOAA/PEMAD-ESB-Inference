from pemad_esb_inference.db_ingest import build_rows


def _sample_manifest():
    return {
        "run_card": "gs://bucket/jeremy/run_cards/star-cascade_x.json",
        "model_type": "star-cascade",
        "stereo_side": "left",
        "created_at": "2026-09-30T19:15:57.447836+00:00",
        "input_file": "gs://bucket/jeremy/inputs/star-cascade_x.json",
        "output_file": "gs://bucket/jeremy/output/star-cascade_x/annotations.json",
        "yaml_config_snapshot": "weights: \"gs://bucket/weights/best.pt\"\n",
        "categories": [{"id": 0, "name": "cancer_crab"}],
        "images": [
            {
                "id": 1,
                "file_name": "frame_a.jpg",
                "source_uri": "gs://bucket/landing/frame_a.jpg",
                "width": 1936,
                "height": 1216,
                "stereo_side": "left",
                "stereo_crop_x_offset": 0,
                "context_thumb": "context/1.jpg",
                "annotation_count": 2,
            },
            {
                "id": 2,
                "file_name": "frame_b.jpg",
                "source_uri": "gs://bucket/landing/frame_b.jpg",
                "width": 1936,
                "height": 1216,
                "stereo_side": "left",
                "stereo_crop_x_offset": 0,
                "context_thumb": "context/2.jpg",
                "annotation_count": 0,
            },
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "bbox": [5.0, 6.0, 20.0, 21.0],
                "category_id": 0,
                "category_name": "cancer_crab",
                "score": 0.9,
                "genus": "Henricia",
                "genus_confidence": 0.99,
                "species": "henricia_sp",
                "species_confidence": 0.98,
                "crop_thumb": "crops/10.jpg",
            },
            {
                "id": 11,
                "image_id": 1,
                "bbox": [50.0, 60.0, 15.0, 16.0],
                "category_id": 0,
                "category_name": "cancer_crab",
                "score": 0.2,
                "genus": "Asterias",
                "genus_confidence": 0.5,
                "species": "vulgaris",
                "species_confidence": 0.5,
                "crop_thumb": "crops/11.jpg",
            },
        ],
        "skipped_images": [],
    }


def test_build_rows_run_row_pulls_from_manifest_top_level():
    rows = build_rows(_sample_manifest())
    run = rows["run"]
    assert run["run_card_uri"] == "gs://bucket/jeremy/run_cards/star-cascade_x.json"
    assert run["model_type"] == "star-cascade"
    assert run["stereo_side"] == "left"
    assert run["input_file_uri"] == "gs://bucket/jeremy/inputs/star-cascade_x.json"
    assert run["output_file_uri"] == "gs://bucket/jeremy/output/star-cascade_x/annotations.json"
    assert run["image_count"] == 2
    assert run["detection_count"] == 2


def test_build_rows_images_without_asset_prefix_keep_relative_paths():
    rows = build_rows(_sample_manifest())
    image = rows["images"][0]
    assert image["_manifest_id"] == 1
    assert image["file_name"] == "frame_a.jpg"
    assert image["source_uri"] == "gs://bucket/landing/frame_a.jpg"
    assert image["context_thumb_uri"] == "context/1.jpg"  # unchanged -- no prefix given


def test_build_rows_images_with_asset_prefix_rewrites_thumb_uris():
    rows = build_rows(_sample_manifest(), asset_uri_prefix="gs://bucket/jeremy/viewer_assets/run1")
    image = rows["images"][0]
    assert image["context_thumb_uri"] == "gs://bucket/jeremy/viewer_assets/run1/context/1.jpg"
    detection = rows["detections"][0]
    assert detection["crop_thumb_uri"] == "gs://bucket/jeremy/viewer_assets/run1/crops/10.jpg"


def test_build_rows_detections_unpack_bbox_and_keep_manifest_image_id_for_fk_resolution():
    rows = build_rows(_sample_manifest())
    det = rows["detections"][0]
    assert det["_manifest_image_id"] == 1  # resolved to a real images.id by load_into_postgres
    assert (det["bbox_x"], det["bbox_y"], det["bbox_w"], det["bbox_h"]) == (5.0, 6.0, 20.0, 21.0)
    assert det["genus"] == "Henricia"
    assert det["score"] == 0.9


def test_build_rows_handles_missing_optional_fields_gracefully():
    manifest = {
        "images": [{"id": 1, "file_name": "f.jpg", "width": 10, "height": 10}],
        "annotations": [{"id": 1, "image_id": 1, "bbox": [0, 0, 1, 1], "score": 0.5}],
    }
    rows = build_rows(manifest)
    assert rows["run"]["run_card_uri"] is None
    assert rows["images"][0]["source_uri"] is None
    assert rows["images"][0]["context_thumb_uri"] is None
    assert rows["images"][0]["stereo_crop_x_offset"] == 0
    assert rows["detections"][0]["genus"] is None
