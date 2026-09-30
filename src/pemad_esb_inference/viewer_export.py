"""Export crop/thumbnail image assets + a manifest.json for the results
viewer, from a completed run's run card.

The kwcoco annotations.json a run produces only has bboxes + bare
filenames -- no pixels, and no full gs:// path (the model container only
ever sees basenames once inputs are staged). This module re-derives
everything the viewer needs to actually show detections:

  1. Read the run card (uploaded by `upload_run_card` at trigger time) to
     get `output_file` (the annotations.json) and `input_file` (the input
     manifest -- the *only* place the full gs:// path for each sampled
     image is recorded; annotations.json has just the basename).
  2. For each image referenced in the annotations, download the ORIGINAL
     full-resolution frame, and re-apply the exact same stereo crop
     model.py applied -- using each image entry's own recorded
     `stereo_side`/`stereo_crop_x_offset`/`width`/`height`, so there's no
     need to duplicate model.py's `_split_stereo` logic here: the
     recorded offset + width already say exactly which sub-rectangle was
     fed to the detector. Confirmed 2026-09-30 against a synthetic
     red/blue stereo frame: a "left" crop (offset=0) samples pure red, a
     "right" crop (offset=width) samples pure blue.
  3. Save a resized "context" thumbnail of that eye-crop (whole frame,
     for orientation) and, per annotation, a padded crop around just that
     bbox (for judging the detection/classification itself) -- both as
     small JPEGs so a viewer built as a self-contained page can embed a
     few hundred of them without ballooning in size.
  4. Write manifest.json bundling image/annotation metadata with paths to
     those thumbnails.

Only supports combined-image-contract runs (a single run-wide
output_file) -- the flat multi-instance contract's per-instance output
paths aren't handled here yet.
"""
from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def _basename_lookup(input_files: List[str]) -> Dict[str, str]:
    """Map each input file's basename -> its full gs:// URI. Raises if
    two different full paths share a basename -- would silently merge
    two distinct images under one lookup key."""
    lookup: Dict[str, str] = {}
    for uri in input_files:
        name = uri.rsplit("/", 1)[-1]
        if name in lookup and lookup[name] != uri:
            raise ValueError(
                f"Duplicate basename {name!r} maps to two different input paths "
                f"({lookup[name]!r} and {uri!r}) -- can't tell which image an "
                f"annotation's file_name refers to."
            )
        lookup[name] = uri
    return lookup


def _resize_capped(image: Image.Image, max_dim: int) -> Image.Image:
    """Resize so the longer side is at most max_dim, preserving aspect.
    No-op if already smaller."""
    width, height = image.size
    longest = max(width, height)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.LANCZOS)


def _padded_crop_box(
    bbox: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    padding_pct: float,
) -> Tuple[int, int, int, int]:
    """bbox is [x, y, w, h] (kwcoco/coco style). Returns a (left, top,
    right, bottom) box padded by padding_pct of the box's own width/height
    on each side and clamped to the image bounds, so a small detection
    gets a bit of surrounding context instead of a tight, hard-to-read
    crop."""
    x, y, w, h = bbox
    pad_x = w * (padding_pct / 100.0)
    pad_y = h * (padding_pct / 100.0)
    left = max(0, int(x - pad_x))
    top = max(0, int(y - pad_y))
    right = min(image_width, int(x + w + pad_x))
    bottom = min(image_height, int(y + h + pad_y))
    # Guard against a degenerate (zero-area) box after clamping, e.g. a
    # detection sitting exactly on the image edge.
    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)
    return left, top, right, bottom


def export(
    run_card_uri: str,
    output_dir: str,
    max_context_dim: int = 900,
    max_crop_dim: int = 320,
    crop_padding_pct: float = 15.0,
    make_zip: bool = True,
) -> Dict:
    """Download the run card, its annotations, and the input manifest it
    references, then write context/crop thumbnails + manifest.json under
    output_dir. Returns a summary dict (also printed by the CLI)."""
    from . import gcs  # deferred: keeps this module importable without google-cloud-storage

    run_card = json.loads(gcs.download_text(run_card_uri))
    output_file = run_card.get("output_file")
    input_file = run_card.get("input_file")
    if not output_file:
        raise ValueError(
            f"Run card {run_card_uri} has no 'output_file' -- was this run triggered "
            f"without --combined-image-contract? Only that manifest shape records a "
            f"single results file this way; export-viewer-assets doesn't yet support "
            f"the flat multi-instance contract's per-instance output paths."
        )
    if not input_file:
        raise ValueError(
            f"Run card {run_card_uri} has no 'input_file' to resolve original image "
            f"paths from."
        )

    annotations = json.loads(gcs.download_text(output_file))
    input_manifest = json.loads(gcs.download_text(input_file))
    input_files = input_manifest["instances"][0]["input_files"]
    basename_to_uri = _basename_lookup(input_files)

    category_names = {cat["id"]: cat["name"] for cat in annotations.get("categories", [])}

    output_root = Path(output_dir)
    context_dir = output_root / "context"
    crops_dir = output_root / "crops"
    context_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    annotations_by_image: Dict[int, List[dict]] = {}
    for ann in annotations.get("annotations", []):
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    manifest_images = []
    manifest_annotations = []
    skipped_images = []

    for image_entry in annotations.get("images", []):
        image_id = image_entry["id"]
        file_name = image_entry["file_name"]
        source_uri = basename_to_uri.get(file_name)
        if source_uri is None:
            skipped_images.append(file_name)
            continue

        print(f"[export-viewer-assets] {file_name}: downloading original frame...", flush=True)
        original = Image.open(io.BytesIO(gcs.download_bytes(source_uri))).convert("RGB")

        offset = image_entry.get("stereo_crop_x_offset", 0) or 0
        crop_width = image_entry.get("width", original.width)
        crop_height = image_entry.get("height", original.height)
        eye_crop = original.crop((offset, 0, offset + crop_width, crop_height))
        if eye_crop.size != (crop_width, crop_height):
            print(
                f"[export-viewer-assets] WARNING: {file_name}: expected the stereo crop "
                f"to be {crop_width}x{crop_height}, got {eye_crop.size} -- the original "
                f"frame's dimensions may not match what this run actually processed.",
                flush=True,
            )

        context_thumb_rel = f"context/{image_id}.jpg"
        _resize_capped(eye_crop, max_context_dim).save(
            context_dir / f"{image_id}.jpg", "JPEG", quality=85
        )

        image_annotations = annotations_by_image.get(image_id, [])
        for ann in image_annotations:
            box = _padded_crop_box(tuple(ann["bbox"]), crop_width, crop_height, crop_padding_pct)
            crop_thumb_rel = f"crops/{ann['id']}.jpg"
            _resize_capped(eye_crop.crop(box), max_crop_dim).save(
                crops_dir / f"{ann['id']}.jpg", "JPEG", quality=85
            )
            manifest_annotations.append(
                {
                    "id": ann["id"],
                    "image_id": image_id,
                    "bbox": ann["bbox"],
                    "category_id": ann.get("category_id"),
                    "category_name": category_names.get(ann.get("category_id")),
                    "score": ann.get("score"),
                    "genus": ann.get("genus"),
                    "genus_confidence": ann.get("genus_confidence"),
                    "species": ann.get("species"),
                    "species_confidence": ann.get("species_confidence"),
                    "crop_thumb": crop_thumb_rel,
                }
            )

        manifest_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": crop_width,
                "height": crop_height,
                "stereo_side": image_entry.get("stereo_side"),
                "stereo_crop_x_offset": offset,
                "context_thumb": context_thumb_rel,
                "annotation_count": len(image_annotations),
            }
        )

    manifest = {
        "run_card": run_card_uri,
        "model_type": run_card.get("model_type"),
        "created_at": run_card.get("created_at"),
        "categories": [{"id": cid, "name": name} for cid, name in category_names.items()],
        "images": manifest_images,
        "annotations": manifest_annotations,
        "skipped_images": skipped_images,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = None
    if make_zip:
        zip_path = shutil.make_archive(str(output_root), "zip", root_dir=str(output_root))

    return {
        "output_dir": str(output_root),
        "manifest_path": str(manifest_path),
        "zip_path": zip_path,
        "image_count": len(manifest_images),
        "annotation_count": len(manifest_annotations),
        "skipped_images": skipped_images,
    }
