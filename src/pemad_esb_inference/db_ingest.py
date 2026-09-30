"""Loads a `pemad-infer export-viewer-assets` export into Postgres, for
the results-auditor web app (Cloud Run + Cloud SQL long-term; a local
Postgres container while the schema/ingestion flow is still being
proven out -- see db/schema.sql and the architecture notes in the
project). This replaces the flat manifest.json + local JPEGs a
session-local viewer reads directly with rows a real backend can
query/filter/paginate server-side, and adds the one thing the static
export can't: a place for a reviewer's decision to live.

Two independent pieces, kept separate because uploading images to GCS
only needs to happen once per export -- re-running ingestion (e.g.
while iterating on the schema) shouldn't re-upload the same thumbnails:

  1. `upload_assets()` -- pushes an export's context/ and crops/ JPEGs
     to a GCS prefix, returns the resulting gs:// URI prefix.
  2. `build_rows()` -- pure function: manifest.json (+ that URI prefix)
     -> the exact row dicts for runs/images/detections. No database
     dependency, so it's unit-testable without a live Postgres.
  3. `load_into_postgres()` -- rows + a DSN -> inserts them in one
     transaction (runs -> images -> detections, using RETURNING ids to
     wire up the foreign keys correctly) via psycopg2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def upload_assets(assets_dir: str, gcs_prefix: str, bucket: Optional[str] = None) -> str:
    """Upload every file under assets_dir/context/ and assets_dir/crops/
    to gs://<bucket>/<gcs_prefix>/<context|crops>/..., preserving the
    relative layout manifest.json's thumb paths already use. Returns
    the gs://<bucket>/<gcs_prefix> URI prefix (no trailing slash)."""
    from . import config, gcs  # deferred: keeps this module importable without google-cloud-storage

    bucket = bucket or config.OUTPUT_BUCKET
    root = Path(assets_dir)
    uploaded = 0
    for sub in ("context", "crops"):
        sub_dir = root / sub
        if not sub_dir.is_dir():
            continue
        for f in sorted(sub_dir.iterdir()):
            if not f.is_file():
                continue
            dest = f"gs://{bucket}/{gcs_prefix}/{sub}/{f.name}"
            gcs.upload_file(str(f), dest)
            uploaded += 1
    print(f"[ingest-run] Uploaded {uploaded} thumbnail(s) to gs://{bucket}/{gcs_prefix}/", flush=True)
    return f"gs://{bucket}/{gcs_prefix}"


def build_rows(manifest: Dict, asset_uri_prefix: Optional[str] = None) -> Dict:
    """manifest is the parsed manifest.json from export-viewer-assets.
    asset_uri_prefix, if given, turns each thumb's relative path (e.g.
    "context/1.jpg") into "<asset_uri_prefix>/context/1.jpg" -- pass the
    return value of upload_assets(). If omitted, the relative path is
    kept as-is, which is only useful for inspecting the rows before
    anything has actually been uploaded anywhere."""

    def thumb_uri(rel_path):
        if rel_path is None:
            return None
        return f"{asset_uri_prefix}/{rel_path}" if asset_uri_prefix else rel_path

    run_row = {
        "run_card_uri": manifest.get("run_card"),
        "model_type": manifest.get("model_type"),
        "stereo_side": manifest.get("stereo_side"),
        "created_at": manifest.get("created_at"),
        "input_file_uri": manifest.get("input_file"),
        "output_file_uri": manifest.get("output_file"),
        "yaml_config_snapshot": manifest.get("yaml_config_snapshot"),
        "image_count": len(manifest.get("images", [])),
        "detection_count": len(manifest.get("annotations", [])),
    }

    image_rows = []
    for im in manifest.get("images", []):
        image_rows.append(
            {
                "_manifest_id": im["id"],  # local key only -- resolved to a real DB id in load_into_postgres
                "file_name": im["file_name"],
                "source_uri": im.get("source_uri"),
                "width": im["width"],
                "height": im["height"],
                "stereo_side": im.get("stereo_side"),
                "stereo_crop_x_offset": im.get("stereo_crop_x_offset", 0) or 0,
                "context_thumb_uri": thumb_uri(im.get("context_thumb")),
            }
        )

    detection_rows = []
    for ann in manifest.get("annotations", []):
        x, y, w, h = ann["bbox"]
        detection_rows.append(
            {
                "_manifest_image_id": ann["image_id"],  # local key only -- see above
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "category_id": ann.get("category_id"),
                "category_name": ann.get("category_name"),
                "score": ann["score"],
                "genus": ann.get("genus"),
                "genus_confidence": ann.get("genus_confidence"),
                "species": ann.get("species"),
                "species_confidence": ann.get("species_confidence"),
                "crop_thumb_uri": thumb_uri(ann.get("crop_thumb")),
            }
        )

    return {"run": run_row, "images": image_rows, "detections": detection_rows}


def load_into_postgres(rows: Dict, dsn: str) -> int:
    """Inserts rows built by build_rows() into Postgres in one
    transaction. Returns the new runs.id. Raises (and rolls back) on
    any error -- including re-ingesting a run_card_uri that's already
    there, since runs.run_card_uri is UNIQUE by design: ingest-run is
    meant to be run once per export, not repeatedly."""
    import psycopg2  # deferred: keeps this module importable without psycopg2 installed

    conn = psycopg2.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (run_card_uri, model_type, stereo_side, created_at,
                                   input_file_uri, output_file_uri, yaml_config_snapshot,
                                   image_count, detection_count)
                VALUES (%(run_card_uri)s, %(model_type)s, %(stereo_side)s, %(created_at)s,
                        %(input_file_uri)s, %(output_file_uri)s, %(yaml_config_snapshot)s,
                        %(image_count)s, %(detection_count)s)
                RETURNING id
                """,
                rows["run"],
            )
            run_id = cur.fetchone()[0]

            manifest_id_to_db_id = {}
            for image in rows["images"]:
                cur.execute(
                    """
                    INSERT INTO images (run_id, file_name, source_uri, width, height,
                                         stereo_side, stereo_crop_x_offset, context_thumb_uri)
                    VALUES (%(run_id)s, %(file_name)s, %(source_uri)s, %(width)s, %(height)s,
                            %(stereo_side)s, %(stereo_crop_x_offset)s, %(context_thumb_uri)s)
                    RETURNING id
                    """,
                    {**image, "run_id": run_id},
                )
                manifest_id_to_db_id[image["_manifest_id"]] = cur.fetchone()[0]

            for det in rows["detections"]:
                cur.execute(
                    """
                    INSERT INTO detections (run_id, image_id, bbox_x, bbox_y, bbox_w, bbox_h,
                                             category_id, category_name, score, genus,
                                             genus_confidence, species, species_confidence,
                                             crop_thumb_uri)
                    VALUES (%(run_id)s, %(image_id)s, %(bbox_x)s, %(bbox_y)s, %(bbox_w)s, %(bbox_h)s,
                            %(category_id)s, %(category_name)s, %(score)s, %(genus)s,
                            %(genus_confidence)s, %(species)s, %(species_confidence)s,
                            %(crop_thumb_uri)s)
                    """,
                    {
                        **det,
                        "run_id": run_id,
                        "image_id": manifest_id_to_db_id[det["_manifest_image_id"]],
                    },
                )
        return run_id
    finally:
        conn.close()
