"""Phase 1 read API for the results-auditor app (see the architecture +
roadmap doc in the project). Exposes the Phase 0 schema (db/schema.sql)
over HTTP so the viewer's JS can query a run's detections at request
time -- filtered and paginated server-side -- instead of fetching an
entire run's manifest.json into the browser at once. This is the
addition the static-manifest viewer couldn't do: "large prediction
datasets" don't all have to come down to the client just to look at a
filtered slice of one run.

Read-only. Submitting review decisions is Phase 2 (not here yet).

Run locally against a local Postgres container (see "Loading a run
into Postgres" in the README for standing that up and loading data):

    pip install -e ".[db,api]"
    export PEMAD_DB_DSN="postgresql://postgres:postgres@localhost:5432/viewer_dev"
    uvicorn pemad_esb_inference.api:app --reload --port 8000

    curl http://localhost:8000/runs
    curl "http://localhost:8000/runs/1?genus=Leptasterias&min_score=0.5"

`GET /runs/{id}` deliberately returns the SAME top-level shape as the
`export-viewer-assets` manifest.json (created_at, model_type, images,
annotations, ...) -- it's meant as a drop-in replacement for
fetch('manifest.json') in the existing viewer, not a new contract the
frontend has to learn. The one real difference: `context_thumb` /
`crop_thumb` point at this API's own /thumb proxy (below) instead of a
local relative path, since the DB only has the gs:// URI ingest-run
uploaded them to, and a browser can't load gs:// directly.
"""
from __future__ import annotations

import mimetypes
import os
from typing import List, Optional
from urllib.parse import quote

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

DSN_ENV_VAR = "PEMAD_DB_DSN"
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/viewer_dev"

app = FastAPI(title="PEMAD Results Auditor API", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"

# Permissive CORS for local dev only -- a real origin allowlist (or
# same-origin serving) is a Phase 3 hardening item, not decided yet.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_pool: Optional["pool.SimpleConnectionPool"] = None


def get_pool() -> "pool.SimpleConnectionPool":
    global _pool
    if _pool is None:
        dsn = os.environ.get(DSN_ENV_VAR, DEFAULT_DSN)
        _pool = pool.SimpleConnectionPool(1, 5, dsn)
    return _pool


def run_query(sql: str, params: tuple = ()) -> List[dict]:
    conn = get_pool().getconn()
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        get_pool().putconn(conn)


def thumb_proxy_url(gs_uri: Optional[str]) -> Optional[str]:
    """Rewrite a stored gs:// thumbnail URI into a same-origin URL this
    API can actually serve to a browser (see /thumb below). None stays
    None -- e.g. a run ingested with --skip-upload has no gs:// URI at
    all, just the export's original local-relative path, which this API
    can't serve either way (it never had the file)."""
    if gs_uri is None or not gs_uri.startswith("gs://"):
        return gs_uri
    return f"/thumb?uri={quote(gs_uri, safe='')}"


@app.get("/", include_in_schema=False)
def viewer():
    """Serves the Phase 1 viewer (static/viewer.html) -- the ported
    version of the claude.ai Artifact prototype, pointed at this API
    instead of a static manifest.json. Same-origin by construction, so
    its fetch('/runs/...') calls need no CORS configuration."""
    return FileResponse(STATIC_DIR / "viewer.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/thumb")
def get_thumb(uri: str = Query(..., description="A gs:// URI previously stored by ingest-run's upload_assets()")):
    """Streams one thumbnail from GCS, restricted to the configured
    output bucket -- this is a proxy for exactly the thumbnails
    ingest-run itself uploaded, not an open fetch-any-gs-object
    endpoint. No caching/CDN here; fine for a few dozen/hundred
    thumbnails in Phase 1 dev, not meant to survive past it unchanged."""
    from . import config, gcs  # deferred: keeps api.py importable without google-cloud-storage installed

    if not uri.startswith(f"gs://{config.OUTPUT_BUCKET}/"):
        raise HTTPException(status_code=403, detail="uri is outside the configured output bucket")
    try:
        data = gcs.download_bytes(uri)
    except Exception as exc:  # noqa: BLE001 -- surface as a 404, not a 500, for a missing/renamed object
        raise HTTPException(status_code=404, detail=f"could not read {uri}: {exc}") from exc
    content_type = mimetypes.guess_type(uri)[0] or "image/jpeg"
    return Response(content=data, media_type=content_type)


@app.get("/runs")
def list_runs():
    """Summary of every ingested run -- what a run picker needs."""
    rows = run_query(
        """
        SELECT id, run_card_uri, model_type, stereo_side, created_at, ingested_at,
               image_count, detection_count
        FROM runs
        ORDER BY ingested_at DESC
        """
    )
    return {"runs": rows}


@app.get("/runs/{run_id}")
def get_run(
    run_id: int,
    genus: Optional[str] = Query(None, description="Case-insensitive substring match on genus"),
    stereo_side: Optional[str] = Query(None, description="Filter detections to images with this stereo_side"),
    min_score: Optional[float] = Query(None, ge=0, le=1),
    max_score: Optional[float] = Query(None, ge=0, le=1),
    limit: int = Query(2000, le=10000, description="Max detections returned"),
    offset: int = Query(0, ge=0),
):
    """manifest.json-shaped payload for one run -- see module docstring.
    Filters apply only to `annotations`; `images` is always returned in
    full for the run, since the detail panel needs every detection's
    sibling frame regardless of which detections are currently filtered
    in."""
    run_rows = run_query("SELECT * FROM runs WHERE id = %s", (run_id,))
    if not run_rows:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    run = run_rows[0]

    image_rows = run_query(
        """
        SELECT id, file_name, source_uri, width, height, stereo_side,
               stereo_crop_x_offset, context_thumb_uri
        FROM images WHERE run_id = %s ORDER BY id
        """,
        (run_id,),
    )
    images = [
        {
            "id": im["id"],
            "file_name": im["file_name"],
            "source_uri": im["source_uri"],
            "width": im["width"],
            "height": im["height"],
            "stereo_side": im["stereo_side"],
            "stereo_crop_x_offset": im["stereo_crop_x_offset"],
            "context_thumb": thumb_proxy_url(im["context_thumb_uri"]),
        }
        for im in image_rows
    ]

    clauses = ["d.run_id = %s"]
    params: list = [run_id]
    if genus:
        clauses.append("d.genus ILIKE %s")
        params.append(f"%{genus}%")
    if stereo_side:
        clauses.append("i.stereo_side = %s")
        params.append(stereo_side)
    if min_score is not None:
        clauses.append("d.score >= %s")
        params.append(min_score)
    if max_score is not None:
        clauses.append("d.score <= %s")
        params.append(max_score)
    params.extend([limit, offset])

    det_rows = run_query(
        f"""
        SELECT d.id, d.image_id, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
               d.category_id, d.category_name, d.score, d.genus, d.genus_confidence,
               d.species, d.species_confidence, d.crop_thumb_uri
        FROM detections d
        JOIN images i ON i.id = d.image_id
        WHERE {' AND '.join(clauses)}
        ORDER BY d.id
        LIMIT %s OFFSET %s
        """,
        tuple(params),
    )
    annotations = [
        {
            "id": d["id"],
            "image_id": d["image_id"],
            "bbox": [d["bbox_x"], d["bbox_y"], d["bbox_w"], d["bbox_h"]],
            "category_id": d["category_id"],
            "category_name": d["category_name"],
            "score": d["score"],
            "genus": d["genus"],
            "genus_confidence": d["genus_confidence"],
            "species": d["species"],
            "species_confidence": d["species_confidence"],
            "crop_thumb": thumb_proxy_url(d["crop_thumb_uri"]),
        }
        for d in det_rows
    ]

    return {
        "run_card": run["run_card_uri"],
        "model_type": run["model_type"],
        "created_at": run["created_at"],
        "stereo_side": run["stereo_side"],
        "input_file": run["input_file_uri"],
        "output_file": run["output_file_uri"],
        "yaml_config_snapshot": run["yaml_config_snapshot"],
        "images": images,
        "annotations": annotations,
    }
