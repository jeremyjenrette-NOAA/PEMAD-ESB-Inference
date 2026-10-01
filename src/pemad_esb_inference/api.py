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

Known gap, deliberately not solved here: `context_thumb_uri` /
`crop_thumb_uri` are raw gs:// paths once a run has been ingested with
real GCS upload (not --skip-upload) -- a browser can't load those
directly. Resolving them to viewable URLs (signed URLs, or a public/
authenticated-read bucket) is a Phase 1 follow-up once the read shape
below is confirmed to be the right one; deliberately not guessed at
here since it's a real decision (signed-URL expiry + who can generate
them) rather than a mechanical detail.
"""
from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

DSN_ENV_VAR = "PEMAD_DB_DSN"
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/viewer_dev"

app = FastAPI(title="PEMAD Results Auditor API", version="0.1.0")

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


@app.get("/health")
def health():
    return {"status": "ok"}


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
    """Manifest-shaped payload for one run: {run, images, annotations}.
    Deliberately the same shape the viewer's manifest.json already has
    (images + annotations, bbox as [x, y, w, h]) so porting the viewer
    to call this instead of fetch('manifest.json') is a data-source
    swap, not a rewrite of the render/filter logic already built.

    Filters apply only to `annotations` -- `images` is always returned
    in full for the run, since the detail panel needs every detection's
    sibling frame regardless of which detections are currently filtered
    in."""
    run_rows = run_query("SELECT * FROM runs WHERE id = %s", (run_id,))
    if not run_rows:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    run = run_rows[0]

    images = run_query(
        """
        SELECT id, file_name, source_uri, width, height, stereo_side,
               stereo_crop_x_offset, context_thumb_uri
        FROM images WHERE run_id = %s ORDER BY id
        """,
        (run_id,),
    )

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

    detections = run_query(
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
            "crop_thumb_uri": d["crop_thumb_uri"],
        }
        for d in detections
    ]

    return {"run": run, "images": images, "annotations": annotations}
