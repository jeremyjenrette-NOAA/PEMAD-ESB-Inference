-- Results-auditor schema (v1) -- backs the point-and-click detection
-- viewer/auditor Jeremy described: ingest a run's predictions once,
-- browse/filter them at query time instead of re-deriving a flat
-- manifest.json per run, and capture reviewer decisions so confirmed/
-- corrected labels can be pulled back out for retraining.
--
-- Design notes:
--   * Images/crops themselves stay in GCS (already true today) --
--     this schema stores gs:// URIs, never image bytes, so Postgres
--     stays small even as prediction volume grows.
--   * `detections` denormalizes run_id (not just image_id) because
--     "show me everything from run X" is the primary query pattern
--     for the viewer -- avoids an image join on the hot path.
--   * `reviews` is 1:many per detection (as of migrations/0002) --
--     every confirm/reject/relabel action adds a row rather than
--     overwriting the last one, so a disagreement between auditors or
--     a changed call is preserved, not discarded. `latest_reviews`
--     resolves "the current verdict" (DISTINCT ON, most recent); the
--     original schema here had this 1:1 with a UNIQUE constraint --
--     changed once Jeremy confirmed performance-tracking needed the
--     full history, not just the latest state.
--   * category_id/category_name are kept even though today's model
--     only emits one category ("cancer_crab") -- known to be a
--     leftover/misleading label from the training data (see prior
--     notes on the cancer_crab embedded class-name issue), kept as
--     raw provenance rather than silently dropped.
--
-- Verified 2026-09-30 by loading Jeremy's real star-cascade_20260930_left_v2
-- export (71 images, 80 detections) through the exact INSERT shapes
-- db_ingest.load_into_postgres() uses -- row counts, FK integrity
-- (every detection's image belongs to its own run), and (on that
-- version of this schema, since superseded by migrations/0002 below)
-- the reviews 1:1 UNIQUE constraint and retraining_labels view, all
-- confirmed against that real data before this was committed.

CREATE TABLE runs (
    id                  BIGSERIAL PRIMARY KEY,
    run_card_uri        TEXT NOT NULL UNIQUE,
    model_type          TEXT NOT NULL,
    stereo_side         TEXT,                    -- 'left' | 'right' | 'full' | NULL
    created_at          TIMESTAMPTZ NOT NULL,     -- from the run card (when the run was triggered)
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    input_file_uri      TEXT,                     -- the input manifest this run read
    output_file_uri     TEXT,                     -- the annotations.json this run wrote
    yaml_config_snapshot TEXT,                     -- exact config text used, for provenance
    image_count         INTEGER,
    detection_count      INTEGER
);

CREATE TABLE images (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    file_name           TEXT NOT NULL,            -- bare basename, as annotations.json has it
    source_uri          TEXT,                     -- full gs:// path to the ORIGINAL (unsplit) frame
    width               INTEGER NOT NULL,
    height              INTEGER NOT NULL,
    stereo_side         TEXT,                     -- this image's own recorded side (usually == runs.stereo_side)
    stereo_crop_x_offset INTEGER NOT NULL DEFAULT 0,
    context_thumb_uri   TEXT,                     -- gs:// path to the rendered eye-crop thumbnail
    UNIQUE (run_id, file_name)
);
CREATE INDEX idx_images_run ON images(run_id);

CREATE TABLE detections (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    image_id            BIGINT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    bbox_x              DOUBLE PRECISION NOT NULL,
    bbox_y              DOUBLE PRECISION NOT NULL,
    bbox_w              DOUBLE PRECISION NOT NULL,
    bbox_h              DOUBLE PRECISION NOT NULL,
    category_id         INTEGER,
    category_name       TEXT,
    score               DOUBLE PRECISION NOT NULL,
    genus               TEXT,
    genus_confidence     DOUBLE PRECISION,
    species             TEXT,
    species_confidence   DOUBLE PRECISION,
    crop_thumb_uri       TEXT                      -- gs:// path to the rendered per-detection crop
);
CREATE INDEX idx_detections_run ON detections(run_id);
CREATE INDEX idx_detections_image ON detections(image_id);
CREATE INDEX idx_detections_genus ON detections(genus);
CREATE INDEX idx_detections_score ON detections(score);

CREATE TYPE review_decision AS ENUM ('confirmed', 'rejected', 'relabeled', 'uncertain');

CREATE TABLE reviews (
    id                  BIGSERIAL PRIMARY KEY,
    detection_id         BIGINT NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
    decision            review_decision NOT NULL,
    corrected_genus      TEXT,                     -- set only when decision = 'relabeled'
    corrected_species    TEXT,
    notes               TEXT,
    reviewer            TEXT NOT NULL,             -- email/name; no auth system yet -- see the architecture doc
    reviewed_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_reviews_decision ON reviews(decision);
CREATE INDEX idx_reviews_detection_latest ON reviews(detection_id, reviewed_at DESC);

-- "The current verdict" per detection -- DISTINCT ON the most recent
-- reviewed_at (id as a tiebreaker for same-timestamp rows). Every
-- consumer that wants "the review", not "all reviews for this
-- detection", reads this rather than re-deriving the same DISTINCT ON.
CREATE VIEW latest_reviews AS
SELECT DISTINCT ON (detection_id) *
FROM reviews
ORDER BY detection_id, reviewed_at DESC, id DESC;

-- Retraining export reads from this view rather than the raw tables,
-- so "what does a corrected label actually mean" only has to be
-- defined once: relabeled -> the correction; confirmed -> the
-- original prediction; rejected/uncertain/no (current) review -> excluded.
-- Reads the *latest* verdict only -- an earlier, superseded review on
-- the same detection never contributes a label here even though it's
-- still in `reviews` for history/agreement purposes.
CREATE VIEW retraining_labels AS
SELECT
    d.id AS detection_id,
    d.run_id,
    d.image_id,
    d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
    COALESCE(lr.corrected_genus, d.genus) AS genus,
    COALESCE(lr.corrected_species, d.species) AS species,
    lr.decision,
    lr.reviewer,
    lr.reviewed_at
FROM detections d
JOIN latest_reviews lr ON lr.detection_id = d.id
WHERE lr.decision IN ('confirmed', 'relabeled');
