-- Phase 2: reviews becomes a full history (1:many per detection)
-- instead of 1:1. A second review of the same detection -- a
-- different auditor, or the same one changing their mind -- now adds
-- a new row instead of overwriting the old one, so disagreement
-- between auditors (or a changed call) is itself preserved data, not
-- silently discarded. Jeremy's call (2026-10-02): with 2-3 auditors
-- and an explicit goal of tracking performance over time, losing that
-- history wasn't acceptable even though the original schema.sql
-- (2026-09-30) deliberately left it out as "not asked for yet."
--
-- Run this once against any database created from schema.sql before
-- this migration existed. A database created from the current
-- schema.sql already has this shape -- do not run it there.
--
-- Safe to run even if `reviews` already has rows: it doesn't touch
-- existing data, only the constraint and the views built on top of it.

ALTER TABLE reviews DROP CONSTRAINT reviews_detection_id_key;

CREATE INDEX IF NOT EXISTS idx_reviews_detection_latest ON reviews(detection_id, reviewed_at DESC);

-- The single current verdict per detection -- every other view/query
-- that wants "the review" rather than "all reviews" reads this,
-- rather than re-deriving DISTINCT ON logic in multiple places.
CREATE OR REPLACE VIEW latest_reviews AS
SELECT DISTINCT ON (detection_id) *
FROM reviews
ORDER BY detection_id, reviewed_at DESC, id DESC;

DROP VIEW IF EXISTS retraining_labels;
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
