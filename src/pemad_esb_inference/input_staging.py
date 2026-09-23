"""Client-side workaround for the inference wrapper's folder-mixing bug.

Confirmed root cause (see `survey_sampler.py`'s module docstring for the
full writeup): the wrapper downloads inputs in ~500MB batches, and for
each batch tries to guess one shared GCS folder all of that batch's
files live under. If a batch's files span more than one GCS folder, that
guess fails and the model gets pointed at the wrong local directory,
crashing with `FileNotFoundError` even though the download itself
succeeded. This is a bug inside the wrapper image, not in this repo, and
this CLI has no way to patch it directly.

What this module does instead: before a run is submitted, copy every
selected input image into one flat, run-specific GCS folder, using a
cheap **server-side** copy (`Bucket.copy_blob` -- no download, no
upload, no egress cost, same primitive `optics-inference-runner`'s own
`perform_bulk_copy` helper uses for a different purpose). Since every
file then lives in exactly one folder, no batch the wrapper forms can
ever span two folders, regardless of how the source images were
originally organized or how large the run is. This removes the bug's
trigger condition entirely, rather than just narrowing the odds the way
staying under a size threshold does.

Scope: this only protects runs triggered through this CLI. It does
nothing for other consumers of the same wrapper image, and it's a
workaround, not a fix -- see survey_sampler.py's module docstring for
the suggested upstream fix and who owns it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

DEFAULT_MAX_WORKERS = 16


def spans_multiple_folders(paths: List[str]) -> bool:
    """True if `paths` (gs:// URIs) come from more than one immediate
    parent folder. "Folder" here means the same thing it does in
    `survey_sampler.group_by_folder`: everything before the final `/`.
    """
    folders = {path.rsplit("/", 1)[0] for path in paths}
    return len(folders) > 1


def _dedupe_name(basename: str, seen_counts: Dict[str, int]) -> str:
    """Return a destination basename guaranteed unique within one staging
    run. The common case (a filename seen for the first time) keeps the
    original basename unchanged, so most staged files still carry a
    meaningful name into the output annotations; a real collision (two
    source folders with a same-named file) gets a numeric suffix instead
    of silently overwriting one file with another.
    """
    count = seen_counts.get(basename, 0)
    seen_counts[basename] = count + 1
    if count == 0:
        return basename
    stem, dot, ext = basename.rpartition(".")
    return f"{stem}_{count}.{ext}" if dot else f"{basename}_{count}"


def plan_flat_names(paths: List[str]) -> List[Tuple[str, str]]:
    """Pure planning step (no GCS calls): for each source gs:// path,
    decide its de-duplicated destination basename. Returns a list of
    (source_uri, dest_basename) pairs, same order and length as `paths`
    -- kept separate from the actual copy so the naming logic is
    unit-testable without credentials.
    """
    seen_counts: Dict[str, int] = {}
    plan = []
    for path in paths:
        basename = path.rsplit("/", 1)[-1]
        dest_name = _dedupe_name(basename, seen_counts)
        plan.append((path, dest_name))
    return plan


def flatten_to_staging(
    paths: List[str], staging_prefix_uri: str, max_workers: int = DEFAULT_MAX_WORKERS
) -> List[str]:
    """Copy every path in `paths` into `staging_prefix_uri` (a gs://.../
    folder) via a server-side GCS copy, and return the new gs:// URIs in
    the same order as `paths`. See module docstring.
    """
    from . import gcs  # deferred: keeps plan_flat_names usable without google-cloud-storage

    prefix = staging_prefix_uri.rstrip("/") + "/"
    plan = plan_flat_names(paths)
    dest_uris = [f"{prefix}{dest_name}" for _source, dest_name in plan]

    def _copy_one(pair: Tuple[str, str]) -> str:
        source_uri, dest_uri = pair
        gcs.copy_blob(source_uri, dest_uri)
        return dest_uri

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # list()/map() preserves input order in its output, so dest_uris
        # (built above, same order as paths) is already the right return
        # value -- this just forces the copies to complete before we return.
        list(executor.map(_copy_one, zip(paths, dest_uris)))

    return dest_uris
