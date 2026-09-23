"""Sample a survey folder's images at a 1-in-N rate.

Stratified per leaf folder: HabCam's landing bucket nests images several
levels deep (.../<YYYY>/<YYYYMM>/<YYYYMMDD>/<YYYYMMDD>_<HH>/<YYYYMMDD>_<HH><MM>/*.png),
and folder sizes vary a lot. Sampling every Nth image *within each leaf
folder* -- rather than across one big combined listing -- keeps the
sample spread evenly across the whole time range of the prefix, instead
of being dominated by whichever folder happens to have the most images.

Leaf-folder naming: a standard leaf folder's basename is `<YYYYMMDD>_<HHMM>`
(e.g. `20230615_0500`), matching that nested structure. A whole-prefix
recursive listing also picks up folders that don't follow it -- e.g. a
flat `auv` folder sitting directly under the year prefix (no date/hour
nesting at all), or a `<YYYYMMDD>_orig` variant alongside the normal
per-minute buckets. Non-conforming folders are excluded from sampling **by
default** (`is_conforming_leaf_folder` gates this) and reported separately
via `SampleReport.skipped_folders` so a run's `--dry-run` output (or any
`trigger`/`build-input`, which always prints the summary) surfaces them
instead of silently including whatever they contain. Pass
`include_nonconforming=True` to `sample_survey_folder` (or
`--include-nonconforming-folders` on the CLI) once a given anomalous
folder is confirmed safe to sample from. This is worth doing regardless
of the point below -- these folders are still structurally different from
the documented layout -- but see that point for what actually turned out
to be causing the failure that first surfaced this.

**Known issue, root cause confirmed 2026-09-21 by reading the wrapper's
own source** (pulled directly from the live `ultralytics-fish-segmentation-bw-wrapper`
image -- not guessed from logs). Short version: the wrapper downloads
images in ~500MB batches (`MAX_CHUNK_BYTES`) and, for each batch, tries to
compute one shared GCS folder that all of that batch's images live under.
If a batch's images span **more than one GCS folder**, that computation
gives up and points the model at the download's top-level temp folder
instead of where the images actually landed (nested under their original
per-image subfolders) -- so the model finds nothing there and crashes
with `FileNotFoundError: No images or videos found in ...`, even though
the download itself succeeded. This is a bug in the wrapper
(`app_utils/gcs_utils.py:download_uri_list_to_local`), not in this repo,
and this CLI has no way to fix it directly -- documented here because it
changes what's actually safe to submit. Two earlier, less certain
descriptions of this failure (a folder-naming theory, then a
timing/race-condition guess) are both superseded by this one; see
`docs/architecture.md`'s "Known limitations" and the Claude Project notes
for the full history.

What this module *can* do about it: `sample_survey_folder` estimates the
sampled data's total size (via `gcs.list_prefix_with_sizes`, free -- GCS's
listing API already returns object sizes) and warns in
`SampleReport.summary()` when a sample is large enough to likely cross
the ~500MB batch boundary. This is *necessary but not sufficient*: since
the real trigger is a batch spanning multiple folders (not just size), a
sample is only truly safe if every image in it shares one GCS folder, or
if the whole sample is small enough to form a single batch on the
wrapper's side *and* still shares one folder. A `--survey-prefix` sample
drawn from many leaf folders is the highest-risk case, since the wrapper
will eventually hit a batch boundary that crosses two folders regardless
of `--sample-rate`. This is advisory only (nothing is capped or blocked)
-- until the wrapper's bug is fixed upstream, the safest workaround is
sampling from a single leaf folder at a time, or keeping `--images`/
`--habcam` lists confined to one folder.

**Update 2026-09-21**: `pemad-infer trigger` now does this automatically.
When the resolved image list spans more than one GCS folder, it copies
every image into one flat, run-specific staging folder before submitting
(a cheap server-side GCS copy, no download/upload) -- see
`input_staging.py`. This removes the folder-crossing failure mode for any
run triggered through this CLI, regardless of size or `--sample-rate`,
without needing a fix from whoever owns the wrapper image. Pass
`--no-flatten-inputs` to skip it. The size-based warning above still
matters even with flattening on: staging doesn't shrink the run, so a
very large single-folder batch can still be slow or expensive to run,
just not crash on the folder-mixing bug specifically.

`group_by_folder` and `sample_groups` are pure functions (no GCS calls)
so they're unit-testable without credentials; `sample_survey_folder`
composes them (plus the conforming/non-conforming partition and the
size-based warning) with a real `list_prefix_with_sizes` call.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Standard HabCam leaf-folder basename: <YYYYMMDD>_<HHMM>, e.g. "20230615_0500".
LEAF_FOLDER_NAME_RE = re.compile(r"^\d{8}_\d{4}$")

# The inference wrapper container chunks its own GCS downloads at this size
# (observed directly in Cloud Batch job logs: `MAX_CHUNK_BYTES: 524288000`)
# and, as of 2026-09-17, has twice reliably failed on the *second* such
# chunk regardless of its contents -- see the module docstring. This isn't
# configurable from here; it's just what the container currently uses.
KNOWN_WRAPPER_CHUNK_BYTES = 524_288_000


def _format_bytes(n: int) -> str:
    return f"{n / (1024 * 1024):.0f}MB"


@dataclass
class SampleReport:
    prefix: str
    sample_rate: int
    folders_seen: int
    total_images: int
    sampled_images: int
    per_folder: Dict[str, Tuple[int, int]]  # folder -> (found, sampled) -- conforming folders only
    skipped_folders: Dict[str, int] = field(default_factory=dict)  # folder -> images found, excluded from sampling
    sampled_bytes: int = 0  # total size of the sampled images, from GCS object metadata

    def summary(self) -> str:
        base = (
            f"Survey sample of {self.prefix}: {self.folders_seen} folder(s), "
            f"{self.total_images} image(s) found, {self.sampled_images} sampled "
            f"(1-in-{self.sample_rate})."
        )
        if self.skipped_folders:
            skipped_images = sum(self.skipped_folders.values())
            base += (
                f" Skipped {len(self.skipped_folders)} non-conforming folder(s), "
                f"{skipped_images} image(s) NOT sampled (pass --include-nonconforming-folders "
                f"to include them once confirmed safe): " + ", ".join(sorted(self.skipped_folders))
            )
        if self.sampled_bytes > KNOWN_WRAPPER_CHUNK_BYTES:
            base += (
                f" WARNING: sampled data is ~{_format_bytes(self.sampled_bytes)}, which spans more "
                f"than one ~{_format_bytes(KNOWN_WRAPPER_CHUNK_BYTES)} download chunk in the "
                f"inference container -- runs this large have twice failed partway through the "
                f"second chunk (see survey_sampler.py's module docstring). Consider a higher "
                f"--sample-rate (or a smaller --images/--habcam list) until that's fixed upstream."
            )
        return base


def is_conforming_leaf_folder(folder: str) -> bool:
    """True if `folder`'s basename matches the standard HabCam leaf-folder
    naming convention, `<YYYYMMDD>_<HHMM>` (see module docstring). `folder`
    is a full gs:// folder path (or any `/`-separated path) -- only its
    final segment is checked.
    """
    basename = folder.rsplit("/", 1)[-1]
    return bool(LEAF_FOLDER_NAME_RE.match(basename))


def group_by_folder(paths: List[str], extensions=DEFAULT_IMAGE_EXTENSIONS) -> Dict[str, List[str]]:
    """Bucket a flat list of gs:// paths by their immediate parent folder,
    keeping only files with an image extension. Includes every folder
    found, conforming or not -- naming-convention filtering happens in
    `sample_survey_folder`, not here, so this stays a simple, total
    bucketing function callers can rely on.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for path in paths:
        if not path.lower().endswith(extensions):
            continue
        folder = path.rsplit("/", 1)[0]
        groups[folder].append(path)
    return groups


def sample_groups(groups: Dict[str, List[str]], sample_rate: int) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """Given {folder: [paths]}, take every `sample_rate`-th path (sorted)
    independently within each folder. Returns (sampled_paths, per_folder_report).
    """
    if sample_rate < 1:
        raise ValueError("sample_rate must be >= 1")

    sampled: List[str] = []
    per_folder: Dict[str, Tuple[int, int]] = {}
    for folder in sorted(groups):
        folder_paths = sorted(groups[folder])
        chosen = folder_paths[::sample_rate]
        per_folder[folder] = (len(folder_paths), len(chosen))
        sampled.extend(chosen)
    return sampled, per_folder


def sample_survey_folder(
    prefix_uri: str,
    sample_rate: int,
    extensions=DEFAULT_IMAGE_EXTENSIONS,
    include_nonconforming: bool = False,
) -> Tuple[List[str], SampleReport]:
    """List every image under `prefix_uri` and return every `sample_rate`-th
    one (e.g. sample_rate=5 -> 1-in-5), sampled independently per leaf
    folder. Returns (sampled_paths, SampleReport).

    By default, only folders matching the standard `<YYYYMMDD>_<HHMM>`
    leaf-folder naming convention are sampled from -- see the module
    docstring for why (a flat `auv` folder and similar anomalies get swept
    up by the recursive listing but aren't confirmed to behave the same
    way downstream). They're reported, not silently dropped:
    `SampleReport.skipped_folders` lists each non-conforming folder found
    and how many images it has. Pass `include_nonconforming=True` to
    sample from them anyway.

    NOTE: this lists the *entire* prefix before sampling (GCS has no
    server-side "every Nth object" listing), which can be slow/costly for
    a whole-year prefix with hundreds of thousands of objects. Always run
    with `--dry-run` first (via the CLI) to see the SampleReport before
    committing to a real trigger.
    """
    # Deferred import: keeps group_by_folder/sample_groups (the pure,
    # unit-testable logic) importable without google-cloud-storage
    # installed -- only this function actually needs a live GCS call.
    from .gcs import list_prefix_with_sizes

    listed = list_prefix_with_sizes(prefix_uri)
    sizes_by_path = dict(listed)
    all_paths = [path for path, _size in listed]
    groups = group_by_folder(all_paths, extensions)

    if include_nonconforming:
        conforming_groups = groups
        skipped_folders: Dict[str, int] = {}
    else:
        conforming_groups = {
            folder: paths for folder, paths in groups.items() if is_conforming_leaf_folder(folder)
        }
        skipped_folders = {
            folder: len(paths) for folder, paths in groups.items() if folder not in conforming_groups
        }

    sampled, per_folder = sample_groups(conforming_groups, sample_rate)
    total_images = sum(len(paths) for paths in groups.values())
    sampled_bytes = sum(sizes_by_path.get(path, 0) for path in sampled)
    report = SampleReport(
        prefix=prefix_uri,
        sample_rate=sample_rate,
        folders_seen=len(groups),
        total_images=total_images,
        sampled_images=len(sampled),
        per_folder=per_folder,
        skipped_folders=skipped_folders,
        sampled_bytes=sampled_bytes,
    )
    return sampled, report
