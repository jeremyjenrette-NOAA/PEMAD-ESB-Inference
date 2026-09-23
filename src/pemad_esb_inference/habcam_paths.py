"""Resolve HabCam image filenames to their expected GCS paths.

Ported from Jeremy Jenrette's original `build_img_path.py` workstation
script. HabCam filenames encode date/time directly
(e.g. "202403.20240508.142352788.7482.png"), and the landing bucket uses
a predictable folder layout built from that timestamp, so given just the
filename we can compute where the image *should* already live -- no
directory listing required.

CAVEAT (see docs/architecture.md "Known limitations"): this layout
hasn't been confirmed against a real production image path yet -- the
one successful end-to-end test so far (2026-09-17) used a test image
placed directly at the landing bucket's root rather than under this
computed dated subfolder structure. Confirm the real layout before
relying on this for anything beyond another plumbing test.
"""
from __future__ import annotations

import os
import posixpath
from typing import Dict, List, Optional

from . import config


def parse_imagename(imagename: str) -> Dict[str, Optional[str]]:
    """Parse a HabCam image filename into its date/time components.

    Example: "202403.20240508.142352788.7482.png"
    """
    imagename = os.path.basename(imagename.strip())
    parts = imagename.split(".")

    if len(parts) < 4:
        return {
            "imagename": imagename,
            "date_yyyymmdd": None,
            "time_token": None,
            "hour": None,
            "minute": None,
            "tenmin": None,
        }

    date_yyyymmdd = parts[1]
    time_token = parts[2]
    hour = time_token[:2]
    minute = time_token[2:4]

    try:
        tenmin = f"{(int(minute) // 10) * 10:02d}"
    except (ValueError, TypeError):
        tenmin = None

    return {
        "imagename": imagename,
        "date_yyyymmdd": date_yyyymmdd,
        "time_token": time_token,
        "hour": hour,
        "minute": minute,
        "tenmin": tenmin,
    }


def build_expected_path(imagename: str, base_path: str = config.HABCAM_BASE_PATH) -> Optional[str]:
    """Construct the expected GCS path for a HabCam image filename.

    Folder layout:
    <base>/<YEAR>/<YYYYMM>/<YYYYMMDD>/<YYYYMMDD>_<HH>/<YYYYMMDD>_<HH><TENMIN>/<imagename>
    """
    p = parse_imagename(imagename)
    if not p["date_yyyymmdd"] or not p["hour"] or p["tenmin"] is None:
        return None

    date_str = p["date_yyyymmdd"]
    f_year = date_str[:4]
    f_month = date_str[:6]
    f_day = date_str
    f_hour = f"{date_str}_{p['hour']}"
    f_tenmin = f"{date_str}_{p['hour']}{p['tenmin']}"

    base_clean = base_path.rstrip("/")
    return posixpath.join(base_clean, f_year, f_month, f_day, f_hour, f_tenmin, p["imagename"])


def resolve_paths(image_names: List[str], base_path: str = config.HABCAM_BASE_PATH) -> List[str]:
    """Resolve a list of filenames to GCS paths, skipping (and reporting)
    any that can't be parsed.
    """
    resolved = []
    for name in image_names:
        path = build_expected_path(name, base_path=base_path)
        if path:
            resolved.append(path)
        else:
            print(f"Warning: could not parse/resolve HabCam path for '{name}'. Skipping.")
    return resolved


def read_names_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
