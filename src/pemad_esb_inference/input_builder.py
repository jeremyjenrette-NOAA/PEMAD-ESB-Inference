"""Build the model-family-specific input manifest JSON that the DAG's
`input_file` parameter expects.

Schemas confirmed from the User Guide and the "Bring Your Self-Trained
VIAME Model to OSI" appendix:

  Ultralytics-family:   {"instances": ["gs://...", ...]}
  VIAME-family images:  {"instances": [{"input_images": ["gs://...", ...]}]}
  VIAME-family video:   {"instances": [{"input_path": "gs://.../video.mp4"}]}
  VIAME-family stereo
    (paired lists):      {"instances": [{"input_images": [[left...], [right...]]}]}
  VIAME-family stereo
    (left/right dirs):   {"instances": [{"input_paths": ["gs://left_dir/", "gs://right_dir/"]}]}
"""
from __future__ import annotations

from typing import Dict, Sequence


def ultralytics_manifest(image_or_video_paths: Sequence[str]) -> Dict:
    return {"instances": list(image_or_video_paths)}


def viame_images_manifest(image_paths: Sequence[str]) -> Dict:
    return {"instances": [{"input_images": list(image_paths)}]}


def viame_video_manifest(video_path: str) -> Dict:
    return {"instances": [{"input_path": video_path}]}


def viame_stereo_images_manifest(left_paths: Sequence[str], right_paths: Sequence[str]) -> Dict:
    return {"instances": [{"input_images": [list(left_paths), list(right_paths)]}]}


def viame_stereo_folders_manifest(left_folder: str, right_folder: str) -> Dict:
    return {"instances": [{"input_paths": [left_folder, right_folder]}]}


def manifest_for_family(family: str, image_paths: Sequence[str]) -> Dict:
    """Pick the right manifest shape for a model family.

    Only covers the common single-camera-images case (the one the
    `trigger` / `build-input` CLI commands drive today) -- call the
    `viame_*` builders above directly for video/stereo inputs.
    """
    if family == "ultralytics":
        return ultralytics_manifest(image_paths)
    if family == "viame":
        return viame_images_manifest(image_paths)
    raise ValueError(f"Unknown model family '{family}' (expected 'ultralytics' or 'viame')")
