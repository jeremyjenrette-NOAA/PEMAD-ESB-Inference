"""Thin wrappers around google-cloud-storage for the paths this CLI needs."""
from __future__ import annotations

import json
from typing import List, Tuple, Union

from google.cloud import storage

_client = None


def _get_client() -> storage.Client:
    global _client
    if _client is None:
        _client = storage.Client()
    return _client


def _split_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {gs_uri}")
    without_scheme = gs_uri[len("gs://"):]
    bucket_name, _, blob_path = without_scheme.partition("/")
    if not blob_path:
        raise ValueError(f"gs:// URI has no object path: {gs_uri}")
    return bucket_name, blob_path


def download_text(gs_uri: str) -> str:
    bucket_name, blob_path = _split_uri(gs_uri)
    blob = _get_client().bucket(bucket_name).blob(blob_path)
    return blob.download_as_text()


def upload_text(text: str, gs_uri: str, content_type: str = "text/plain") -> str:
    bucket_name, blob_path = _split_uri(gs_uri)
    blob = _get_client().bucket(bucket_name).blob(blob_path)
    blob.upload_from_string(text, content_type=content_type)
    return gs_uri


def upload_json(payload: Union[dict, list], gs_uri: str) -> str:
    return upload_text(json.dumps(payload, indent=2), gs_uri, content_type="application/json")


def upload_file(local_path: str, gs_uri: str) -> str:
    bucket_name, blob_path = _split_uri(gs_uri)
    blob = _get_client().bucket(bucket_name).blob(blob_path)
    blob.upload_from_filename(local_path)
    return gs_uri


def copy_blob(source_uri: str, dest_uri: str) -> str:
    """Server-side copy of one GCS object to another location -- no data
    is downloaded or re-uploaded, so this is fast and has no egress cost,
    even across buckets. Used by `input_staging.py` to flatten a
    multi-folder input list into one folder before a run is triggered, as
    a client-side workaround for the inference wrapper's folder-mixing
    bug (see survey_sampler.py's module docstring for the confirmed root
    cause).
    """
    src_bucket_name, src_blob_path = _split_uri(source_uri)
    dest_bucket_name, dest_blob_path = _split_uri(dest_uri)
    client = _get_client()
    src_bucket = client.bucket(src_bucket_name)
    src_blob = src_bucket.blob(src_blob_path)
    dest_bucket = client.bucket(dest_bucket_name)
    src_bucket.copy_blob(src_blob, dest_bucket, dest_blob_path)
    return dest_uri


def list_prefix_with_sizes(gs_uri_prefix: str) -> List[Tuple[str, int]]:
    """Like `list_prefix`, but also returns each object's size in bytes.
    Free relative to `list_prefix` -- GCS's `list_blobs` already returns
    this metadata per blob, this just doesn't throw it away. Used by
    `survey_sampler.py` to warn when a sample is large enough to cross a
    known problem boundary in the inference container's own chunked
    downloader (see that module's docstring).
    """
    bucket_name, blob_path = _split_uri(gs_uri_prefix)
    blobs = _get_client().list_blobs(bucket_name, prefix=blob_path)
    return [(f"gs://{bucket_name}/{b.name}", b.size or 0) for b in blobs]


def list_prefix(gs_uri_prefix: str) -> List[str]:
    return [path for path, _size in list_prefix_with_sizes(gs_uri_prefix)]
