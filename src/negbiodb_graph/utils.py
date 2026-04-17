"""Utility helpers for graph normalization, hashing, and JSON encoding."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def stable_json_dumps(value: Any) -> str:
    """Return deterministic JSON for hashing and persistence."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha1_text(text: str) -> str:
    """Return the SHA1 digest of a text payload."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, max_bytes: int = 50_000_000) -> str | None:
    """Return SHA256 for moderately sized files, else None to avoid expensive scans."""
    if not path.exists() or not path.is_file():
        return None
    if path.stat().st_size > max_bytes:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str | None) -> str | None:
    """Normalize a free-text identifier for matching."""
    if value is None:
        return None
    norm = _WS_RE.sub(" ", str(value).strip())
    return norm or None


def normalize_name_key(value: str | None) -> str | None:
    """Normalize a display name to a matching-friendly lowercase key."""
    norm = normalize_text(value)
    if norm is None:
        return None
    lowered = _NON_ALNUM_RE.sub("_", norm.lower()).strip("_")
    return lowered or None


def context_hash(payload: dict[str, Any]) -> str:
    """Hash a context dictionary into a stable content-addressed identifier."""
    return sha1_text(stable_json_dumps(payload))


def anchor_key(*parts: str | None) -> str:
    """Join normalized anchor parts, skipping missing components."""
    clean = [part for part in parts if part]
    return "|".join(clean)


def as_jsonable(value: Any) -> Any:
    """Convert nested values into JSON-friendly structures."""
    if isinstance(value, dict):
        return {str(key): as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
