"""Reference-manifest helpers for optional positive and external feeds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReferenceFeed:
    name: str
    kind: str
    path: Path
    domain_code: str | None = None
    description: str | None = None
    required: bool = False


def load_reference_manifest(path: str | Path | None) -> list[ReferenceFeed]:
    """Load a reference manifest describing optional external feeds."""
    if path is None:
        return []
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    with manifest_path.open() as handle:
        payload = json.load(handle)
    feeds = payload.get("feeds", [])
    result = []
    for item in feeds:
        feed_path = Path(item["path"])
        if not feed_path.is_absolute():
            feed_path = (manifest_path.parent / feed_path).resolve()
        result.append(
            ReferenceFeed(
                name=item["name"],
                kind=item["kind"],
                path=feed_path,
                domain_code=item.get("domain_code"),
                description=item.get("description"),
                required=bool(item.get("required", False)),
            )
        )
    return result


def manifest_summary(feeds: list[ReferenceFeed]) -> list[dict[str, Any]]:
    """Return a JSON-friendly manifest summary."""
    return [
        {
            "name": feed.name,
            "kind": feed.kind,
            "path": str(feed.path),
            "domain_code": feed.domain_code,
            "description": feed.description,
            "required": feed.required,
            "available": feed.path.exists(),
        }
        for feed in feeds
    ]
