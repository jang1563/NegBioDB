"""Tests for NegBioRL adapter merge helpers."""

from __future__ import annotations

import json
from pathlib import Path

from negbiorl.adapter_utils import (
    MERGE_METADATA_FILENAME,
    MERGE_FORMAT_VERSION,
    build_merge_metadata,
    get_merged_dir,
    merged_cache_is_fresh,
    resolve_adapter_dir,
    write_merge_metadata,
)


def _seed_adapter_dir(path: Path, base_model: str = "Qwen/Qwen2.5-7B-Instruct") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model})
    )
    (path / "adapter_model.safetensors").write_bytes(b"weights-v1")
    return path


class TestResolveAdapterDir:
    def test_prefers_final_subdir_when_present(self, tmp_path: Path):
        run_dir = tmp_path / "sft_run"
        final_dir = _seed_adapter_dir(run_dir / "final")

        assert resolve_adapter_dir(run_dir) == final_dir

    def test_returns_path_when_already_adapter_dir(self, tmp_path: Path):
        adapter_dir = _seed_adapter_dir(tmp_path / "adapter")

        assert resolve_adapter_dir(adapter_dir) == adapter_dir


class TestMergedCacheFreshness:
    def test_metadata_includes_merge_format_version(self, tmp_path: Path):
        adapter_dir = _seed_adapter_dir(tmp_path / "run" / "final")

        metadata = build_merge_metadata(adapter_dir)

        assert metadata["merge_format_version"] == MERGE_FORMAT_VERSION

    def test_cache_is_fresh_only_when_metadata_matches(self, tmp_path: Path):
        adapter_dir = _seed_adapter_dir(tmp_path / "run" / "final")
        metadata = build_merge_metadata(adapter_dir)
        merged_dir = get_merged_dir(adapter_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        (merged_dir / "config.json").write_text("{}")
        write_merge_metadata(merged_dir, metadata)

        assert merged_cache_is_fresh(merged_dir, metadata) is True

    def test_cache_becomes_stale_after_adapter_changes(self, tmp_path: Path):
        adapter_dir = _seed_adapter_dir(tmp_path / "run" / "final")
        metadata = build_merge_metadata(adapter_dir)
        merged_dir = get_merged_dir(adapter_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        (merged_dir / "config.json").write_text("{}")
        write_merge_metadata(merged_dir, metadata)

        (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights-v2")
        updated = build_merge_metadata(adapter_dir)

        assert merged_cache_is_fresh(merged_dir, updated) is False

    def test_cache_is_stale_without_metadata_file(self, tmp_path: Path):
        adapter_dir = _seed_adapter_dir(tmp_path / "run" / "final")
        metadata = build_merge_metadata(adapter_dir)
        merged_dir = get_merged_dir(adapter_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        (merged_dir / "config.json").write_text("{}")

        assert (merged_dir / MERGE_METADATA_FILENAME).exists() is False
        assert merged_cache_is_fresh(merged_dir, metadata) is False

    def test_cache_is_stale_when_metadata_version_changes(self, tmp_path: Path):
        adapter_dir = _seed_adapter_dir(tmp_path / "run" / "final")
        metadata = build_merge_metadata(adapter_dir)
        merged_dir = get_merged_dir(adapter_dir)
        merged_dir.mkdir(parents=True, exist_ok=True)
        (merged_dir / "config.json").write_text("{}")

        legacy_metadata = dict(metadata)
        legacy_metadata.pop("merge_format_version")
        write_merge_metadata(merged_dir, legacy_metadata)

        assert merged_cache_is_fresh(merged_dir, metadata) is False
