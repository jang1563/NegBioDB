"""Tests for HMDB download helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parent.parent / "scripts_md" / "01_download_hmdb.py"
    spec = importlib.util.spec_from_file_location("download_hmdb_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_existing_hmdb_zip_prefers_explicit_path(tmp_path, monkeypatch):
    module = _load_module()
    explicit_zip = tmp_path / "hmdb_metabolites.zip"
    explicit_zip.write_bytes(b"zip")

    downloads_dir = tmp_path / "Downloads"
    downloads_dir.mkdir()
    (downloads_dir / "hmdb_metabolites_browser.zip").write_bytes(b"other")

    monkeypatch.setenv("HMDB_DOWNLOADS_DIR", str(downloads_dir))
    found = module.resolve_existing_hmdb_zip(explicit_zip=explicit_zip, search_downloads=True)

    assert found == explicit_zip


def test_resolve_existing_hmdb_zip_searches_downloads_dir(tmp_path, monkeypatch):
    module = _load_module()
    downloads_dir = tmp_path / "Downloads"
    downloads_dir.mkdir()
    browser_zip = downloads_dir / "hmdb_metabolites (1).zip"
    browser_zip.write_bytes(b"zip")

    # Point the default project-data zip path into tmp_path so a real zip
    # in ./data/ cannot shadow the downloads-dir probe.
    monkeypatch.setattr(module, "HMDB_ZIP_PATH", tmp_path / "missing.zip")
    monkeypatch.setenv("HMDB_DOWNLOADS_DIR", str(downloads_dir))
    monkeypatch.delenv("HMDB_ZIP_PATH", raising=False)
    found = module.resolve_existing_hmdb_zip(explicit_zip=None, search_downloads=True)

    assert found == browser_zip


def test_is_cloudflare_challenge_detects_header_and_body():
    module = _load_module()

    assert module.is_cloudflare_challenge({"cf-mitigated": "challenge"}, b"")
    assert module.is_cloudflare_challenge({}, b"Just a moment... __cf_chl_opt")
    assert not module.is_cloudflare_challenge({}, b"regular zip bytes")
