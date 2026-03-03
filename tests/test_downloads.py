"""Tests for NegBioDB download utilities."""

from pathlib import Path

import pytest

from negbiodb.download import (
    check_disk_space,
    load_config,
    verify_file_exists,
)


class TestLoadConfig:

    def test_load_config_returns_dict(self):
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "downloads" in cfg

    def test_config_has_all_sources(self):
        cfg = load_config()
        downloads = cfg["downloads"]
        for source in ["pubchem", "chembl", "bindingdb", "davis"]:
            assert source in downloads, f"Missing download config: {source}"

    def test_config_pubchem_has_url(self):
        cfg = load_config()
        assert cfg["downloads"]["pubchem"]["url"].startswith("ftp://")

    def test_config_davis_has_files(self):
        cfg = load_config()
        davis = cfg["downloads"]["davis"]
        assert "files" in davis
        assert len(davis["files"]) == 3


class TestDiskSpace:

    def test_sufficient_space(self, tmp_path):
        # Current disk should have at least 0.001 GB free
        check_disk_space(tmp_path, required_gb=0.001)

    def test_insufficient_space_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Insufficient disk space"):
            check_disk_space(tmp_path, required_gb=999999)

    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "deep" / "nested" / "dir"
        check_disk_space(new_dir, required_gb=0.001)
        assert new_dir.exists()


class TestFileVerification:

    def test_verify_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        assert verify_file_exists(f) is True

    def test_verify_missing_file(self, tmp_path):
        assert verify_file_exists(tmp_path / "nonexistent.txt") is False

    def test_verify_small_file(self, tmp_path):
        f = tmp_path / "small.txt"
        f.write_text("hi")
        assert verify_file_exists(f, min_bytes=1000) is False

    def test_verify_file_meets_min_size(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_bytes(b"x" * 1000)
        assert verify_file_exists(f, min_bytes=1000) is True

    def test_verify_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.touch()
        assert verify_file_exists(f) is True
        assert verify_file_exists(f, min_bytes=1) is False


@pytest.mark.integration
class TestDavisDownload:
    """Integration test — requires network access. Run with: pytest -m integration"""

    def test_davis_download(self, tmp_path):
        import pandas as pd
        from negbiodb.download import download_file_http

        cfg = load_config()
        davis = cfg["downloads"]["davis"]
        base_url = davis["base_url"]

        # Download only drugs.csv (smallest file)
        url = f"{base_url}/drugs.csv"
        dest = tmp_path / "drugs.csv"
        download_file_http(url, dest, desc="drugs.csv")

        assert dest.exists()
        df = pd.read_csv(dest)
        assert len(df) == 68
        assert "Canonical_SMILES" in df.columns
