#!/usr/bin/env python3
"""Download HMDB XML and build internal standardization cache.

LICENSE NOTE: HMDB data is CC BY-NC 4.0 (non-commercial). This script builds
an INTERNAL cache only. HMDB-derived data is NEVER exported as part of the
benchmark. All redistributed metabolite identifiers come from PubChem.

Usage:
    python scripts_md/01_download_hmdb.py [--limit N]
"""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
HMDB_XML_URL = "https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip"
HMDB_XML_GZ = _PROJECT_ROOT / "data" / "hmdb_metabolites.xml.gz"
HMDB_ZIP_PATH = _PROJECT_ROOT / "data" / "hmdb_metabolites.zip"


class HMDBDownloadBlocked(RuntimeError):
    """Raised when HMDB blocks automated download access."""


def _manual_download_message() -> str:
    return (
        "HMDB automated download is blocked by Cloudflare. Download "
        "'All Metabolites' from https://www.hmdb.ca/downloads in a browser, "
        f"save the file as hmdb_metabolites.zip, then place it at {HMDB_ZIP_PATH} "
        "or set HMDB_ZIP_PATH=/path/to/hmdb_metabolites.zip and rerun."
    )


def _iter_candidate_zip_paths(
    explicit_zip: Path | None,
    search_downloads: bool,
) -> list[Path]:
    candidates: list[Path] = []

    def add(path: Path | None) -> None:
        if path is None:
            return
        resolved = path.expanduser()
        if resolved not in candidates:
            candidates.append(resolved)

    add(explicit_zip)
    env_zip = os.environ.get("HMDB_ZIP_PATH")
    if env_zip:
        add(Path(env_zip))
    add(HMDB_ZIP_PATH)

    if search_downloads:
        download_dirs = []
        env_downloads = os.environ.get("HMDB_DOWNLOADS_DIR")
        if env_downloads:
            download_dirs.append(Path(env_downloads))
        download_dirs.append(Path.home() / "Downloads")

        for download_dir in download_dirs:
            if not download_dir.exists():
                continue
            for pattern in ("hmdb_metabolites*.zip", "hmdb*.zip"):
                for match in sorted(download_dir.glob(pattern), reverse=True):
                    add(match)

    return candidates


def resolve_existing_hmdb_zip(
    explicit_zip: Path | None = None,
    search_downloads: bool = True,
) -> Path | None:
    for candidate in _iter_candidate_zip_paths(explicit_zip, search_downloads):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _copy_zip_if_needed(source_zip: Path, dest_zip: Path) -> Path:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if source_zip.resolve() == dest_zip.resolve():
        return dest_zip

    logger.info("Copying HMDB zip from %s to %s", source_zip, dest_zip)
    shutil.copy2(source_zip, dest_zip)
    return dest_zip


def is_cloudflare_challenge(headers: dict[str, str], body_prefix: bytes = b"") -> bool:
    if headers.get("cf-mitigated", "").lower() == "challenge":
        return True

    lower_prefix = body_prefix.lower()
    return any(
        marker in lower_prefix
        for marker in (
            b"just a moment",
            b"enable javascript and cookies to continue",
            b"__cf_chl_opt",
        )
    )


def download_hmdb_zip(
    dest_zip: Path,
    explicit_zip: Path | None = None,
    search_downloads: bool = True,
) -> Path:
    """Resolve or download the HMDB metabolites zip."""
    import requests

    try:
        import certifi
        verify = certifi.where()
    except ImportError:
        verify = True

    existing_zip = resolve_existing_hmdb_zip(
        explicit_zip=explicit_zip,
        search_downloads=search_downloads,
    )
    if existing_zip is not None:
        logger.info("Using existing HMDB zip: %s", existing_zip)
        return _copy_zip_if_needed(existing_zip, dest_zip)

    logger.info("Downloading HMDB metabolites from %s ...", HMDB_XML_URL)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/135.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        with requests.get(
            HMDB_XML_URL,
            stream=True,
            timeout=600,
            verify=verify,
            headers=headers,
            allow_redirects=True,
        ) as resp:
            body_prefix = next(resp.iter_content(chunk_size=16 * 1024), b"")
            if is_cloudflare_challenge(dict(resp.headers), body_prefix):
                raise HMDBDownloadBlocked(_manual_download_message())

            resp.raise_for_status()
            dest_zip.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_zip, "wb") as fh:
                if body_prefix:
                    fh.write(body_prefix)
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
    except Exception:
        if dest_zip.exists():
            dest_zip.unlink()
        raise

    logger.info("Downloaded: %s (%.1f MB)", dest_zip, dest_zip.stat().st_size / 1e6)
    return dest_zip


def extract_hmdb_xml(zip_path: Path, dest_path: Path) -> None:
    """Extract the HMDB XML from a zip into a gzipped XML file.

    Streams in 8 MB chunks to keep memory flat (HMDB XML is ~6 GB uncompressed).
    """
    import gzip
    import shutil
    import zipfile

    logger.info("Extracting HMDB XML (streaming)...")
    with zipfile.ZipFile(zip_path) as zf:
        xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
        if not xml_names:
            raise RuntimeError("No XML files found in HMDB zip")
        xml_name = xml_names[0]
        with zf.open(xml_name) as xml_fh, gzip.open(dest_path, "wb") as gz_fh:
            shutil.copyfileobj(xml_fh, gz_fh, length=8 * 1024 * 1024)
    logger.info("HMDB XML extracted (gzipped): %s", dest_path)


def download_hmdb_xml(
    dest_path: Path,
    explicit_zip: Path | None = None,
    search_downloads: bool = True,
) -> None:
    """Download HMDB full metabolite XML (zipped) or use an existing zip."""
    zip_path = download_hmdb_zip(
        HMDB_ZIP_PATH,
        explicit_zip=explicit_zip,
        search_downloads=search_downloads,
    )
    extract_hmdb_xml(zip_path, dest_path)


def main():
    parser = argparse.ArgumentParser(description="Download HMDB and build internal cache")
    parser.add_argument("--limit", type=int, default=None,
                        help="Parse only first N metabolites (for testing)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if XML already exists")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Use an existing HMDB zip instead of downloading",
    )
    parser.add_argument(
        "--no-search-downloads",
        action="store_true",
        help="Do not search ~/Downloads or HMDB_DOWNLOADS_DIR for a browser-downloaded zip",
    )
    args = parser.parse_args()

    from negbiodb_md.etl_hmdb import DEFAULT_HMDB_XML, DEFAULT_CACHE_DB, build_hmdb_cache

    xml_gz_ready = HMDB_XML_GZ.exists() and HMDB_XML_GZ.stat().st_size > 0
    if not xml_gz_ready and HMDB_XML_GZ.exists():
        logger.warning("Removing stale/empty %s before rebuild", HMDB_XML_GZ)
        HMDB_XML_GZ.unlink()

    if not xml_gz_ready and not args.skip_download:
        download_hmdb_xml(
            HMDB_XML_GZ,
            explicit_zip=args.zip_path,
            search_downloads=not args.no_search_downloads,
        )
    elif not xml_gz_ready:
        logger.error("HMDB XML not found and --skip-download specified: %s", HMDB_XML_GZ)
        sys.exit(1)

    logger.info("Building HMDB internal cache...")
    n = build_hmdb_cache(HMDB_XML_GZ, DEFAULT_CACHE_DB, limit=args.limit)
    logger.info("HMDB cache built: %d metabolites in %s", n, DEFAULT_CACHE_DB)


if __name__ == "__main__":
    main()
