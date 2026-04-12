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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
HMDB_XML_URL = "https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip"
HMDB_XML_GZ = _PROJECT_ROOT / "data" / "hmdb_metabolites.xml.gz"
HMDB_ZIP_PATH = _PROJECT_ROOT / "data" / "hmdb_metabolites.zip"


def download_hmdb_xml(dest_path: Path) -> None:
    """Download HMDB full metabolite XML (zipped)."""
    import zipfile
    import requests

    try:
        import certifi
        verify = certifi.where()
    except ImportError:
        verify = True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest_path.parent / "hmdb_metabolites.zip"

    if zip_path.exists():
        logger.info("HMDB zip already downloaded: %s", zip_path)
    else:
        logger.info("Downloading HMDB metabolites from %s ...", HMDB_XML_URL)
        try:
            with requests.get(
                HMDB_XML_URL,
                stream=True,
                timeout=600,
                verify=verify,
            ) as resp:
                resp.raise_for_status()
                with open(zip_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
        except Exception:
            if zip_path.exists():
                zip_path.unlink()
            raise
        logger.info("Downloaded: %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)

    # Extract XML
    logger.info("Extracting HMDB XML...")
    with zipfile.ZipFile(zip_path) as zf:
        xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
        if not xml_names:
            raise RuntimeError("No XML files found in HMDB zip")
        xml_name = xml_names[0]
        import gzip
        with zf.open(xml_name) as xml_fh, gzip.open(dest_path, "wb") as gz_fh:
            gz_fh.write(xml_fh.read())
    logger.info("HMDB XML extracted (gzipped): %s", dest_path)


def main():
    parser = argparse.ArgumentParser(description="Download HMDB and build internal cache")
    parser.add_argument("--limit", type=int, default=None,
                        help="Parse only first N metabolites (for testing)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download if XML already exists")
    args = parser.parse_args()

    from negbiodb_md.etl_hmdb import DEFAULT_HMDB_XML, DEFAULT_CACHE_DB, build_hmdb_cache

    if not HMDB_XML_GZ.exists() and not args.skip_download:
        download_hmdb_xml(HMDB_XML_GZ)
    elif not HMDB_XML_GZ.exists():
        logger.error("HMDB XML not found and --skip-download specified: %s", HMDB_XML_GZ)
        sys.exit(1)

    logger.info("Building HMDB internal cache...")
    n = build_hmdb_cache(HMDB_XML_GZ, DEFAULT_CACHE_DB, limit=args.limit)
    logger.info("HMDB cache built: %d metabolites in %s", n, DEFAULT_CACHE_DB)


if __name__ == "__main__":
    main()
