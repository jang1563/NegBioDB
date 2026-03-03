"""Shared download utilities for NegBioDB data acquisition."""

import shutil
import urllib.request
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """Load NegBioDB configuration from YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_disk_space(path: str | Path, required_gb: float) -> None:
    """Raise ValueError if disk has less than required_gb free."""
    path = Path(path)
    # Treat path as directory unless it has a file-like suffix
    target = path.parent if path.suffix else path
    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / (1024**3)
    if free_gb < required_gb:
        raise ValueError(
            f"Insufficient disk space: {free_gb:.1f} GB free, "
            f"{required_gb:.1f} GB required at {target}"
        )


def verify_file_exists(path: str | Path, min_bytes: int = 0) -> bool:
    """Check that file exists and meets minimum size."""
    path = Path(path)
    if not path.exists():
        return False
    if min_bytes > 0 and path.stat().st_size < min_bytes:
        return False
    return True


class _TqdmReportHook:
    """Adapter to use tqdm with urllib.request.urlretrieve."""

    def __init__(self, desc: str):
        self.pbar = None
        self.desc = desc

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.pbar is None:
            self.pbar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                desc=self.desc,
            )
        downloaded = block_num * block_size
        if total_size > 0:
            self.pbar.n = min(downloaded, total_size)
            self.pbar.refresh()
        else:
            self.pbar.update(block_size)

    def close(self):
        if self.pbar:
            self.pbar.close()


def download_file_ftp(url: str, dest: str | Path, desc: str = "Downloading") -> Path:
    """Download a file via FTP with tqdm progress bar.

    Skips download if dest already exists and is non-empty.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already exists: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    hook = _TqdmReportHook(desc)
    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=hook)
    except Exception:
        # Remove partial file so next run retries
        if dest.exists():
            dest.unlink()
        raise
    finally:
        hook.close()

    print(f"Downloaded: {dest} ({dest.stat().st_size / (1024**2):.1f} MB)")
    return dest


def download_file_http(
    url: str, dest: str | Path, desc: str = "Downloading"
) -> Path:
    """Download a file via HTTP with tqdm progress bar.

    Skips download if dest already exists and is non-empty.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already exists: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, stream=True, allow_redirects=True, timeout=300)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        with (
            open(dest, "wb") as f,
            tqdm(total=total or None, unit="B", unit_scale=True, desc=desc) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    except Exception:
        # Remove partial file so next run retries
        if dest.exists():
            dest.unlink()
        raise

    print(f"Downloaded: {dest} ({dest.stat().st_size / (1024**2):.1f} MB)")
    return dest
