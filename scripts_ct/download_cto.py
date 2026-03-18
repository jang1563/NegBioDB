"""Download CTO binary outcome labels from HuggingFace.

CTO (Clinical Trial Outcome) provides 125K+ trial outcome labels
derived via weak supervision (GPT-3.5 + news sentiment + stock prices).

Source: Gao et al. 2026 (Nature Health)
License: MIT

Usage:
    python scripts_ct/download_cto.py
"""

from pathlib import Path

from negbiodb.download import download_file_http, load_config, verify_file_exists


def main():
    cfg = load_config()
    dl = cfg["ct_domain"]["downloads"]["cto"]
    dest = Path(dl["dest"])

    print("=== CTO Outcome Labels Download ===")
    print(f"Dest: {dest}")

    download_file_http(dl["url"], dest, desc="CTO outcome labels")

    if not verify_file_exists(dest, min_bytes=10000):
        raise ValueError(f"CTO file too small or missing: {dest}")

    size_mb = dest.stat().st_size / (1024**2)
    print(f"Downloaded: {dest} ({size_mb:.1f} MB)")
    print("\nCTO download complete.")


if __name__ == "__main__":
    main()
