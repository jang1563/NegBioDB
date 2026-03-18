"""Download Shi & Du 2024 structured p-value datasets from GitHub.

Two files: efficacy_results.csv (~119K rows), safety_results.csv (~803K rows).
Contains structured p-values, effect sizes, and safety outcomes from clinical trials.

Source: Shi & Du 2024 (Scientific Data)
License: CC0 (public domain)

Usage:
    python scripts_ct/download_shi_du.py
"""

from pathlib import Path

from negbiodb.download import download_file_http, load_config, verify_file_exists


def main():
    cfg = load_config()
    dl = cfg["ct_domain"]["downloads"]["shi_du"]

    print("=== Shi & Du 2024 Download ===")

    for key, url_key, dest_key, desc in [
        ("efficacy", "efficacy_url", "efficacy_dest", "Efficacy results"),
        ("safety", "safety_url", "safety_dest", "Safety results"),
    ]:
        url = dl[url_key]
        dest = Path(dl[dest_key])
        print(f"\n--- {desc} ---")
        print(f"Dest: {dest}")

        download_file_http(url, dest, desc=desc)

        if not verify_file_exists(dest, min_bytes=1000):
            raise ValueError(f"Shi & Du {key} file too small or missing: {dest}")

        size_mb = dest.stat().st_size / (1024**2)
        print(f"Downloaded: {dest} ({size_mb:.1f} MB)")

    print("\nShi & Du download complete.")


if __name__ == "__main__":
    main()
