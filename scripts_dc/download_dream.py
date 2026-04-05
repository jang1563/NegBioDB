#!/usr/bin/env python3
"""Download AstraZeneca-Sanger DREAM Challenge data for DC domain ETL.

Downloads from Synapse (requires synapseclient + authentication):
  - DREAM challenge combination synergy scores
  - 11,576 experiments, 910 combinations, 85 cell lines
  - Loewe and Bliss scores

Source: https://www.synapse.org/Synapse:syn4231880
License: CC BY 4.0

Prerequisite:
    pip install synapseclient
    synapse login --rememberMe  (or set SYNAPSE_AUTH_TOKEN env var)

Usage:
    python scripts_dc/download_dream.py [--output-dir data/dc/dream] [--force]
"""

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Synapse entity IDs for DREAM challenge files
DREAM_FILES = {
    # Combination synergy scores (main file)
    "ch1_train_combination_and_monoTherapy.csv": "syn4231880",
    # Leaderboard data
    "ch2_leaderboard_monoTherapy.csv": "syn5614767",
}


def download_from_synapse(
    syn_id: str, output_dir: Path, force: bool = False
) -> bool:
    """Download a file from Synapse. Returns True on success."""
    import synapseclient

    # Check auth token
    auth_token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not auth_token:
        print(
            "NOTE: SYNAPSE_AUTH_TOKEN not set. "
            "Attempting cached login (synapse login --rememberMe)."
        )

    syn = synapseclient.Synapse()
    try:
        syn.login(silent=True)
    except Exception as e:
        print(
            f"ERROR: Synapse login failed: {e}\n"
            "Set SYNAPSE_AUTH_TOKEN or run: synapse login --rememberMe",
            file=sys.stderr,
        )
        return False

    print(f"  Downloading {syn_id} ...")
    try:
        collision = "overwrite.local" if force else "keep.local"
        entity = syn.get(syn_id, downloadLocation=str(output_dir), ifcollision=collision)
        size_mb = Path(entity.path).stat().st_size / 1e6
        print(f"    Done: {Path(entity.path).name} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ERROR downloading {syn_id}: {e}", file=sys.stderr)
        return False


def main():
    # Early check for synapseclient
    try:
        import synapseclient  # noqa: F401
    except ImportError:
        print(
            "ERROR: synapseclient not installed. Install with:\n"
            "  pip install synapseclient",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(
        description="Download AZ-Sanger DREAM Challenge data from Synapse"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "data" / "dc" / "dream",
        help="Output directory (default: data/dc/dream)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    print(f"DREAM Challenge download → {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for filename, syn_id in DREAM_FILES.items():
        output_path = args.output_dir / filename
        if output_path.exists() and not args.force:
            size_mb = output_path.stat().st_size / 1e6
            print(f"  Already exists ({size_mb:.1f} MB), skipping: {filename}")
            continue
        if not download_from_synapse(syn_id, args.output_dir, args.force):
            success = False

    if success:
        print("\nAll DREAM Challenge files downloaded.")
        return 0
    else:
        print("\nSome downloads failed. See errors above.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
