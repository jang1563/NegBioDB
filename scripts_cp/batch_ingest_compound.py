#!/usr/bin/env python3
"""Batch-ingest COMPOUND plates from JUMP Cell Painting (source_3).

Downloads plate profiles + backend CSVs from S3, builds metadata from
GitHub-sourced well.csv/compound.csv, then runs production ingest.

Usage:
    PYTHONPATH=src python scripts_cp/batch_ingest_compound.py --batches CP59 CP60
    PYTHONPATH=src python scripts_cp/batch_ingest_compound.py --batches CP59 --max-plates 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SCRATCH = Path(os.environ.get("SCRATCH", "/athena/masonlab/scratch/users/jak4013"))
S3_BASE = "https://cellpainting-gallery.s3.amazonaws.com"
SOURCE = "source_3"


def download_if_missing(url: str, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 100:
        log.info("  [cached] %s", out.name)
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("  [downloading] %s", url.split("/")[-1])
    result = subprocess.run(
        ["curl", "-sL", "--connect-timeout", "30", "--retry", "3", "--fail", "-o", str(out), url],
        capture_output=True,
    )
    if result.returncode != 0:
        log.warning("  Download failed: %s", url)
        out.unlink(missing_ok=True)
        return False
    return True


def build_metadata_for_batch(
    target: Path, meta_dir: Path, batch: str, plates: list[str],
    well_df: pd.DataFrame, compound_df: pd.DataFrame,
) -> None:
    md_root = target / "constructed_metadata" / SOURCE
    barcode_dir = md_root / "workspace" / "metadata" / "platemaps" / batch
    platemap_dir = barcode_dir / "platemap"
    platemap_dir.mkdir(parents=True, exist_ok=True)

    # Per-plate platemaps (build first to know which plates have data)
    plates_with_data = []
    for plate in plates:
        plate_wells = well_df[well_df["Metadata_Plate"] == plate]
        if plate_wells.empty:
            log.warning("  No wells for %s, skipping metadata", plate)
            continue
        plates_with_data.append(plate)
        pm = plate_wells[["Metadata_Well", "Metadata_JCP2022"]].copy()
        pm.columns = ["well_position", "broad_sample"]
        pm = pm.merge(
            compound_df[["Metadata_JCP2022", "Metadata_InChIKey", "Metadata_SMILES"]].rename(
                columns={"Metadata_JCP2022": "broad_sample", "Metadata_InChIKey": "InChIKey", "Metadata_SMILES": "SMILES"}
            ),
            on="broad_sample",
            how="left",
        )
        pm["control_type"] = "perturbation"
        pm.loc[pm["InChIKey"].isna(), "control_type"] = "dmso"
        pm.to_csv(platemap_dir / f"{plate}_platemap.txt", sep="\t", index=False)
        matched = int(pm["InChIKey"].notna().sum())
        dmso = int((pm["control_type"] == "dmso").sum())
        log.info("  %s: %d wells, %d compounds, %d DMSO", plate, len(pm), matched, dmso)

    # barcode_platemap (only plates with well data)
    rows = [{"Assay_Plate_Barcode": p, "Plate_Map_Name": f"{p}_platemap"} for p in plates_with_data]
    pd.DataFrame(rows).to_csv(barcode_dir / "barcode_platemap.csv", index=False)

    # External metadata
    ext = compound_df[["Metadata_JCP2022", "Metadata_InChIKey", "Metadata_SMILES"]].copy()
    ext.columns = ["broad_sample", "InChIKey", "SMILES"]
    ext.to_csv(barcode_dir / "external_metadata.tsv", sep="\t", index=False)
    log.info("Metadata built for %s: %d plates (%d with well data)", batch, len(plates), len(plates_with_data))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ingest COMPOUND plates")
    parser.add_argument("--batches", nargs="+", default=["CP59", "CP60"])
    parser.add_argument("--max-plates", type=int, default=None, help="Max plates per batch")
    parser.add_argument("--db-path", type=Path, default=SCRATCH / "negbiodb" / "data" / "negbiodb_cp.db")
    parser.add_argument("--fresh", action="store_true", help="Remove existing DB before ingest")
    args = parser.parse_args()

    target = SCRATCH / "jump_mirror"
    meta_dir = target / "jump_metadata"

    if not (meta_dir / "well.csv").exists():
        log.error("Missing %s — download GitHub metadata first", meta_dir / "well.csv")
        sys.exit(1)

    log.info("Loading GitHub metadata...")
    well_df = pd.read_csv(meta_dir / "well.csv")
    compound_df = pd.read_csv(meta_dir / "compound.csv")
    plate_df = pd.read_csv(meta_dir / "plate.csv")

    if args.fresh and args.db_path.exists():
        log.info("Removing existing DB: %s", args.db_path)
        args.db_path.unlink()

    project_root = Path(__file__).resolve().parent.parent

    for batch in args.batches:
        log.info("=" * 50)
        log.info("Processing batch: %s", batch)

        # Get COMPOUND plates for this batch
        batch_plates = plate_df[
            (plate_df["Metadata_Source"] == SOURCE)
            & (plate_df["Metadata_Batch"] == batch)
            & (plate_df["Metadata_PlateType"] == "COMPOUND")
        ]["Metadata_Plate"].tolist()

        if args.max_plates:
            batch_plates = batch_plates[: args.max_plates]

        log.info("Plates: %d", len(batch_plates))

        # Build metadata
        build_metadata_for_batch(target, meta_dir, batch, batch_plates, well_df, compound_df)

        # Download + prepare each plate
        work_dir = SCRATCH / "negbiodb_cp_production" / "work" / SOURCE / batch
        work_dir.mkdir(parents=True, exist_ok=True)
        prepared = 0

        for plate in batch_plates:
            obs_out = work_dir / f"{plate}_observations.parquet"
            prof_out = work_dir / f"{plate}_profile_features.parquet"
            meta_out = work_dir / f"{plate}_meta.json"

            if obs_out.exists():
                log.info("  [cached] %s", plate)
                prepared += 1
                continue

            log.info("--- Plate: %s ---", plate)

            # Download
            prof_path = target / "profiles" / f"{plate}.parquet"
            back_path = target / "backend" / f"{plate}.csv"

            prof_url = f"{S3_BASE}/cpg0016-jump/{SOURCE}/workspace/profiles/{batch}/{plate}/{plate}.parquet"
            back_url = f"{S3_BASE}/cpg0016-jump/{SOURCE}/workspace/backend/{batch}/{plate}/{plate}.csv"

            if not download_if_missing(prof_url, prof_path):
                continue
            if not download_if_missing(back_url, back_path):
                continue

            # Prepare
            md_root = target / "constructed_metadata" / SOURCE
            cmd = [
                sys.executable,
                str(project_root / "scripts_cp" / "prepare_jump_plate_production.py"),
                "--metadata-root", str(md_root),
                "--batch-name", batch,
                "--plate-name", plate,
                "--source-name", SOURCE,
                "--plate-profile-parquet", str(prof_path),
                "--backend-csv", str(back_path),
                "--output-observations", str(obs_out),
                "--output-profile-features", str(prof_out),
                "--output-meta", str(meta_out),
                "--cell-line-name", "U2OS",
                "--timepoint-h", "48.0",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
            if result.returncode != 0:
                log.warning("  prepare failed for %s: %s", plate, result.stderr[-200:] if result.stderr else "")
                continue
            prepared += 1

        log.info("Prepared %d/%d plates for %s", prepared, len(batch_plates), batch)

        # Concatenate
        import glob
        obs_files = sorted(glob.glob(str(work_dir / "*_observations.parquet")))
        prof_files = sorted(glob.glob(str(work_dir / "*_profile_features.parquet")))

        if not obs_files:
            log.warning("No observations for %s, skipping DB load", batch)
            continue

        obs = pd.concat([pd.read_parquet(f) for f in obs_files], ignore_index=True)
        prof = pd.concat([pd.read_parquet(f) for f in prof_files], ignore_index=True)
        concat_obs = work_dir / "batch_observations.parquet"
        concat_prof = work_dir / "batch_profiles.parquet"
        obs.to_parquet(concat_obs, index=False)
        prof.to_parquet(concat_prof, index=False)
        log.info("Batch %s: %d plates, %d observations, %d profiles", batch, len(obs_files), len(obs), len(prof))

        # Load into DB
        cmd = [
            sys.executable,
            str(project_root / "scripts_cp" / "load_jump_cp.py"),
            "--db-path", str(args.db_path),
            "--observations", str(concat_obs),
            "--profile-features", str(concat_prof),
            "--dataset-name", "cpg0016-jump",
            "--dataset-version", "1.0",
            "--annotation-mode", "annotated",
            "--source-url", f"{S3_BASE}/cpg0016-jump/{SOURCE}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
        if result.returncode != 0:
            log.error("DB load failed for %s: %s", batch, result.stderr[-500:] if result.stderr else "")
        else:
            log.info("Loaded %s into DB", batch)
            if result.stdout:
                log.info(result.stdout.strip())

    # Final stats
    import sqlite3
    if args.db_path.exists():
        db = sqlite3.connect(str(args.db_path))
        tiers = dict(db.execute("SELECT confidence_tier, COUNT(*) FROM cp_perturbation_results GROUP BY confidence_tier").fetchall())
        outcomes = dict(db.execute("SELECT outcome_label, COUNT(*) FROM cp_perturbation_results GROUP BY outcome_label").fetchall())
        total = db.execute("SELECT COUNT(*) FROM cp_perturbation_results").fetchone()[0]
        compounds = db.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        db.close()
        log.info("=" * 50)
        log.info("Final DB: %d results, %d compounds", total, compounds)
        log.info("Tiers: %s", tiers)
        log.info("Outcomes: %s", outcomes)


if __name__ == "__main__":
    main()
