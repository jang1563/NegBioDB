#!/usr/bin/env python3
"""Prepare one annotation-backed JUMP plate for production CP ingest."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from negbiodb_cp.jump_metadata import load_plate_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_plate_profile(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in ("Metadata_Plate", "Metadata_Well", "Metadata_Source"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _load_backend_counts(path: Path) -> pd.DataFrame:
    usecols = [
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site_Count",
        "Metadata_Count_Cells",
        "Metadata_Count_CellsIncludingEdges",
        "Metadata_Count_Cytoplasm",
        "Metadata_Count_Nuclei",
        "Metadata_Count_NucleiIncludingEdges",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols)
    for col in ("Metadata_Plate", "Metadata_Well"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def _select_profile_features(df: pd.DataFrame, max_features: int) -> list[str]:
    excluded_cols = {"dose", "timepoint_h", "pubchem_cid"}
    feature_cols = [
        col for col in df.columns
        if is_numeric_dtype(df[col])
        and not str(col).startswith("Metadata_")
        and str(col) not in excluded_cols
    ]
    feature_cols = [col for col in feature_cols if not str(col).startswith("Metadata_Count_")]
    feature_cols = sorted(feature_cols)
    return feature_cols[:max_features] if max_features > 0 else feature_cols


def _compute_reproducibility(features: pd.DataFrame, groups: pd.Series) -> pd.Series:
    values = features.fillna(0.0).to_numpy(dtype=float)
    repro = pd.Series(np.nan, index=features.index, dtype=float)
    for _, idx in groups.groupby(groups).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        mat = values[idx]
        corr = np.corrcoef(mat)
        if np.ndim(corr) == 0:
            repro.iloc[idx] = 1.0
            continue
        for row_i, source_idx in enumerate(idx):
            row = np.delete(corr[row_i], row_i)
            finite = row[np.isfinite(row)]
            repro.iloc[source_idx] = float(finite.mean()) if finite.size else np.nan
    return repro


def build_production_tables(
    annotations: pd.DataFrame,
    plate_profile: pd.DataFrame,
    backend_counts: pd.DataFrame,
    *,
    batch_name: str,
    cell_line_name: str,
    timepoint_h: float,
    max_profile_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    annotations = annotations.copy()
    plate_profile = plate_profile.copy()
    backend_counts = backend_counts.copy()

    annotations["well_id"] = annotations["well_id"].astype(str)
    for col in ("Metadata_Plate", "Metadata_Well"):
        if col in plate_profile.columns:
            plate_profile[col] = plate_profile[col].astype(str)
        if col in backend_counts.columns:
            backend_counts[col] = backend_counts[col].astype(str)

    merged = plate_profile.merge(
        annotations,
        left_on="Metadata_Well",
        right_on="well_id",
        how="left",
    ).merge(
        backend_counts.drop_duplicates(subset=["Metadata_Plate", "Metadata_Well"]),
        on=["Metadata_Plate", "Metadata_Well"],
        how="left",
    )

    annotation_coverage = int(merged["compound_name"].notna().sum())
    if annotation_coverage == 0:
        raise ValueError("Annotation coverage is zero; refusing to fall back to proxy mode")

    feature_cols = _select_profile_features(merged, max_profile_features)
    if not feature_cols:
        raise ValueError("No numeric profile features were found in the plate parquet.")

    profile_matrix = merged[feature_cols].fillna(0.0).to_numpy(dtype=float)
    control_mask = merged["control_type"].fillna("").astype(str).str.lower().eq("dmso").to_numpy()
    if not control_mask.any():
        raise ValueError("No control wells were recovered from annotation-backed metadata")

    dmso_centroid = profile_matrix[control_mask].mean(axis=0)
    dmso_distance = np.linalg.norm(profile_matrix - dmso_centroid, axis=1)

    dmso_counts = merged.loc[control_mask, "Metadata_Count_Cells"].dropna()
    baseline_count = float(dmso_counts.median()) if not dmso_counts.empty else np.nan
    viability_ratio = (
        merged["Metadata_Count_Cells"].astype(float) / baseline_count
        if np.isfinite(baseline_count) and baseline_count > 0
        else pd.Series(np.nan, index=merged.index, dtype=float)
    )

    reproducibility_group = (
        merged["annotation_key"].fillna(merged["compound_name"]).astype(str)
        + "::"
        + merged["dose"].fillna(-1).astype(str)
    )
    reproducibility = _compute_reproducibility(merged[feature_cols], reproducibility_group)

    non_perturbation_mask = merged["control_type"].fillna("").astype(str).ne("perturbation")

    observations = pd.DataFrame(
        {
            "batch_name": batch_name,
            "plate_name": merged["Metadata_Plate"],
            "cell_line_name": cell_line_name,
            "well_id": merged["Metadata_Well"],
            "site_id": None,
            "replicate_id": None,
            "dose": np.where(non_perturbation_mask, 0.0, merged["dose"]),
            "dose_unit": merged["dose_unit"].fillna("uM"),
            "timepoint_h": timepoint_h,
            "control_type": merged["control_type"].fillna("perturbation"),
            "dmso_distance": dmso_distance,
            "replicate_reproducibility": reproducibility,
            "viability_ratio": viability_ratio,
            "qc_pass": (merged["Metadata_Count_Cells"].fillna(0).astype(float) > 0).astype(int),
            "compound_name": merged["compound_name"].fillna("UNKNOWN_COMPOUND"),
            "canonical_smiles": merged["canonical_smiles"],
            "inchikey": merged["inchikey"],
            "pubchem_cid": merged["pubchem_cid"],
            "chembl_id": merged["chembl_id"],
            "source_record_id": (
                batch_name
                + "::"
                + merged["Metadata_Plate"].astype(str)
                + "::"
                + merged["Metadata_Well"].astype(str)
            ),
        }
    )

    consensus_profile = (
        pd.concat(
            [
                observations[
                    [
                        "compound_name",
                        "batch_name",
                        "cell_line_name",
                        "dose",
                        "dose_unit",
                        "timepoint_h",
                    ]
                ],
                merged[feature_cols].reset_index(drop=True),
            ],
            axis=1,
        )
        .groupby(
            ["compound_name", "batch_name", "cell_line_name", "dose", "dose_unit", "timepoint_h"],
            dropna=False,
        )[feature_cols]
        .mean()
        .reset_index()
    )
    consensus_profile["feature_source"] = "jump_plate_profile_annotated"
    consensus_profile["storage_uri"] = str(Path(batch_name) / str(merged["Metadata_Plate"].iloc[0]))

    meta = {
        "annotation_mode": "annotated",
        "annotation_coverage": annotation_coverage,
        "n_wells": int(len(observations)),
        "n_qc_pass": int(observations["qc_pass"].sum()),
        "n_control_wells": int(control_mask.sum()),
        "feature_count": len(feature_cols),
    }
    return observations, consensus_profile, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare one annotation-backed JUMP plate for CP ingest.")
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--plate-name", required=True)
    parser.add_argument("--plate-profile-parquet", type=Path, required=True)
    parser.add_argument("--backend-csv", type=Path, required=True)
    parser.add_argument("--output-observations", type=Path, required=True)
    parser.add_argument("--output-profile-features", type=Path, required=True)
    parser.add_argument("--output-meta", type=Path, required=True)
    parser.add_argument("--structure-json", type=Path, default=None)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--cell-line-name", default="U2OS")
    parser.add_argument("--timepoint-h", type=float, default=48.0)
    parser.add_argument("--default-compound-dose", type=float, default=10.0)
    parser.add_argument("--dose-unit", default="uM")
    parser.add_argument("--max-profile-features", type=int, default=256)
    args = parser.parse_args(argv)

    annotations, annotation_meta = load_plate_annotations(
        args.metadata_root,
        batch_name=args.batch_name,
        plate_name=args.plate_name,
        source_name=args.source_name,
        structure_json=args.structure_json,
        default_compound_dose_um=args.default_compound_dose,
        default_dose_unit=args.dose_unit,
    )
    plate = _load_plate_profile(args.plate_profile_parquet)
    backend = _load_backend_counts(args.backend_csv)
    observations, profile_features, meta = build_production_tables(
        annotations,
        plate,
        backend,
        batch_name=args.batch_name,
        cell_line_name=args.cell_line_name,
        timepoint_h=args.timepoint_h,
        max_profile_features=args.max_profile_features,
    )
    meta.update(annotation_meta)

    args.output_observations.parent.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(args.output_observations, index=False)
    profile_features.to_parquet(args.output_profile_features, index=False)
    with open(args.output_meta, "w") as handle:
        json.dump(meta, handle, indent=2)

    logger.info("Wrote %d observation rows -> %s", len(observations), args.output_observations)
    logger.info("Wrote %d consensus profile rows -> %s", len(profile_features), args.output_profile_features)
    logger.info("Production metadata: %s", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
