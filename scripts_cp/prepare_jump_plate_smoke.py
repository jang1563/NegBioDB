#!/usr/bin/env python3
"""Prepare a JUMP plate into CP observation/profile tables for proxy smoke ingest.

This script is intentionally narrow and smoke-oriented:
- input 1: assembled COMPOUND parquet with Metadata_JCP2022
- input 2: one per-plate profile parquet
- input 3: one per-plate backend CSV with cell-count QC fields

It joins by (Metadata_Source, Metadata_Plate, Metadata_Well), infers the
modal repeated JCP2022 code as the DMSO-like control, and falls back to
`plate_proxy` controls when real annotation coverage is missing. This helper
is only for plumbing smoke validation, not benchmark-grade production ingest.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_assembled_metadata(path: Path) -> pd.DataFrame:
    cols = ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]
    return pd.read_parquet(path, columns=cols)


def _load_plate_profile(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


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
    return pd.read_csv(path, usecols=lambda c: c in usecols)


def _infer_control_jcp(jcp_series: pd.Series) -> str | None:
    counts = jcp_series.dropna().astype(str).value_counts()
    if counts.empty:
        return None
    for jcp in counts.index:
        upper = jcp.upper()
        if "DMSO" in upper or "VEHICLE" in upper:
            return jcp
    if counts.iloc[0] >= 8:
        return str(counts.index[0])
    return None


def _select_profile_features(df: pd.DataFrame, max_features: int) -> list[str]:
    excluded_prefixes = ("Metadata_",)
    feature_cols = [
        col for col in df.columns
        if is_numeric_dtype(df[col])
        and not any(col.startswith(prefix) for prefix in excluded_prefixes)
    ]
    feature_cols = [
        col for col in feature_cols
        if not col.startswith("Metadata_Count_")
    ]
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
        if corr.ndim == 0:
            repro.iloc[idx] = 1.0
            continue
        for row_i, source_idx in enumerate(idx):
            row = np.delete(corr[row_i], row_i)
            finite = row[np.isfinite(row)]
            repro.iloc[source_idx] = float(finite.mean()) if finite.size else np.nan
    return repro


def _infer_proxy_controls(feature_matrix: np.ndarray, min_controls: int = 16) -> np.ndarray:
    """Pick centroid-nearest wells as DMSO-like proxies for smoke-only fallback."""
    if feature_matrix.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    centroid = feature_matrix.mean(axis=0)
    distances = np.linalg.norm(feature_matrix - centroid, axis=1)
    n_controls = min(feature_matrix.shape[0] - 1, max(min_controls, feature_matrix.shape[0] // 20))
    if n_controls <= 0:
        return np.zeros(feature_matrix.shape[0], dtype=bool)
    mask = np.zeros(feature_matrix.shape[0], dtype=bool)
    mask[np.argsort(distances)[:n_controls]] = True
    return mask


def build_smoke_tables(
    assembled_metadata: pd.DataFrame,
    plate_profile: pd.DataFrame,
    backend_counts: pd.DataFrame,
    *,
    batch_name: str,
    cell_line_name: str,
    default_dose: float,
    dose_unit: str,
    timepoint_h: float,
    max_profile_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    merged = plate_profile.merge(
        assembled_metadata,
        on=["Metadata_Source", "Metadata_Plate", "Metadata_Well"],
        how="left",
    ).merge(
        backend_counts.drop_duplicates(subset=["Metadata_Plate", "Metadata_Well"]),
        on=["Metadata_Plate", "Metadata_Well"],
        how="left",
    )

    feature_cols = _select_profile_features(merged, max_profile_features)
    if not feature_cols:
        raise ValueError("No numeric profile features were found in the plate parquet.")

    profile_matrix = merged[feature_cols].fillna(0.0).to_numpy(dtype=float)
    annotation_matches = int(merged["Metadata_JCP2022"].notna().sum())
    annotation_mode = "jcp2022" if annotation_matches > 0 else "plate_proxy"

    if annotation_matches > 0:
        merged["compound_name"] = merged["Metadata_JCP2022"].fillna("UNKNOWN_JCP2022")
        control_jcp = _infer_control_jcp(merged["compound_name"])
        merged["control_type"] = "perturbation"
        if control_jcp is not None:
            merged.loc[merged["compound_name"] == control_jcp, "control_type"] = "dmso"
        control_mask = (merged["control_type"] == "dmso").to_numpy()
    else:
        merged["compound_name"] = (
            merged["Metadata_Plate"].astype(str)
            + "::"
            + merged["Metadata_Well"].astype(str)
        )
        control_jcp = None
        control_mask = _infer_proxy_controls(profile_matrix)
        merged["control_type"] = np.where(control_mask, "dmso", "perturbation")

    if control_mask.any():
        dmso_centroid = profile_matrix[control_mask].mean(axis=0)
        dmso_distance = np.linalg.norm(profile_matrix - dmso_centroid, axis=1)
    else:
        dmso_distance = np.full(len(merged), np.nan)

    dmso_counts = merged.loc[control_mask, "Metadata_Count_Cells"].dropna()
    baseline_count = float(dmso_counts.median()) if not dmso_counts.empty else np.nan
    viability_ratio = (
        merged["Metadata_Count_Cells"].astype(float) / baseline_count
        if np.isfinite(baseline_count) and baseline_count > 0
        else pd.Series(np.nan, index=merged.index, dtype=float)
    )

    reproducibility = _compute_reproducibility(merged[feature_cols], merged["compound_name"])

    observations = pd.DataFrame(
        {
            "batch_name": batch_name,
            "plate_name": merged["Metadata_Plate"],
            "cell_line_name": cell_line_name,
            "well_id": merged["Metadata_Well"],
            "site_id": None,
            "replicate_id": None,
            "dose": np.where(control_mask, 0.0, default_dose),
            "dose_unit": dose_unit,
            "timepoint_h": timepoint_h,
            "control_type": merged["control_type"],
            "dmso_distance": dmso_distance,
            "replicate_reproducibility": reproducibility,
            "viability_ratio": viability_ratio,
            "qc_pass": (
                merged["Metadata_Count_Cells"].fillna(0).astype(float) > 0
            ).astype(int),
            "compound_name": merged["compound_name"],
            "canonical_smiles": None,
            "inchikey": None,
            "pubchem_cid": None,
            "chembl_id": None,
            "source_record_id": (
                merged["Metadata_Source"].astype(str)
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
                observations[[
                    "compound_name",
                    "batch_name",
                    "cell_line_name",
                    "dose",
                    "dose_unit",
                    "timepoint_h",
                ]],
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
    consensus_profile["feature_source"] = "jump_plate_profile_smoke"
    consensus_profile["storage_uri"] = str(Path(batch_name) / str(merged["Metadata_Plate"].iloc[0]))

    meta = {
        "n_wells": int(len(observations)),
        "n_qc_pass": int(observations["qc_pass"].sum()),
        "n_control_wells": int(control_mask.sum()),
        "control_jcp2022": control_jcp,
        "annotation_matches": annotation_matches,
        "annotation_mode": annotation_mode,
        "feature_count": len(feature_cols),
    }
    return observations, consensus_profile, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a real JUMP plate for CP smoke ingest.")
    parser.add_argument("--assembled-compound-parquet", type=Path, required=True)
    parser.add_argument("--plate-profile-parquet", type=Path, required=True)
    parser.add_argument("--backend-csv", type=Path, required=True)
    parser.add_argument("--output-observations", type=Path, required=True)
    parser.add_argument("--output-profile-features", type=Path, required=True)
    parser.add_argument("--output-meta", type=Path, required=True)
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--cell-line-name", default="U2OS")
    parser.add_argument("--default-dose", type=float, default=1.0)
    parser.add_argument("--dose-unit", default="uM")
    parser.add_argument("--timepoint-h", type=float, default=48.0)
    parser.add_argument("--max-profile-features", type=int, default=256)
    args = parser.parse_args(argv)

    assembled = _load_assembled_metadata(args.assembled_compound_parquet)
    plate = _load_plate_profile(args.plate_profile_parquet)
    backend = _load_backend_counts(args.backend_csv)

    observations, profile_features, meta = build_smoke_tables(
        assembled,
        plate,
        backend,
        batch_name=args.batch_name,
        cell_line_name=args.cell_line_name,
        default_dose=args.default_dose,
        dose_unit=args.dose_unit,
        timepoint_h=args.timepoint_h,
        max_profile_features=args.max_profile_features,
    )

    args.output_observations.parent.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(args.output_observations, index=False)
    profile_features.to_parquet(args.output_profile_features, index=False)
    with open(args.output_meta, "w") as handle:
        json.dump(meta, handle, indent=2)

    logger.info("Wrote %d observation rows -> %s", len(observations), args.output_observations)
    logger.info("Wrote %d consensus profile rows -> %s", len(profile_features), args.output_profile_features)
    logger.info("Smoke metadata: %s", meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
