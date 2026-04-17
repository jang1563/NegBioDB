"""Helpers for resolving JUMP Cell Painting metadata from an HPC mirror."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

NORMALIZED_METADATA_FILES = {
    "plate": "plate.csv.gz",
    "well": "well.csv.gz",
    "compound": "compound.csv.gz",
    "compound_source": "compound_source.csv.gz",
    "perturbation_control": "perturbation_control.csv",
}
CHEMICAL_PLATE_TYPES = {"compound", "compound_empty"}


def _collect_string_paths(obj) -> list[str]:
    paths: list[str] = []
    if isinstance(obj, str):
        paths.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            paths.extend(_collect_string_paths(value))
    elif isinstance(obj, list):
        for value in obj:
            paths.extend(_collect_string_paths(value))
    return paths


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".txt", ".tsv"}:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            prefix = handle.read(256)
        if prefix.lstrip().startswith("<?xml") and "<Error>" in prefix:
            raise ValueError(f"Metadata file is an XML error page, not a data table: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported metadata table format: {path}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        col: (
            str(col)
            .strip()
            .lower()
            .replace("metadata_", "")
            .replace(" ", "_")
            .replace("-", "_")
        )
        for col in df.columns
    }
    return df.rename(columns=renamed).copy()


def _first_existing(root: Path, candidates: list[str | Path]) -> Path | None:
    for candidate in candidates:
        if isinstance(candidate, Path):
            path = candidate if candidate.is_absolute() else root / candidate
        else:
            path = root / candidate
        if path.exists():
            return path
    return None


def _find_files(root: Path, name_fragment: str) -> list[Path]:
    needle = name_fragment.lower()
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and needle in path.name.lower()
    )


def _normalized_metadata_path(metadata_root: Path, filename: str) -> Path | None:
    return _first_existing(
        metadata_root,
        [
            Path("metadata") / filename,
            Path("github_metadata") / filename,
            filename,
        ],
    )


def _load_normalized_metadata_tables(metadata_root: Path) -> dict[str, pd.DataFrame] | None:
    paths: dict[str, Path] = {}
    for key, filename in NORMALIZED_METADATA_FILES.items():
        path = _normalized_metadata_path(metadata_root, filename)
        if path is None:
            return None
        paths[key] = path

    return {key: _normalize_columns(_read_table(path)) for key, path in paths.items()}


def _resolve_control_type(pert_type: str | None, metadata_name: str | None) -> str:
    value = pert_type.strip().lower() if isinstance(pert_type, str) else ""
    label = metadata_name.strip().lower() if isinstance(metadata_name, str) else ""
    if value in {"negcon", "negative_control"} or label in {"dmso", "vehicle"}:
        return "dmso"
    if value in {"empty", "poscon", "positive_control"}:
        return "other"
    return "perturbation"


def _load_annotations_from_normalized_tables(
    metadata_root: Path,
    *,
    batch_name: str,
    plate_name: str,
    source_name: str | None,
    default_compound_dose_um: float,
    default_dose_unit: str,
) -> tuple[pd.DataFrame, dict]:
    tables = _load_normalized_metadata_tables(metadata_root)
    if tables is None:
        raise FileNotFoundError("Normalized GitHub metadata tables are not present")

    plate = tables["plate"]
    plate_mask = plate["plate"].astype(str).eq(plate_name)
    if source_name and "source" in plate.columns:
        plate_mask &= plate["source"].astype(str).eq(source_name)
    if "batch" in plate.columns:
        plate_mask &= plate["batch"].astype(str).eq(batch_name)
    plate_rows = plate.loc[plate_mask].copy()
    if plate_rows.empty:
        raise ValueError(f"Could not find plate {plate_name} in normalized metadata tables")

    plate_row = plate_rows.iloc[0]
    plate_type = str(plate_row.get("platetype", "")).strip()
    if plate_type.lower() not in CHEMICAL_PLATE_TYPES:
        raise ValueError(
            f"Plate {plate_name} is not a chemical perturbation plate: plate_type={plate_type}"
        )

    well = tables["well"]
    well_mask = well["plate"].astype(str).eq(plate_name)
    if source_name and "source" in well.columns:
        well_mask &= well["source"].astype(str).eq(source_name)
    well_rows = well.loc[well_mask].copy()
    if well_rows.empty:
        raise ValueError(f"No wells found for plate {plate_name} in normalized metadata tables")

    compound = tables["compound"]
    perturbation_control = tables["perturbation_control"]
    if "compound_source" in tables["compound_source"].columns and source_name:
        allowed = set(
            tables["compound_source"]
            .loc[
                tables["compound_source"]["compound_source"].astype(str).eq(source_name),
                "jcp2022",
            ]
            .astype(str)
        )
        if allowed:
            well_rows = well_rows[
                well_rows["jcp2022"].astype(str).isin(allowed)
                | well_rows["jcp2022"].astype(str).isin(
                    perturbation_control["jcp2022"].astype(str)
                )
            ].copy()

    merged = (
        well_rows.merge(compound, on="jcp2022", how="left", suffixes=("", "_compound"))
        .merge(perturbation_control, on="jcp2022", how="left", suffixes=("", "_control"))
    )
    if merged.empty:
        raise ValueError(f"Normalized metadata join returned no annotations for plate {plate_name}")

    rows = []
    for row in merged.to_dict(orient="records"):
        record = pd.Series(row)
        control_type = _resolve_control_type(record.get("pert_type"), record.get("name"))
        compound_name = None
        if control_type != "perturbation":
            metadata_name = record.get("name")
            if isinstance(metadata_name, str) and metadata_name.strip():
                compound_name = metadata_name.strip()
        if not compound_name:
            compound_name = str(record.get("jcp2022", "")).strip() or None
        dose = 0.0 if control_type != "perturbation" else float(default_compound_dose_um)
        rows.append(
            {
                "well_id": str(record["well"]),
                "compound_name": compound_name,
                "annotation_key": str(record.get("jcp2022", "")).strip() or None,
                "canonical_smiles": record.get("smiles"),
                "inchikey": record.get("inchikey"),
                "pubchem_cid": None,
                "chembl_id": None,
                "dose": dose,
                "dose_unit": default_dose_unit,
                "control_type": control_type,
            }
        )

    annotations = pd.DataFrame(rows)
    meta = {
        "metadata_backend": "github_tables",
        "plate_type": plate_type,
        "annotation_coverage": int(annotations["compound_name"].notna().sum()),
        "normalized_metadata_files": {
            key: str(_normalized_metadata_path(metadata_root, filename))
            for key, filename in NORMALIZED_METADATA_FILES.items()
        },
    }
    return annotations, meta


def load_structure_paths(metadata_root: Path, structure_json: Path | None = None) -> list[str]:
    """Load all string paths mentioned in a JUMP structure manifest, if present."""
    candidate = structure_json
    if candidate is None:
        candidate = _first_existing(
            metadata_root,
            [
                "structure.json",
                "workspace/structure/structure.json",
                "workspace/metadata/structure.json",
            ],
        )
    if candidate is None or not candidate.exists():
        return []
    data = json.loads(candidate.read_text())
    return _collect_string_paths(data)


def _match_plate_row(barcode_df: pd.DataFrame, plate_name: str, source_name: str | None) -> pd.Series:
    df = _normalize_columns(barcode_df)
    if source_name and "source" in df.columns:
        source_mask = df["source"].astype(str).str.contains(source_name, case=False, na=False)
        if source_mask.any():
            df = df[source_mask].copy()

    plate_cols = [col for col in df.columns if "barcode" in col or col == "plate" or col.endswith("_plate")]
    if not plate_cols:
        raise ValueError("barcode_platemap is missing a plate/barcode column")

    # Prioritize: exact match > endswith > contains
    for plate_col in plate_cols:
        values = df[plate_col].astype(str)
        exact = values.eq(plate_name)
        if exact.any():
            return df.loc[exact].iloc[0]
    for plate_col in plate_cols:
        values = df[plate_col].astype(str)
        suffix = values.str.endswith(plate_name)
        if suffix.any():
            return df.loc[suffix].iloc[0]
    for plate_col in plate_cols:
        values = df[plate_col].astype(str)
        partial = values.str.contains(plate_name, regex=False)
        if partial.any():
            return df.loc[partial].iloc[0]
    raise ValueError(f"Could not find plate {plate_name} in barcode_platemap")


def _resolve_platemap_path(metadata_root: Path, paths: list[str], batch_name: str, plate_name: str, plate_row: pd.Series) -> Path:
    platemap_name = None
    for col in plate_row.index:
        if "plate_map" in col or "platemap" in col:
            value = plate_row.get(col)
            if isinstance(value, str) and value.strip():
                platemap_name = value.strip()
                break
    search_roots = []
    for value in paths:
        if "platemap" in value.lower():
            search_roots.append(metadata_root / value)
    search_roots.extend(
        [
            metadata_root / "workspace" / "metadata" / "platemaps" / batch_name,
            metadata_root / "workspace" / "platemaps" / batch_name,
            metadata_root / "metadata" / "platemaps" / batch_name,
            metadata_root / batch_name,
        ]
    )

    candidates: list[Path] = []
    if platemap_name:
        stem = Path(platemap_name).stem
        for root in search_roots:
            if root.is_file() and root.stem == stem:
                candidates.append(root)
            if root.is_dir():
                for suffix in (".csv", ".tsv", ".txt"):
                    candidates.extend(sorted(root.rglob(f"{stem}{suffix}")))
    if not candidates:
        for root in search_roots:
            if root.is_dir():
                for suffix in (".csv", ".tsv", ".txt"):
                    candidates.extend(sorted(root.rglob(f"*{plate_name}*{suffix}")))
    if not candidates:
        raise ValueError(f"Could not resolve platemap file for plate {plate_name}")
    return candidates[0]


def _load_external_metadata(metadata_root: Path, paths: list[str]) -> pd.DataFrame | None:
    candidates: list[Path] = []
    for value in paths:
        if "external_metadata" in value.lower():
            path = metadata_root / value
            if path.exists():
                candidates.append(path)
    if not candidates:
        candidates.extend(_find_files(metadata_root, "external_metadata"))
    for candidate in candidates:
        try:
            return _normalize_columns(_read_table(candidate))
        except Exception:
            continue
    return None


def _infer_control_type(row: pd.Series) -> str:
    for col in row.index:
        if "control" in col or "pert_type" in col or "sample_type" in col or "role" in col:
            value = str(row.get(col, "")).strip().lower()
            if value in {"dmso", "vehicle", "negcon", "negative_control", "control"}:
                return "dmso"
            if value in {"other", "empty", "background"}:
                return "other"
    for col in row.index:
        value = row.get(col)
        if isinstance(value, str) and value.strip().lower() in {"dmso", "vehicle"}:
            return "dmso"
    return "perturbation"


def _infer_dose(row: pd.Series) -> tuple[float | None, str]:
    for col in row.index:
        value = row.get(col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        col_lower = str(col).lower()
        if "mmoles_per_liter" in col_lower:
            return float(value) * 1000.0, "uM"
        if "micromolar" in col_lower or col_lower.endswith("_um"):
            return float(value), "uM"
        if col_lower in {"dose", "concentration", "perturbation_dose"}:
            return float(value), "uM"
    return None, "uM"


def _merge_external_annotations(platemap: pd.DataFrame, external_metadata: pd.DataFrame | None) -> pd.DataFrame:
    if external_metadata is None or external_metadata.empty:
        return _normalize_columns(platemap)

    left = _normalize_columns(platemap)
    right = _normalize_columns(external_metadata)
    candidate_pairs = [
        ("jcp2022", "jcp2022"),
        ("broad_sample", "broad_sample"),
        ("perturbation", "perturbation"),
        ("compound_name", "compound_name"),
    ]
    for left_key, right_key in candidate_pairs:
        if left_key in left.columns and right_key in right.columns:
            merged = left.merge(
                right,
                left_on=left_key,
                right_on=right_key,
                how="left",
                suffixes=("", "_ext"),
            )
            if merged.filter(regex="_ext$").notna().any().any():
                return merged
    return left


def load_plate_annotations(
    metadata_root: Path,
    *,
    batch_name: str,
    plate_name: str,
    source_name: str | None = None,
    structure_json: Path | None = None,
    default_compound_dose_um: float = 10.0,
    default_dose_unit: str = "uM",
) -> tuple[pd.DataFrame, dict]:
    """Load one plate's well-level annotation from mirrored JUMP metadata."""
    metadata_root = Path(metadata_root)
    try:
        return _load_annotations_from_normalized_tables(
            metadata_root,
            batch_name=batch_name,
            plate_name=plate_name,
            source_name=source_name,
            default_compound_dose_um=default_compound_dose_um,
            default_dose_unit=default_dose_unit,
        )
    except (FileNotFoundError, KeyError):
        pass

    structure_paths = load_structure_paths(metadata_root, structure_json=structure_json)

    barcode_candidates = [path for path in structure_paths if "barcode_platemap" in path.lower()]
    barcode_path = _first_existing(metadata_root, barcode_candidates)
    if barcode_path is None:
        barcode_path = _first_existing(
            metadata_root,
            [
                f"workspace/metadata/platemaps/{batch_name}/barcode_platemap.csv",
                f"workspace/platemaps/{batch_name}/barcode_platemap.csv",
                f"metadata/platemaps/{batch_name}/barcode_platemap.csv",
            ],
        )
    if barcode_path is None:
        matches = _find_files(metadata_root, "barcode_platemap")
        if matches:
            barcode_path = matches[0]
    if barcode_path is None:
        raise ValueError("Could not locate barcode_platemap metadata")

    barcode_df = _read_table(barcode_path)
    plate_row = _match_plate_row(barcode_df, plate_name, source_name)
    platemap_path = _resolve_platemap_path(metadata_root, structure_paths, batch_name, plate_name, plate_row)
    platemap = _normalize_columns(_read_table(platemap_path))
    external_metadata = _load_external_metadata(metadata_root, structure_paths)
    merged = _merge_external_annotations(platemap, external_metadata)

    well_col = next((col for col in merged.columns if col in {"well", "well_position", "well_id"}), None)
    if well_col is None:
        raise ValueError("Platemap file is missing a well column")

    compound_candidates = ["compound_name", "pert_iname", "jcp2022", "broad_sample", "perturbation"]
    compound_col = next((col for col in compound_candidates if col in merged.columns), None)
    if compound_col is None:
        raise ValueError("Platemap/external metadata do not provide a compound identifier column")

    rows = []
    for row in merged.to_dict(orient="records"):
        record = pd.Series(row)
        dose, dose_unit = _infer_dose(record)
        control_type = _infer_control_type(record)
        annotation_key = None
        for col in ("jcp2022", "broad_sample", "compound_name", "perturbation"):
            value = record.get(col)
            if isinstance(value, str) and value.strip():
                annotation_key = value.strip()
                break
        rows.append(
            {
                "well_id": str(record[well_col]),
                "compound_name": str(record.get(compound_col, "")).strip() or annotation_key,
                "annotation_key": annotation_key,
                "canonical_smiles": record.get("canonical_smiles") or record.get("smiles"),
                "inchikey": record.get("inchikey"),
                "pubchem_cid": record.get("pubchem_cid"),
                "chembl_id": record.get("chembl_id"),
                "dose": dose,
                "dose_unit": dose_unit,
                "control_type": control_type,
            }
        )

    annotations = pd.DataFrame(rows)
    meta = {
        "metadata_backend": "legacy_platemap",
        "barcode_platemap": str(barcode_path),
        "platemap": str(platemap_path),
        "external_metadata_found": external_metadata is not None,
        "annotation_coverage": int(annotations["compound_name"].notna().sum()),
    }
    return annotations, meta
