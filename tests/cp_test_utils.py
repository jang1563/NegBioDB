"""Shared synthetic fixtures for CP-domain tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from negbiodb_cp.cp_db import get_connection, run_cp_migrations
from negbiodb_cp.etl_jump import ingest_jump_tables

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_cp"
CP_MIGRATIONS_DIR = MIGRATIONS_DIR


def build_cp_observations() -> pd.DataFrame:
    rows = []

    def add_obs(
        *,
        batch: str,
        plate: str,
        well: str,
        compound: str,
        smiles: str | None,
        dose: float,
        control_type: str,
        dmso_distance: float,
        repro: float,
        viability: float,
        qc_pass: int = 1,
        replicate_id: int | None = None,
    ) -> None:
        rows.append(
            {
                "batch_name": batch,
                "plate_name": plate,
                "cell_line_name": "U2OS",
                "well_id": well,
                "replicate_id": replicate_id,
                "dose": dose,
                "dose_unit": "uM",
                "timepoint_h": 48.0,
                "control_type": control_type,
                "dmso_distance": dmso_distance,
                "replicate_reproducibility": repro,
                "viability_ratio": viability,
                "qc_pass": qc_pass,
                "compound_name": compound,
                "canonical_smiles": smiles,
                "source_record_id": f"{batch}-{plate}-{well}-{compound}",
            }
        )

    dmso_smiles = "CS(C)=O"
    for idx, distance in enumerate([0.01, 0.02, 0.03, 0.04, 0.05], start=1):
        add_obs(
            batch="B1",
            plate="P1",
            well=f"A{idx:02d}",
            compound="DMSO",
            smiles=dmso_smiles,
            dose=0.0,
            control_type="dmso",
            dmso_distance=distance,
            repro=1.0,
            viability=1.0,
        )
    for idx, distance in enumerate([0.01, 0.02, 0.03, 0.04, 0.05], start=1):
        add_obs(
            batch="B2",
            plate="P2",
            well=f"A{idx:02d}",
            compound="DMSO",
            smiles=dmso_smiles,
            dose=0.0,
            control_type="dmso",
            dmso_distance=distance,
            repro=1.0,
            viability=1.0,
        )

    for rep, dist in enumerate([0.04, 0.05], start=1):
        add_obs(
            batch="B1",
            plate="P1",
            well=f"B{rep:02d}",
            compound="Aspirin",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            dose=1.0,
            control_type="perturbation",
            dmso_distance=dist,
            repro=0.90,
            viability=1.00,
            replicate_id=rep,
        )
    for rep, dist in enumerate([0.03, 0.04], start=1):
        add_obs(
            batch="B2",
            plate="P2",
            well=f"B{rep:02d}",
            compound="Aspirin",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            dose=1.0,
            control_type="perturbation",
            dmso_distance=dist,
            repro=0.88,
            viability=1.00,
            replicate_id=rep,
        )

    for rep, dist in enumerate([0.25, 0.27], start=1):
        add_obs(
            batch="B1",
            plate="P1",
            well=f"C{rep:02d}",
            compound="Ibuprofen",
            smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            dose=2.0,
            control_type="perturbation",
            dmso_distance=dist,
            repro=0.85,
            viability=0.92,
            replicate_id=rep,
        )

    add_obs(
        batch="B1",
        plate="P1",
        well="D01",
        compound="Caffeine",
        smiles="Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        dose=3.0,
        control_type="perturbation",
        dmso_distance=0.12,
        repro=0.40,
        viability=0.96,
        replicate_id=1,
    )

    for rep, dist in enumerate([0.18, 0.21], start=1):
        add_obs(
            batch="B1",
            plate="P1",
            well=f"E{rep:02d}",
            compound="Toxicol",
            smiles="CCN(CC)CCOC(=O)C1=CC=CC=C1Cl",
            dose=4.0,
            control_type="perturbation",
            dmso_distance=dist,
            repro=0.87,
            viability=0.30,
            replicate_id=rep,
        )

    add_obs(
        batch="B1",
        plate="P1",
        well="F01",
        compound="Mystery",
        smiles=None,
        dose=1.5,
        control_type="perturbation",
        dmso_distance=0.07,
        repro=0.55,
        viability=0.98,
        replicate_id=1,
    )

    add_obs(
        batch="B1",
        plate="P1",
        well="G01",
        compound="QCFail",
        smiles="CCO",
        dose=5.0,
        control_type="perturbation",
        dmso_distance=0.20,
        repro=0.80,
        viability=0.95,
        qc_pass=0,
        replicate_id=1,
    )

    return pd.DataFrame(rows)


def build_cp_feature_df(kind: str) -> pd.DataFrame:
    base = [
        ("Aspirin", "B1", 1.0, 0.10, 0.20),
        ("Aspirin", "B2", 1.0, 0.11, 0.22),
        ("Ibuprofen", "B1", 2.0, 0.80, 0.70),
        ("Caffeine", "B1", 3.0, 0.35, 0.40),
        ("Toxicol", "B1", 4.0, 0.95, 0.15),
        ("Mystery", "B1", 1.5, 0.30, 0.45),
    ]
    rows = []
    for compound, batch, dose, f1, f2 in base:
        rows.append(
            {
                "compound_name": compound,
                "batch_name": batch,
                "cell_line_name": "U2OS",
                "dose": dose,
                "dose_unit": "uM",
                "timepoint_h": 48.0,
                "feature_source": f"{kind}_synthetic",
                "storage_uri": f"s3://synthetic/{kind}/{compound}_{batch}.parquet",
                f"{kind}_f1": f1,
                f"{kind}_f2": f2,
            }
        )
    return pd.DataFrame(rows)


def build_cp_orthogonal_evidence_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "compound_name": "Aspirin",
                "batch_name": "B1",
                "cell_line_name": "U2OS",
                "dose": 1.0,
                "dose_unit": "uM",
                "timepoint_h": 48.0,
                "evidence_domain": "dti",
                "evidence_label": "inactive",
                "source_name": "synthetic_dti",
                "source_record_id": "DTI-001",
                "match_key": "aspirin-u2os-1.0",
            }
        ]
    )


def create_seeded_cp_database(tmp_path: Path, annotation_mode: str = "annotated") -> Path:
    db_path = tmp_path / "test_cp.db"
    run_cp_migrations(db_path, MIGRATIONS_DIR)
    conn = get_connection(db_path)
    try:
        ingest_jump_tables(
            conn,
            build_cp_observations(),
            profile_features=build_cp_feature_df("profile"),
            image_features=build_cp_feature_df("image"),
            orthogonal_evidence=build_cp_orthogonal_evidence_df(),
            annotation_mode=annotation_mode,
        )
    finally:
        conn.close()
    return db_path


def create_synthetic_jump_metadata_root(tmp_path: Path) -> Path:
    root = tmp_path / "jump_mirror"
    batch = "2021_04_26_Batch1"
    plate = "BR00117035"

    platemap_dir = root / "workspace" / "metadata" / "platemaps" / batch / "platemap"
    external_dir = root / "workspace" / "metadata" / "external_metadata"
    structure_dir = root / "workspace" / "structure"
    platemap_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)

    barcode = pd.DataFrame(
        [
            {
                "Metadata_Plate": plate,
                "Metadata_Source": "source_4",
                "Plate_Map_Name": "plate_map_A",
            }
        ]
    )
    barcode.to_csv(platemap_dir.parent / "barcode_platemap.csv", index=False)

    platemap = pd.DataFrame(
        [
            {"well_position": "A01", "broad_sample": "DMSO", "pert_type": "negcon", "mmoles_per_liter": 0.0},
            {"well_position": "A02", "broad_sample": "DMSO", "pert_type": "negcon", "mmoles_per_liter": 0.0},
            {"well_position": "B01", "broad_sample": "BRD-A", "pert_type": "trt", "mmoles_per_liter": 0.001},
            {"well_position": "B02", "broad_sample": "BRD-A", "pert_type": "trt", "mmoles_per_liter": 0.001},
            {"well_position": "C01", "broad_sample": "BRD-B", "pert_type": "trt", "mmoles_per_liter": 0.002},
            {"well_position": "C02", "broad_sample": "BRD-B", "pert_type": "trt", "mmoles_per_liter": 0.002},
        ]
    )
    platemap.to_csv(platemap_dir / "plate_map_A.csv", index=False)

    external = pd.DataFrame(
        [
            {"broad_sample": "DMSO", "compound_name": "DMSO", "canonical_smiles": "CS(C)=O", "inchikey": "DMSOINCHI"},
            {"broad_sample": "BRD-A", "compound_name": "Aspirin", "canonical_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "inchikey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"},
            {"broad_sample": "BRD-B", "compound_name": "Ibuprofen", "canonical_smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "inchikey": "HEFNNWSXXWATRW-UHFFFAOYSA-N"},
        ]
    )
    external.to_csv(external_dir / "external_metadata.tsv", sep="\t", index=False)

    structure = {
        "paths": [
            f"workspace/metadata/platemaps/{batch}/barcode_platemap.csv",
            f"workspace/metadata/platemaps/{batch}/platemap/plate_map_A.csv",
            "workspace/metadata/external_metadata/external_metadata.tsv",
        ]
    }
    (structure_dir / "structure.json").write_text(json.dumps(structure, indent=2))
    return root


def create_synthetic_jump_metadata_tables_root(
    tmp_path: Path,
    *,
    plate_type: str = "COMPOUND",
) -> Path:
    root = tmp_path / "jump_tables"
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    plate = pd.DataFrame(
        [
            {
                "Metadata_Source": "source_1",
                "Metadata_Batch": "Batch1_20221004",
                "Metadata_Plate": "UL001641",
                "Metadata_PlateType": plate_type,
            }
        ]
    )
    well = pd.DataFrame(
        [
            {"Metadata_Source": "source_1", "Metadata_Plate": "UL001641", "Metadata_Well": "A01", "Metadata_JCP2022": "JCP2022_033924"},
            {"Metadata_Source": "source_1", "Metadata_Plate": "UL001641", "Metadata_Well": "A02", "Metadata_JCP2022": "JCP2022_033924"},
            {"Metadata_Source": "source_1", "Metadata_Plate": "UL001641", "Metadata_Well": "B01", "Metadata_JCP2022": "JCP2022_000001"},
            {"Metadata_Source": "source_1", "Metadata_Plate": "UL001641", "Metadata_Well": "B02", "Metadata_JCP2022": "JCP2022_000001"},
            {"Metadata_Source": "source_1", "Metadata_Plate": "UL001641", "Metadata_Well": "C01", "Metadata_JCP2022": "JCP2022_000002"},
            {"Metadata_Source": "source_1", "Metadata_Plate": "UL001641", "Metadata_Well": "C02", "Metadata_JCP2022": "JCP2022_000002"},
        ]
    )
    compound = pd.DataFrame(
        [
            {"Metadata_JCP2022": "JCP2022_000001", "Metadata_InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "Metadata_InChI": "InChI=1S/C9H8O4", "Metadata_SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"},
            {"Metadata_JCP2022": "JCP2022_000002", "Metadata_InChIKey": "HEFNNWSXXWATRW-UHFFFAOYSA-N", "Metadata_InChI": "InChI=1S/C13H18O2", "Metadata_SMILES": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
        ]
    )
    compound_source = pd.DataFrame(
        [
            {"Metadata_Compound_Source": "source_1", "Metadata_JCP2022": "JCP2022_000001"},
            {"Metadata_Compound_Source": "source_1", "Metadata_JCP2022": "JCP2022_000002"},
        ]
    )
    perturbation_control = pd.DataFrame(
        [
            {"Metadata_JCP2022": "JCP2022_033924", "Metadata_pert_type": "negcon", "Metadata_Name": "DMSO", "Metadata_modality": "compound"},
            {"Metadata_JCP2022": "JCP2022_000001", "Metadata_pert_type": "trt", "Metadata_Name": "AspirinLike", "Metadata_modality": "compound"},
            {"Metadata_JCP2022": "JCP2022_000002", "Metadata_pert_type": "trt", "Metadata_Name": "IbuprofenLike", "Metadata_modality": "compound"},
        ]
    )

    plate.to_csv(metadata_dir / "plate.csv.gz", index=False, compression="gzip")
    well.to_csv(metadata_dir / "well.csv.gz", index=False, compression="gzip")
    compound.to_csv(metadata_dir / "compound.csv.gz", index=False, compression="gzip")
    compound_source.to_csv(metadata_dir / "compound_source.csv.gz", index=False, compression="gzip")
    perturbation_control.to_csv(metadata_dir / "perturbation_control.csv", index=False)
    return root


def create_synthetic_plate_profile_and_backend(
    tmp_path: Path,
    *,
    plate: str = "BR00117035",
    source: str = "source_4",
    backend_plate_as_int: bool = False,
) -> tuple[Path, Path]:
    wells = ["A01", "A02", "B01", "B02", "C01", "C02"]
    profile = pd.DataFrame(
        {
            "Metadata_Plate": [plate] * len(wells),
            "Metadata_Well": wells,
            "Metadata_Source": [source] * len(wells),
            "Cells_AreaShape_Area": [100.0, 101.0, 110.0, 111.0, 150.0, 151.0],
            "Cytoplasm_AreaShape_Area": [80.0, 80.5, 90.0, 89.5, 120.0, 119.0],
        }
    )
    backend = pd.DataFrame(
        {
            "Metadata_Plate": [int(plate) if backend_plate_as_int else plate] * len(wells),
            "Metadata_Well": wells,
            "Metadata_Count_Cells": [100, 98, 96, 97, 80, 82],
            "Metadata_Count_Nuclei": [100, 98, 96, 97, 80, 82],
        }
    )
    profile_path = tmp_path / "plate.parquet"
    backend_path = tmp_path / "backend.csv"
    profile.to_parquet(profile_path, index=False)
    backend.to_csv(backend_path, index=False)
    return profile_path, backend_path


make_cp_observations = build_cp_observations
make_cp_profile_features = lambda: build_cp_feature_df("profile")
make_cp_image_features = lambda: build_cp_feature_df("image")
make_cp_orthogonal_evidence = build_cp_orthogonal_evidence_df
create_seeded_cp_db = create_seeded_cp_database
