"""Tests for annotation-backed CP production ingest helpers."""

import json

import pandas as pd
import pytest

import scripts_cp.prepare_jump_plate_production as prepare_jump_plate_production
from negbiodb_cp.jump_metadata import load_plate_annotations
from tests.cp_test_utils import (
    create_synthetic_jump_metadata_root,
    create_synthetic_jump_metadata_tables_root,
    create_synthetic_plate_profile_and_backend,
)


def test_prepare_jump_plate_production_builds_annotated_outputs(tmp_path):
    metadata_root = create_synthetic_jump_metadata_root(tmp_path)
    profile_path, backend_path = create_synthetic_plate_profile_and_backend(
        tmp_path,
        plate="UL001641",
        source="source_1",
    )
    observations_path = tmp_path / "obs.parquet"
    features_path = tmp_path / "features.parquet"
    meta_path = tmp_path / "meta.json"

    assert prepare_jump_plate_production.main(
        [
            "--metadata-root", str(metadata_root),
            "--batch-name", "2021_04_26_Batch1",
            "--plate-name", "BR00117035",
            "--plate-profile-parquet", str(profile_path),
            "--backend-csv", str(backend_path),
            "--output-observations", str(observations_path),
            "--output-profile-features", str(features_path),
            "--output-meta", str(meta_path),
            "--source-name", "source_4",
        ]
    ) == 0

    observations = pd.read_parquet(observations_path)
    features = pd.read_parquet(features_path)
    meta = json.loads(meta_path.read_text())

    assert not observations.empty
    assert not features.empty
    assert meta["annotation_mode"] == "annotated"
    assert meta["annotation_coverage"] > 0
    assert meta["n_control_wells"] > 0


def test_prepare_jump_plate_production_prefers_normalized_metadata_tables(tmp_path):
    metadata_root = create_synthetic_jump_metadata_tables_root(tmp_path)
    profile_path, backend_path = create_synthetic_plate_profile_and_backend(tmp_path)
    observations_path = tmp_path / "obs.parquet"
    features_path = tmp_path / "features.parquet"
    meta_path = tmp_path / "meta.json"

    assert prepare_jump_plate_production.main(
        [
            "--metadata-root", str(metadata_root),
            "--batch-name", "Batch1_20221004",
            "--plate-name", "UL001641",
            "--plate-profile-parquet", str(profile_path),
            "--backend-csv", str(backend_path),
            "--output-observations", str(observations_path),
            "--output-profile-features", str(features_path),
            "--output-meta", str(meta_path),
            "--source-name", "source_1",
        ]
    ) == 0

    observations = pd.read_parquet(observations_path)
    meta = json.loads(meta_path.read_text())
    assert not observations.empty
    assert set(observations["compound_name"]) >= {"DMSO", "JCP2022_000001", "JCP2022_000002"}
    assert float(observations.loc[observations["control_type"] == "perturbation", "dose"].iloc[0]) == 10.0
    assert meta["metadata_backend"] == "github_tables"
    assert meta["plate_type"] == "COMPOUND"


def test_prepare_jump_plate_production_handles_mixed_plate_key_types(tmp_path):
    metadata_root = create_synthetic_jump_metadata_tables_root(tmp_path)
    profile_path, backend_path = create_synthetic_plate_profile_and_backend(
        tmp_path,
        plate="1053601756",
        source="source_1",
        backend_plate_as_int=True,
    )
    observations_path = tmp_path / "obs_mixed.parquet"
    features_path = tmp_path / "features_mixed.parquet"
    meta_path = tmp_path / "meta_mixed.json"

    # Rebuild metadata root for the numeric-looking plate id used in this test.
    metadata_dir = metadata_root / "metadata"
    plate_df = pd.read_csv(metadata_dir / "plate.csv.gz")
    plate_df["Metadata_Plate"] = "1053601756"
    plate_df.to_csv(metadata_dir / "plate.csv.gz", index=False, compression="gzip")
    well_df = pd.read_csv(metadata_dir / "well.csv.gz")
    well_df["Metadata_Plate"] = "1053601756"
    well_df.to_csv(metadata_dir / "well.csv.gz", index=False, compression="gzip")

    assert prepare_jump_plate_production.main(
        [
            "--metadata-root", str(metadata_root),
            "--batch-name", "Batch1_20221004",
            "--plate-name", "1053601756",
            "--plate-profile-parquet", str(profile_path),
            "--backend-csv", str(backend_path),
            "--output-observations", str(observations_path),
            "--output-profile-features", str(features_path),
            "--output-meta", str(meta_path),
            "--source-name", "source_1",
        ]
    ) == 0

    meta = json.loads(meta_path.read_text())
    assert meta["annotation_mode"] == "annotated"
    assert meta["annotation_coverage"] > 0


def test_load_plate_annotations_rejects_non_compound_plate_types(tmp_path):
    metadata_root = create_synthetic_jump_metadata_tables_root(tmp_path, plate_type="ORF")

    with pytest.raises(ValueError, match="not a chemical perturbation plate"):
        load_plate_annotations(
            metadata_root,
            batch_name="Batch1_20221004",
            plate_name="UL001641",
            source_name="source_1",
        )
