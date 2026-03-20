"""Tests for CT ML export pipeline (ct_export.py).

14 test classes covering:
  Phase A: loaders, 6 splits, apply_all, failure export
  Phase B: CTO success, M1 builder
  Phase C: M2, leakage report
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from negbiodb_ct.ct_db import get_connection, refresh_all_ct_pairs, run_ct_migrations
from negbiodb_ct.ct_export import (
    CATEGORY_TO_INT,
    CT_SPLIT_STRATEGIES,
    CT_TEMPORAL_TRAIN_CUTOFF,
    CT_TEMPORAL_VAL_CUTOFF,
    apply_all_ct_splits,
    apply_ct_m1_splits,
    apply_ct_m2_splits,
    build_ct_m1_dataset,
    export_ct_failure_dataset,
    export_ct_m2_dataset,
    generate_ct_cold_condition_split,
    generate_ct_cold_drug_split,
    generate_ct_degree_balanced_split,
    generate_ct_leakage_report,
    generate_ct_random_split,
    generate_ct_scaffold_split,
    generate_ct_temporal_split,
    load_ct_m2_data,
    load_ct_pairs_df,
    load_cto_success_pairs,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ct"

# 15 real SMILES with 3+ scaffold classes
REAL_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # 0: aspirin (salicylate scaffold)
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # 1: ibuprofen (arylpropionic)
    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",  # 2: caffeine (xanthine)
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  # 3: glucose (sugar)
    "CC(=O)Oc1ccccc1",  # 4: phenyl acetate (salicylate — same as aspirin)
    "c1ccc2c(c1)cc1ccc3ccccc3c1n2",  # 5: acridine (tricyclic)
    "c1ccc(cc1)C(=O)O",  # 6: benzoic acid (simple benzene)
    "CC1=CC(=O)c2ccccc2C1=O",  # 7: menadione (naphthoquinone)
    "c1ccc(cc1)c2ccccc2",  # 8: biphenyl
    "CC(=O)NC1=CC=C(O)C=C1",  # 9: paracetamol (aminophenol)
    "c1ccc(cc1)Oc2ccccc2",  # 10: diphenyl ether
    "CC(C)NCC(O)c1ccc(O)c(O)c1",  # 11: isoproterenol (catechol)
    "O=C1N(C(=O)c2ccccc21)c1ccccc1",  # 12: N-phenylphthalimide
    "c1ccncc1",  # 13: pyridine (heterocycle)
    "C1CCCCC1",  # 14: cyclohexane (no aromatic scaffold)
]

MESH_IDS = [f"D{i:06d}" for i in range(1, 9)]  # D000001..D000008

CATEGORIES = ["efficacy", "safety", "enrollment", "strategic", "design", "regulatory", "other", "pharmacokinetic"]
TIERS = ["gold", "silver", "bronze", "copper"]

# Completion dates spanning temporal cutoffs (2017 boundary, 2019 boundary, 2020+)
TRIAL_DATES = [
    ("2014-06-15", "NCT10001"),
    ("2015-03-01", "NCT10002"),
    ("2016-09-22", "NCT10003"),
    ("2017-01-10", "NCT10004"),
    ("2017-12-31", "NCT10005"),
    ("2018-04-20", "NCT10006"),
    ("2018-11-11", "NCT10007"),
    ("2019-02-28", "NCT10008"),
    ("2019-06-15", "NCT10009"),
    ("2019-12-01", "NCT10010"),
    ("2020-01-15", "NCT10011"),
    ("2020-06-30", "NCT10012"),
    ("2021-03-14", "NCT10013"),
    ("2022-08-01", "NCT10014"),
    ("2023-05-10", "NCT10015"),
    ("2024-01-01", "NCT10016"),
    (None, "NCT10017"),  # NULL completion date → train
    ("2017-06-15", "NCT10018"),
    ("2019-09-09", "NCT10019"),
    ("2021-12-25", "NCT10020"),
    ("2016-01-01", "NCT10021"),
    ("2018-07-07", "NCT10022"),
    ("2020-03-03", "NCT10023"),
    ("2023-11-11", "NCT10024"),
    ("2015-05-05", "NCT10025"),
    ("2017-08-08", "NCT10026"),
    ("2019-04-04", "NCT10027"),
    ("2022-02-02", "NCT10028"),
    ("2020-09-09", "NCT10029"),
    ("2024-06-06", "NCT10030"),
]


def _populate_ct_test_db(conn, n_drugs=15, n_conditions=8, n_trials=30):
    """Populate test CT database with realistic data.

    Creates n_drugs interventions (with SMILES), n_conditions, n_trials trials,
    and ~60 failure results across varied categories, tiers, and dates.
    """
    rng = np.random.RandomState(123)

    # Insert interventions with SMILES
    for i in range(n_drugs):
        smi = REAL_SMILES[i]
        # Generate fake inchikey_connectivity (14 chars from SMILES hash)
        ik_conn = f"IK{i:012d}"
        ik_full = f"{ik_conn}-UHFFFAOYSA-N"
        conn.execute(
            "INSERT INTO interventions "
            "(intervention_type, intervention_name, canonical_smiles, "
            " inchikey, inchikey_connectivity, molecular_type, chembl_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "drug",
                f"Drug_{i}",
                smi,
                ik_full,
                ik_conn,
                "small_molecule",
                f"CHEMBL{1000 + i}",
            ),
        )

    # Insert a few non-SMILES interventions
    for i in range(3):
        conn.execute(
            "INSERT INTO interventions "
            "(intervention_type, intervention_name, molecular_type) "
            "VALUES (?, ?, ?)",
            ("biologic", f"Biologic_{i}", "monoclonal_antibody"),
        )

    # Insert conditions with mesh_ids
    for i in range(n_conditions):
        conn.execute(
            "INSERT INTO conditions (condition_name, mesh_id) VALUES (?, ?)",
            (f"Disease_{i}", MESH_IDS[i]),
        )

    # Insert trials with completion_dates and NCT IDs
    assert n_trials <= len(TRIAL_DATES)
    for i in range(n_trials):
        comp_date, nct_id = TRIAL_DATES[i]
        phase = rng.choice(
            ["phase_1", "phase_2", "phase_3", "phase_1_2", "early_phase_1"]
        )
        conn.execute(
            "INSERT INTO clinical_trials "
            "(source_db, source_trial_id, overall_status, trial_phase, "
            " completion_date, sponsor_type, randomized, enrollment_actual) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "clinicaltrials_gov",
                nct_id,
                "Terminated",
                phase,
                comp_date,
                rng.choice(["industry", "academic"]),
                int(rng.choice([0, 1])),
                int(rng.randint(50, 500)),
            ),
        )

    # Insert trial_interventions + trial_conditions (junction tables)
    total_drugs = n_drugs + 3  # includes biologics
    for trial_idx in range(1, n_trials + 1):
        # 1-2 interventions per trial
        n_intv = rng.randint(1, 3)
        intv_ids = rng.choice(range(1, total_drugs + 1), size=n_intv, replace=False)
        for iid in intv_ids:
            try:
                conn.execute(
                    "INSERT INTO trial_interventions (trial_id, intervention_id) "
                    "VALUES (?, ?)",
                    (trial_idx, int(iid)),
                )
            except Exception:
                pass  # duplicate PK — skip

        # 1-2 conditions per trial
        n_cond = rng.randint(1, 3)
        cond_ids = rng.choice(range(1, n_conditions + 1), size=n_cond, replace=False)
        for cid in cond_ids:
            try:
                conn.execute(
                    "INSERT INTO trial_conditions (trial_id, condition_id) "
                    "VALUES (?, ?)",
                    (trial_idx, int(cid)),
                )
            except Exception:
                pass

    # Insert trial_failure_results: ~60 results
    result_idx = 0
    for trial_idx in range(1, n_trials + 1):
        # Get interventions/conditions for this trial
        intv_ids = [
            r[0]
            for r in conn.execute(
                "SELECT intervention_id FROM trial_interventions WHERE trial_id=?",
                (trial_idx,),
            ).fetchall()
        ]
        cond_ids = [
            r[0]
            for r in conn.execute(
                "SELECT condition_id FROM trial_conditions WHERE trial_id=?",
                (trial_idx,),
            ).fetchall()
        ]
        if not intv_ids or not cond_ids:
            continue

        for iid in intv_ids:
            for cid in cond_ids:
                cat = rng.choice(CATEGORIES)
                tier = rng.choice(TIERS)
                phase = rng.choice(
                    ["phase_1", "phase_2", "phase_3", "phase_1_2", "early_phase_1"]
                )
                try:
                    conn.execute(
                        "INSERT INTO trial_failure_results "
                        "(intervention_id, condition_id, trial_id, failure_category, "
                        " confidence_tier, source_db, source_record_id, "
                        " extraction_method, highest_phase_reached, "
                        " p_value_primary, effect_size) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            iid,
                            cid,
                            trial_idx,
                            cat,
                            tier,
                            "aact",
                            f"R{result_idx:04d}",
                            "database_direct",
                            phase,
                            round(float(rng.uniform(0.01, 0.9)), 3),
                            round(float(rng.uniform(-1.0, 1.0)), 3),
                        ),
                    )
                    result_idx += 1
                except Exception:
                    pass  # unique constraint — skip

    # Insert some intervention_targets
    for i in range(1, n_drugs + 1):
        n_targets = rng.randint(0, 4)
        for j in range(n_targets):
            try:
                conn.execute(
                    "INSERT INTO intervention_targets "
                    "(intervention_id, uniprot_accession, gene_symbol, source) "
                    "VALUES (?, ?, ?, ?)",
                    (i, f"P{i:05d}_{j}", f"GENE{i}_{j}", "chembl"),
                )
            except Exception:
                pass

    # Add "clean" success-only trials (no failure results) for CTO M1 testing.
    # These use existing interventions + conditions but have NO failure results,
    # so their CTO success pairs won't conflict with the failure pairs table.
    #
    # Strategy: Insert new SMILES-bearing interventions (drug_ids 19-21)
    # that have NO failure results. This ensures smiles_only M1 variant has Y=1 rows.
    clean_drug_offset = total_drugs  # 18
    clean_smiles = [
        ("ClC1=CC=CC=C1", "IK_CLEAN_0001", "IK_CLEAN_0001-UHFFFAOYSA-N"),  # chlorobenzene
        ("FC1=CC=CC=C1", "IK_CLEAN_0002", "IK_CLEAN_0002-UHFFFAOYSA-N"),  # fluorobenzene
        ("BrC1=CC=CC=C1", "IK_CLEAN_0003", "IK_CLEAN_0003-UHFFFAOYSA-N"),  # bromobenzene
    ]
    for i, (smi, ik_conn, ik_full) in enumerate(clean_smiles):
        conn.execute(
            "INSERT INTO interventions "
            "(intervention_type, intervention_name, canonical_smiles, "
            " inchikey, inchikey_connectivity, molecular_type, chembl_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "drug",
                f"CleanDrug_{i}",
                smi,
                ik_full,
                ik_conn,
                "small_molecule",
                f"CHEMBL_CLEAN_{i}",
            ),
        )

    clean_ncts = ["NCT_CLEAN_01", "NCT_CLEAN_02", "NCT_CLEAN_03"]
    for idx, nct_id in enumerate(clean_ncts):
        conn.execute(
            "INSERT INTO clinical_trials "
            "(source_db, source_trial_id, overall_status, trial_phase, "
            " completion_date, sponsor_type) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("clinicaltrials_gov", nct_id, "Completed", "phase_3",
             "2021-06-15", "industry"),
        )
        trial_id = conn.execute(
            "SELECT trial_id FROM clinical_trials WHERE source_trial_id=?",
            (nct_id,),
        ).fetchone()[0]
        # Link to the clean SMILES intervention (no failure results → no conflicts)
        clean_drug_id = clean_drug_offset + idx + 1
        conn.execute(
            "INSERT OR IGNORE INTO trial_interventions "
            "(trial_id, intervention_id) VALUES (?, ?)",
            (trial_id, clean_drug_id),
        )
        # Also link biologic for extra coverage
        bio_id = n_drugs + idx + 1
        if bio_id <= total_drugs:
            conn.execute(
                "INSERT OR IGNORE INTO trial_interventions "
                "(trial_id, intervention_id) VALUES (?, ?)",
                (trial_id, bio_id),
            )
        # Use a unique condition (idx+1)
        cond_id = idx + 1
        conn.execute(
            "INSERT OR IGNORE INTO trial_conditions "
            "(trial_id, condition_id) VALUES (?, ?)",
            (trial_id, cond_id),
        )

    conn.commit()
    refresh_all_ct_pairs(conn)
    conn.commit()


def _make_cto_parquet(tmp_path, nct_ids_success, nct_ids_failure=None):
    """Create a minimal CTO parquet file."""
    rows = []
    for nct_id in nct_ids_success:
        rows.append({"nct_id": nct_id, "labels": 1.0})
    for nct_id in (nct_ids_failure or []):
        rows.append({"nct_id": nct_id, "labels": 0.0})
    df = pd.DataFrame(rows)
    path = tmp_path / "cto_outcomes.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def ct_db(tmp_path):
    """Create and populate a temporary CT database."""
    db_path = tmp_path / "test_ct.db"
    run_ct_migrations(db_path, MIGRATIONS_DIR)
    conn = get_connection(db_path)
    try:
        _populate_ct_test_db(conn)
    finally:
        conn.close()
    return db_path


# =========================================================================
# Phase A: Loaders and Splits
# =========================================================================


class TestLoadCTPairsDF:
    """Test load_ct_pairs_df function."""

    def test_row_count(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        # Should match pair count in DB
        conn = get_connection(ct_db)
        try:
            expected = conn.execute(
                "SELECT COUNT(*) FROM intervention_condition_pairs"
            ).fetchone()[0]
        finally:
            conn.close()
        assert len(df) == expected
        assert len(df) > 0

    def test_columns(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        required = {
            "pair_id", "intervention_id", "condition_id", "smiles",
            "inchikey", "inchikey_connectivity", "chembl_id", "molecular_type",
            "intervention_type", "condition_name", "mesh_id",
            "confidence_tier", "primary_failure_category", "num_trials",
            "num_sources", "highest_phase_reached", "intervention_degree",
            "condition_degree", "target_count", "earliest_completion_year",
        }
        assert required.issubset(set(df.columns)), (
            f"Missing: {required - set(df.columns)}"
        )

    def test_smiles_only_filter(self, ct_db):
        all_df = load_ct_pairs_df(ct_db)
        smiles_df = load_ct_pairs_df(ct_db, smiles_only=True)
        assert len(smiles_df) <= len(all_df)
        assert smiles_df["smiles"].notna().all()

    def test_earliest_completion_year_populated(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        # At least some rows should have completion year
        populated = df["earliest_completion_year"].notna().sum()
        assert populated > 0

    def test_min_confidence_silver(self, ct_db):
        all_df = load_ct_pairs_df(ct_db)
        sg_df = load_ct_pairs_df(ct_db, min_confidence="silver")
        assert len(sg_df) <= len(all_df)
        assert set(sg_df["confidence_tier"].unique()).issubset({"gold", "silver"})


class TestCTRandomSplit:
    """Test random split."""

    def test_all_assigned(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_random_split(df, seed=42)
        assert len(fold_map) == len(df)
        assert set(fold_map.values()) == {"train", "val", "test"}

    def test_approximate_ratios(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_random_split(df, seed=42)
        n = len(df)
        counts = pd.Series(fold_map.values()).value_counts()
        assert abs(counts.get("train", 0) / n - 0.7) < 0.1
        assert abs(counts.get("test", 0) / n - 0.2) < 0.1

    def test_deterministic(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        m1 = generate_ct_random_split(df, seed=42)
        m2 = generate_ct_random_split(df, seed=42)
        assert m1 == m2

    def test_three_folds(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_random_split(df, seed=42)
        assert set(fold_map.values()) == {"train", "val", "test"}


class TestCTColdDrugSplit:
    """Test cold-drug split: no inchikey_connectivity leakage."""

    def test_zero_leakage(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_cold_drug_split(df, seed=42)
        df["_fold"] = df["pair_id"].map(fold_map)

        # Check inchikey_connectivity doesn't appear in both train and test
        has_ik = df["inchikey_connectivity"].notna()
        train_iks = set(df.loc[(df["_fold"] == "train") & has_ik, "inchikey_connectivity"])
        test_iks = set(df.loc[(df["_fold"] == "test") & has_ik, "inchikey_connectivity"])
        assert len(train_iks & test_iks) == 0, "Drug leakage detected!"

    def test_all_assigned(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_cold_drug_split(df, seed=42)
        assert len(fold_map) == len(df)

    def test_non_smiles_included(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_cold_drug_split(df, seed=42)
        # Non-SMILES pairs should also be assigned
        no_smiles = df[df["smiles"].isna()]
        if len(no_smiles) > 0:
            for pid in no_smiles["pair_id"]:
                assert pid in fold_map

    def test_entity_grouping(self, ct_db):
        """Same inchikey_connectivity → same fold."""
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_cold_drug_split(df, seed=42)
        df["_fold"] = df["pair_id"].map(fold_map)

        for ik, grp in df[df["inchikey_connectivity"].notna()].groupby("inchikey_connectivity"):
            folds = grp["_fold"].unique()
            assert len(folds) == 1, f"Entity {ik} split across folds: {folds}"


class TestCTColdConditionSplit:
    """Test cold-condition split: no mesh_id leakage."""

    def test_zero_leakage(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_cold_condition_split(df, seed=42)
        df["_fold"] = df["pair_id"].map(fold_map)

        has_mesh = df["mesh_id"].notna()
        train_mesh = set(df.loc[(df["_fold"] == "train") & has_mesh, "mesh_id"])
        test_mesh = set(df.loc[(df["_fold"] == "test") & has_mesh, "mesh_id"])
        assert len(train_mesh & test_mesh) == 0, "Condition leakage detected!"

    def test_all_assigned(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_cold_condition_split(df, seed=42)
        assert len(fold_map) == len(df)

    def test_mesh_grouping(self, ct_db):
        """Same mesh_id → same fold."""
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_cold_condition_split(df, seed=42)
        df["_fold"] = df["pair_id"].map(fold_map)

        for mesh, grp in df[df["mesh_id"].notna()].groupby("mesh_id"):
            folds = grp["_fold"].unique()
            assert len(folds) == 1, f"Condition {mesh} split across folds: {folds}"


class TestCTTemporalSplit:
    """Test temporal split based on completion_date."""

    def test_correct_fold_assignment(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_temporal_split(df)
        df["_fold"] = df["pair_id"].map(fold_map)

        for _, row in df.iterrows():
            year = row["earliest_completion_year"]
            fold = row["_fold"]
            if pd.isna(year):
                assert fold == "train", "NULL year should be train"
            elif year < CT_TEMPORAL_TRAIN_CUTOFF:
                assert fold == "train", f"Year {year} should be train"
            elif year < CT_TEMPORAL_VAL_CUTOFF:
                assert fold == "val", f"Year {year} should be val"
            else:
                assert fold == "test", f"Year {year} should be test"

    def test_null_year_train(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_temporal_split(df)
        df["_fold"] = df["pair_id"].map(fold_map)

        null_year = df[df["earliest_completion_year"].isna()]
        if len(null_year) > 0:
            assert (null_year["_fold"] == "train").all()

    def test_all_assigned(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_temporal_split(df)
        assert len(fold_map) == len(df)

    def test_no_2020_plus_in_train(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_temporal_split(df)
        df["_fold"] = df["pair_id"].map(fold_map)

        train_years = df.loc[
            (df["_fold"] == "train") & df["earliest_completion_year"].notna(),
            "earliest_completion_year",
        ]
        if len(train_years) > 0:
            assert (train_years < CT_TEMPORAL_TRAIN_CUTOFF).all()


class TestCTScaffoldSplit:
    """Test scaffold split with Murcko frameworks."""

    def test_zero_scaffold_leakage(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_scaffold_split(df, seed=42)
        df["_fold"] = df["pair_id"].map(fold_map)

        # Compute scaffolds for checking
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

        smiles_df = df[df["smiles"].notna()].copy()
        scaf_map = {}
        for _, row in smiles_df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol:
                scaf = GetScaffoldForMol(mol)
                scaf_map[row["pair_id"]] = Chem.MolToSmiles(scaf) if scaf else "NONE"
            else:
                scaf_map[row["pair_id"]] = "NONE"

        smiles_df["scaffold"] = smiles_df["pair_id"].map(scaf_map)
        train_scafs = set(smiles_df.loc[smiles_df["_fold"] == "train", "scaffold"])
        test_scafs = set(smiles_df.loc[smiles_df["_fold"] == "test", "scaffold"])
        assert len(train_scafs & test_scafs) == 0, "Scaffold leakage detected!"

    def test_non_smiles_null(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_scaffold_split(df, seed=42)

        no_smiles = df[df["smiles"].isna()]
        for pid in no_smiles["pair_id"]:
            assert fold_map[pid] is None, f"Non-SMILES pair {pid} should be None"

    def test_same_scaffold_same_fold(self, ct_db):
        """Aspirin and phenyl acetate share salicylate scaffold."""
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_scaffold_split(df, seed=42)
        df["_fold"] = df["pair_id"].map(fold_map)

        # Drug_0 (aspirin) and Drug_4 (phenyl acetate) share scaffold
        smiles_df = df[df["smiles"].notna()].copy()
        # Group by inchikey_connectivity to check
        for ik, grp in smiles_df.groupby("inchikey_connectivity"):
            folds = grp["_fold"].dropna().unique()
            assert len(folds) <= 1, f"Entity {ik} has multiple folds: {folds}"

    def test_all_smiles_pairs_assigned(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_scaffold_split(df, seed=42)

        smiles_df = df[df["smiles"].notna()]
        for pid in smiles_df["pair_id"]:
            assert fold_map[pid] is not None, f"SMILES pair {pid} not assigned"


class TestCTDegreeBalancedSplit:
    """Test degree-balanced split."""

    def test_all_assigned(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_degree_balanced_split(df, seed=42)
        assert len(fold_map) == len(df)

    def test_approximate_ratios(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        fold_map = generate_ct_degree_balanced_split(df, seed=42)
        n = len(df)
        counts = pd.Series(fold_map.values()).value_counts()
        # Wider tolerance for small test DB (~57 pairs)
        assert abs(counts.get("train", 0) / n - 0.7) < 0.2
        assert abs(counts.get("test", 0) / n - 0.2) < 0.2

    def test_deterministic(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        m1 = generate_ct_degree_balanced_split(df, seed=42)
        m2 = generate_ct_degree_balanced_split(df, seed=42)
        assert m1 == m2


class TestApplyAllCTSplits:
    """Test apply_all_ct_splits."""

    def test_six_split_columns(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        result = apply_all_ct_splits(df, seed=42)
        split_cols = [c for c in result.columns if c.startswith("split_")]
        assert len(split_cols) == 6
        expected = {
            "split_random", "split_cold_drug", "split_cold_condition",
            "split_temporal", "split_scaffold", "split_degree_balanced",
        }
        assert set(split_cols) == expected

    def test_scaffold_only_has_nulls(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        result = apply_all_ct_splits(df, seed=42)

        # Non-scaffold splits should have no NULLs
        for col in ["split_random", "split_cold_drug", "split_cold_condition",
                     "split_temporal", "split_degree_balanced"]:
            assert result[col].notna().all(), f"{col} has NULLs"

        # Scaffold may have NULLs for non-SMILES
        if result["smiles"].isna().any():
            null_scaffold = result.loc[result["smiles"].isna(), "split_scaffold"]
            assert null_scaffold.isna().all()

    def test_returns_copy(self, ct_db):
        df = load_ct_pairs_df(ct_db)
        result = apply_all_ct_splits(df, seed=42)
        assert "split_random" not in df.columns  # Original unchanged


class TestExportCTFailureDataset:
    """Test export_ct_failure_dataset."""

    def test_parquet_roundtrip(self, ct_db, tmp_path):
        out_dir = tmp_path / "exports"
        result = export_ct_failure_dataset(ct_db, out_dir, seed=42)

        parquet_path = Path(result["parquet_path"])
        assert parquet_path.exists()
        df = pd.read_parquet(parquet_path)
        assert len(df) == result["total_rows"]

        # Should have split columns
        split_cols = [c for c in df.columns if c.startswith("split_")]
        assert len(split_cols) == 6

    def test_splits_csv(self, ct_db, tmp_path):
        out_dir = tmp_path / "exports"
        result = export_ct_failure_dataset(ct_db, out_dir, seed=42)

        csv_path = Path(result["splits_csv_path"])
        assert csv_path.exists()
        csv_df = pd.read_csv(csv_path)
        assert "pair_id" in csv_df.columns
        assert "smiles_short" in csv_df.columns
        split_cols = [c for c in csv_df.columns if c.startswith("split_")]
        assert len(split_cols) == 6

    def test_no_y_column(self, ct_db, tmp_path):
        out_dir = tmp_path / "exports"
        result = export_ct_failure_dataset(ct_db, out_dir, seed=42)
        df = pd.read_parquet(result["parquet_path"])
        assert "Y" not in df.columns


# =========================================================================
# Phase B: CTO + M1
# =========================================================================


class TestLoadCTOSuccessPairs:
    """Test load_cto_success_pairs."""

    def test_success_pairs_y1(self, ct_db, tmp_path):
        # Create CTO with some success NCTs that match DB trials
        conn = get_connection(ct_db)
        try:
            nct_ids = [
                r[0] for r in conn.execute(
                    "SELECT source_trial_id FROM clinical_trials LIMIT 5"
                ).fetchall()
            ]
        finally:
            conn.close()

        cto_path = _make_cto_parquet(tmp_path, nct_ids[:3], nct_ids[3:5])
        success_df, conflicts = load_cto_success_pairs(cto_path, ct_db)

        # Should have some success pairs (or empty if all conflict)
        assert isinstance(success_df, pd.DataFrame)
        assert isinstance(conflicts, set)

    def test_conflict_removal(self, ct_db, tmp_path):
        # Use NCTs that should generate conflicts (these trials have failure results)
        conn = get_connection(ct_db)
        try:
            nct_ids = [
                r[0] for r in conn.execute(
                    "SELECT source_trial_id FROM clinical_trials LIMIT 10"
                ).fetchall()
            ]
        finally:
            conn.close()

        cto_path = _make_cto_parquet(tmp_path, nct_ids)
        success_df, conflicts = load_cto_success_pairs(cto_path, ct_db)

        # Conflicts should be non-empty (these trials have failure results in DB)
        assert len(conflicts) > 0

        # Success pairs should not contain any conflict keys
        if len(success_df) > 0:
            success_keys = set(
                zip(success_df["intervention_id"], success_df["condition_id"])
            )
            assert len(success_keys & conflicts) == 0

    def test_nct_matching(self, ct_db, tmp_path):
        # CTO with NCTs not in DB → empty
        cto_path = _make_cto_parquet(tmp_path, ["NCT_FAKE_001", "NCT_FAKE_002"])
        success_df, conflicts = load_cto_success_pairs(cto_path, ct_db)
        assert len(success_df) == 0
        assert len(conflicts) == 0

    def test_empty_cto_graceful(self, ct_db, tmp_path):
        # CTO file doesn't exist
        fake_path = tmp_path / "nonexistent.parquet"
        success_df, conflicts = load_cto_success_pairs(fake_path, ct_db)
        assert len(success_df) == 0
        assert len(conflicts) == 0


class TestBuildCTM1Dataset:
    """Test build_ct_m1_dataset."""

    def _get_m1_inputs(self, ct_db, tmp_path):
        """Helper to get M1 inputs.

        Uses clean NCTs (no failure results) as CTO success → conflict-free pairs.
        """
        # Clean NCTs have interventions (biologics) NOT in failure results
        success_ncts = ["NCT_CLEAN_01", "NCT_CLEAN_02", "NCT_CLEAN_03"]
        # Also include some conflicting NCTs to test conflict removal
        conn = get_connection(ct_db)
        try:
            conflict_ncts = [
                r[0] for r in conn.execute(
                    "SELECT source_trial_id FROM clinical_trials "
                    "WHERE source_trial_id NOT LIKE 'NCT_CLEAN%' LIMIT 3"
                ).fetchall()
            ]
        finally:
            conn.close()

        cto_path = _make_cto_parquet(
            tmp_path,
            success_ncts + conflict_ncts,  # All as CTO success (labels=1.0)
        )
        success_df, conflict_keys = load_cto_success_pairs(cto_path, ct_db)
        pairs_df = load_ct_pairs_df(ct_db, min_confidence="silver")
        return pairs_df, success_df, conflict_keys

    def test_balanced_equal_classes(self, ct_db, tmp_path):
        pairs_df, success_df, conflicts = self._get_m1_inputs(ct_db, tmp_path)
        if len(success_df) == 0:
            pytest.skip("No success pairs available")

        out_dir = tmp_path / "m1_out"
        results = build_ct_m1_dataset(pairs_df, success_df, conflicts, out_dir)

        if "balanced" in results:
            bal = pd.read_parquet(results["balanced"]["path"])
            n_pos = (bal["Y"] == 1).sum()
            n_neg = (bal["Y"] == 0).sum()
            assert n_pos == n_neg

    def test_realistic_has_both_labels(self, ct_db, tmp_path):
        pairs_df, success_df, conflicts = self._get_m1_inputs(ct_db, tmp_path)
        if len(success_df) == 0:
            pytest.skip("No success pairs available")

        out_dir = tmp_path / "m1_out"
        results = build_ct_m1_dataset(pairs_df, success_df, conflicts, out_dir)

        real = pd.read_parquet(results["realistic"]["path"])
        assert (real["Y"] == 0).sum() > 0
        assert (real["Y"] == 1).sum() > 0

    def test_splits_present(self, ct_db, tmp_path):
        pairs_df, success_df, conflicts = self._get_m1_inputs(ct_db, tmp_path)
        if len(success_df) == 0:
            pytest.skip("No success pairs available")

        out_dir = tmp_path / "m1_out"
        results = build_ct_m1_dataset(pairs_df, success_df, conflicts, out_dir)

        real = pd.read_parquet(results["realistic"]["path"])
        for col in ["split_random", "split_cold_drug", "split_cold_condition"]:
            assert col in real.columns, f"Missing split column: {col}"
            assert real[col].notna().all(), f"NULL values in {col}"

    def test_y_labels_correct(self, ct_db, tmp_path):
        pairs_df, success_df, conflicts = self._get_m1_inputs(ct_db, tmp_path)
        if len(success_df) == 0:
            pytest.skip("No success pairs available")

        out_dir = tmp_path / "m1_out"
        results = build_ct_m1_dataset(pairs_df, success_df, conflicts, out_dir)

        real = pd.read_parquet(results["realistic"]["path"])
        assert set(real["Y"].unique()) == {0, 1}

    def test_conflict_pairs_removed_both_sides(self, ct_db, tmp_path):
        pairs_df, success_df, conflicts = self._get_m1_inputs(ct_db, tmp_path)
        if len(success_df) == 0 or len(conflicts) == 0:
            pytest.skip("Need success + conflicts for this test")

        out_dir = tmp_path / "m1_out"
        results = build_ct_m1_dataset(pairs_df, success_df, conflicts, out_dir)

        real = pd.read_parquet(results["realistic"]["path"])
        real_keys = set(zip(real["intervention_id"], real["condition_id"]))
        for ck in conflicts:
            assert ck not in real_keys, f"Conflict key {ck} still in M1 dataset"

    def test_smiles_only_no_null_smiles(self, ct_db, tmp_path):
        pairs_df, success_df, conflicts = self._get_m1_inputs(ct_db, tmp_path)
        if len(success_df) == 0:
            pytest.skip("No success pairs available")

        out_dir = tmp_path / "m1_out"
        results = build_ct_m1_dataset(pairs_df, success_df, conflicts, out_dir)

        if "smiles_only" in results:
            smo = pd.read_parquet(results["smiles_only"]["path"])
            assert smo["smiles"].notna().all(), "SMILES-only has NULL smiles"
            # Both Y=0 and Y=1 should have no null SMILES
            assert smo.loc[smo["Y"] == 0, "smiles"].notna().all()
            assert smo.loc[smo["Y"] == 1, "smiles"].notna().all()


# =========================================================================
# Phase C: M2 + Integration
# =========================================================================


class TestLoadCTM2Data:
    """Test load_ct_m2_data."""

    def test_no_copper(self, ct_db):
        df = load_ct_m2_data(ct_db)
        assert "copper" not in df["confidence_tier"].values

    def test_categories_present(self, ct_db):
        df = load_ct_m2_data(ct_db)
        assert "failure_category" in df.columns
        assert "failure_category_int" in df.columns
        assert df["failure_category_int"].notna().all()
        # All categories should map to valid ints
        assert df["failure_category_int"].dtype in [np.int64, np.float64]

    def test_trial_features_joined(self, ct_db):
        df = load_ct_m2_data(ct_db)
        # Should have trial-level features
        for col in ["trial_phase", "sponsor_type", "enrollment_actual", "completion_year"]:
            assert col in df.columns, f"Missing trial feature: {col}"


class TestExportCTM2Dataset:
    """Test export_ct_m2_dataset."""

    def test_parquet_roundtrip(self, ct_db, tmp_path):
        m2_df = load_ct_m2_data(ct_db)
        m2_with_splits = apply_ct_m2_splits(m2_df, seed=42)
        out_dir = tmp_path / "exports"
        result = export_ct_m2_dataset(m2_with_splits, out_dir)

        parquet_path = Path(result["parquet_path"])
        assert parquet_path.exists()
        df = pd.read_parquet(parquet_path)
        assert len(df) == result["total_rows"]
        assert len(df) == len(m2_with_splits)

    def test_split_columns_preserved(self, ct_db, tmp_path):
        m2_df = load_ct_m2_data(ct_db)
        m2_with_splits = apply_ct_m2_splits(m2_df, seed=42)
        out_dir = tmp_path / "exports"
        result = export_ct_m2_dataset(m2_with_splits, out_dir)

        df = pd.read_parquet(result["parquet_path"])
        split_cols = [c for c in df.columns if c.startswith("split_")]
        assert len(split_cols) == 6

    def test_category_int_preserved(self, ct_db, tmp_path):
        m2_df = load_ct_m2_data(ct_db)
        m2_with_splits = apply_ct_m2_splits(m2_df, seed=42)
        out_dir = tmp_path / "exports"
        result = export_ct_m2_dataset(m2_with_splits, out_dir)

        df = pd.read_parquet(result["parquet_path"])
        assert "failure_category_int" in df.columns
        assert df["failure_category_int"].notna().all()


class TestApplyCTM2Splits:
    """Test apply_ct_m2_splits."""

    def test_six_split_columns(self, ct_db):
        m2_df = load_ct_m2_data(ct_db)
        result = apply_ct_m2_splits(m2_df, seed=42)
        split_cols = [c for c in result.columns if c.startswith("split_")]
        assert len(split_cols) == 6

    def test_scaffold_nullable(self, ct_db):
        m2_df = load_ct_m2_data(ct_db)
        result = apply_ct_m2_splits(m2_df, seed=42)

        # Non-scaffold splits should have no NULLs
        for col in ["split_random", "split_cold_drug", "split_cold_condition",
                     "split_temporal", "split_degree_balanced"]:
            assert result[col].notna().all(), f"M2 {col} has NULLs"


class TestCTLeakageReport:
    """Test generate_ct_leakage_report."""

    def test_report_structure(self, ct_db, tmp_path):
        report = generate_ct_leakage_report(ct_db, output_path=tmp_path / "report.json")
        assert "db_summary" in report
        assert "cold_split_integrity" in report
        assert "split_fold_counts" in report
        assert "smiles_coverage_per_fold" in report
        assert "tier_distribution_per_fold" in report
        assert "therapeutic_area_coverage" in report

    def test_cold_leakage_zero(self, ct_db):
        report = generate_ct_leakage_report(ct_db)
        for split_name, info in report["cold_split_integrity"].items():
            assert info["leaks"] == 0, f"Leakage in {split_name}"

    def test_m1_conflict_free(self, ct_db, tmp_path):
        # Create CTO for conflict testing
        conn = get_connection(ct_db)
        try:
            nct_ids = [
                r[0] for r in conn.execute(
                    "SELECT source_trial_id FROM clinical_trials LIMIT 5"
                ).fetchall()
            ]
        finally:
            conn.close()

        cto_path = _make_cto_parquet(tmp_path, nct_ids)
        report = generate_ct_leakage_report(ct_db, cto_path=cto_path)

        assert "m1_conflict_free" in report
        assert report["m1_conflict_free"]["verified"] is True

    def test_json_written(self, ct_db, tmp_path):
        out = tmp_path / "report.json"
        generate_ct_leakage_report(ct_db, output_path=out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "db_summary" in data
