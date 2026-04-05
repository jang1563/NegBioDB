"""Tests for DC LLM dataset builder module."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from negbiodb_dc.dc_db import get_connection, run_dc_migrations
from negbiodb_dc.llm_dataset import (
    FEWSHOT_SEEDS,
    L1_CLASS_MAP,
    L4_GROUPS,
    MAX_PER_DRUG,
    SYNERGY_DESCRIPTIONS,
    apply_max_per_drug,
    assign_splits,
    construct_l1_context,
    construct_l2_context,
    construct_l3_context,
    construct_l4_context,
    load_dc_candidate_pool,
    write_dataset_metadata,
    write_jsonl,
)

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_dc"


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_dc.db"
    run_dc_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    c = get_connection(tmp_db)
    yield c
    c.close()


def _seed_candidate_data(conn):
    """Seed compounds and drug_drug_pairs for candidate pool tests."""
    conn.execute(
        "INSERT INTO compounds (compound_id, drug_name, canonical_smiles, known_targets, atc_code) "
        "VALUES (1, 'Aspirin', 'CC(=O)OC1=CC=CC=C1C(=O)O', 'PTGS1;PTGS2', 'N02BA01')"
    )
    conn.execute(
        "INSERT INTO compounds (compound_id, drug_name, canonical_smiles, known_targets, atc_code) "
        "VALUES (2, 'Ibuprofen', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'PTGS1;PTGS2', 'M01AE01')"
    )
    conn.execute(
        "INSERT INTO compounds (compound_id, drug_name, canonical_smiles, known_targets, atc_code) "
        "VALUES (3, 'Caffeine', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'ADORA1;ADORA2A', 'N06BC01')"
    )
    conn.execute(
        "INSERT INTO drug_drug_pairs "
        "(pair_id, compound_a_id, compound_b_id, num_cell_lines, num_sources, "
        "num_measurements, median_zip, median_bliss, antagonism_fraction, synergy_fraction, "
        "consensus_class, best_confidence, num_shared_targets, target_jaccard, "
        "compound_a_degree, compound_b_degree) "
        "VALUES (1, 1, 2, 5, 2, 20, -8.5, -6.2, 0.70, 0.10, "
        "'antagonistic', 'gold', 2, 0.50, 3, 3)"
    )
    conn.execute(
        "INSERT INTO drug_drug_pairs "
        "(pair_id, compound_a_id, compound_b_id, num_cell_lines, num_sources, "
        "num_measurements, median_zip, median_bliss, antagonism_fraction, synergy_fraction, "
        "consensus_class, best_confidence, num_shared_targets, target_jaccard, "
        "compound_a_degree, compound_b_degree) "
        "VALUES (2, 1, 3, 3, 1, 10, 12.0, 8.5, 0.05, 0.85, "
        "'synergistic', 'silver', 0, 0.0, 3, 1)"
    )
    conn.commit()


# ── Constants tests ───────────────────────────────────────────────────


class TestConstants:
    def test_l1_class_map_values_subset(self):
        # L1_CLASS_MAP maps DB consensus_class → L1 letter.
        # DB only has: synergistic, additive, antagonistic, context_dependent.
        # L1 gold_answer (A-D) is assigned by ZIP threshold, not consensus_class,
        # so A and D need not appear in L1_CLASS_MAP values.
        assert set(L1_CLASS_MAP.values()).issubset({"A", "B", "C", "D"})

    def test_l1_class_map_keys(self):
        # Must only contain valid DB consensus_class values (no strongly_* which don't exist in DB)
        expected = {"synergistic", "additive", "antagonistic", "context_dependent"}
        assert set(L1_CLASS_MAP.keys()) == expected

    def test_synergy_descriptions_covers_detail_classes(self):
        # SYNERGY_DESCRIPTIONS provides display text for more granular classes
        # (including strongly_synergistic / strongly_antagonistic used in context text).
        # It does NOT need to match L1_CLASS_MAP keys.
        assert "synergistic" in SYNERGY_DESCRIPTIONS
        assert "antagonistic" in SYNERGY_DESCRIPTIONS
        assert "additive" in SYNERGY_DESCRIPTIONS

    def test_l4_groups(self):
        assert set(L4_GROUPS.keys()) == {"classic_combos", "recent_combos", "untested_plausible", "untested_rare"}

    def test_fewshot_seeds(self):
        assert len(FEWSHOT_SEEDS) == 3
        assert all(isinstance(s, int) for s in FEWSHOT_SEEDS)


# ── Candidate pool tests ─────────────────────────────────────────────


class TestCandidatePool:
    def test_load_basic(self, conn):
        _seed_candidate_data(conn)
        df = load_dc_candidate_pool(conn)
        assert len(df) == 2
        assert "pair_id" in df.columns
        assert "drug_a_name" in df.columns
        assert "drug_b_name" in df.columns

    def test_load_with_tier_filter(self, conn):
        _seed_candidate_data(conn)
        df = load_dc_candidate_pool(conn, min_confidence="gold")
        assert len(df) == 1
        assert df.iloc[0]["drug_a_name"] == "Aspirin"

    def test_load_with_class_filter(self, conn):
        _seed_candidate_data(conn)
        df = load_dc_candidate_pool(conn, consensus_class="antagonistic")
        assert len(df) == 1
        assert df.iloc[0]["consensus_class"] == "antagonistic"

    def test_load_empty(self, conn):
        df = load_dc_candidate_pool(conn)
        assert len(df) == 0

    def test_columns_present(self, conn):
        _seed_candidate_data(conn)
        df = load_dc_candidate_pool(conn)
        expected_cols = {
            "pair_id", "compound_a_id", "compound_b_id",
            "drug_a_name", "drug_b_name", "smiles_a", "smiles_b",
            "drug_a_targets", "drug_b_targets", "atc_a", "atc_b",
            "num_cell_lines", "num_sources", "num_measurements",
            "median_zip", "median_bliss", "antagonism_fraction", "synergy_fraction",
            "consensus_class", "confidence_tier",
            "num_shared_targets", "target_jaccard",
            "compound_a_degree", "compound_b_degree",
        }
        assert expected_cols.issubset(set(df.columns))


# ── Context construction tests ────────────────────────────────────────


class TestContextConstruction:
    @pytest.fixture
    def sample_row(self):
        return pd.Series({
            "drug_a_name": "Aspirin",
            "drug_b_name": "Ibuprofen",
            "drug_a_targets": "PTGS1;PTGS2",
            "drug_b_targets": "PTGS1;PTGS2",
            "smiles_a": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "smiles_b": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "atc_a": "N02BA01",
            "atc_b": "M01AE01",
            "num_cell_lines": 5,
            "num_sources": 2,
            "num_shared_targets": 2,
            "target_jaccard": 0.50,
            "num_measurements": 20,
            "median_zip": -8.5,
            "median_bliss": -6.2,
            "antagonism_fraction": 0.70,
            "synergy_fraction": 0.10,
            "consensus_class": "antagonistic",
        })

    def test_l1_context_has_drug_names(self, sample_row):
        ctx = construct_l1_context(sample_row)
        assert "Aspirin" in ctx
        assert "Ibuprofen" in ctx

    def test_l1_context_has_targets(self, sample_row):
        ctx = construct_l1_context(sample_row)
        assert "PTGS1" in ctx

    def test_l1_context_has_cell_lines(self, sample_row):
        ctx = construct_l1_context(sample_row)
        assert "5" in ctx

    def test_l2_context_has_report_format(self, sample_row):
        ctx = construct_l2_context(sample_row)
        assert "Drug Combination Report" in ctx
        assert "Drug A" in ctx
        assert "Drug B" in ctx

    def test_l2_context_has_smiles(self, sample_row):
        ctx = construct_l2_context(sample_row)
        assert "CC(=O)OC1" in ctx

    def test_l2_context_has_atc(self, sample_row):
        ctx = construct_l2_context(sample_row)
        assert "N02BA01" in ctx

    def test_l3_context_has_outcome(self, sample_row):
        ctx = construct_l3_context(sample_row)
        assert "Antagonistic" in ctx or "antagonistic" in ctx.lower()

    def test_l3_context_has_zip(self, sample_row):
        ctx = construct_l3_context(sample_row)
        assert "-8.5" in ctx

    def test_l3_context_has_antagonism_fraction(self, sample_row):
        ctx = construct_l3_context(sample_row)
        assert "70%" in ctx

    def test_l4_context_minimal(self, sample_row):
        ctx = construct_l4_context(sample_row)
        assert "Aspirin" in ctx
        assert "Ibuprofen" in ctx
        # No SMILES or detailed data in L4
        assert "CC(=O)" not in ctx


# ── Sampling utility tests ────────────────────────────────────────────


class TestSamplingUtilities:
    def _make_df(self, n=50):
        """Create a synthetic DataFrame for sampling tests."""
        drugs = [f"Drug_{i}" for i in range(10)]
        rng = np.random.RandomState(42)
        rows = []
        for i in range(n):
            a, b = sorted(rng.choice(drugs, 2, replace=False))
            rows.append({"drug_a_name": a, "drug_b_name": b, "pair_id": i})
        return pd.DataFrame(rows)

    def test_apply_max_per_drug_reduces_count(self):
        df = self._make_df(100)
        result = apply_max_per_drug(df, max_per_drug=5)
        assert len(result) < len(df)

    def test_apply_max_per_drug_respects_cap(self):
        df = self._make_df(100)
        result = apply_max_per_drug(df, max_per_drug=5)
        # Count per drug
        from collections import Counter
        counts = Counter()
        for _, row in result.iterrows():
            counts[row["drug_a_name"]] += 1
            counts[row["drug_b_name"]] += 1
        assert all(c <= 5 for c in counts.values())

    def test_apply_max_per_drug_reproducible(self):
        df = self._make_df(100)
        r1 = apply_max_per_drug(df, max_per_drug=5, rng=np.random.RandomState(42))
        r2 = apply_max_per_drug(df, max_per_drug=5, rng=np.random.RandomState(42))
        assert list(r1["pair_id"]) == list(r2["pair_id"])

    def test_assign_splits_all_labeled(self):
        df = self._make_df(50)
        result = assign_splits(df, fewshot_size=5, val_size=5)
        assert "split" in result.columns
        assert set(result["split"]).issubset({"fewshot", "val", "test"})
        assert len(result) == 50

    def test_assign_splits_sizes(self):
        df = self._make_df(50)
        result = assign_splits(df, fewshot_size=5, val_size=5)
        counts = result["split"].value_counts()
        assert counts["fewshot"] == 5
        assert counts["val"] == 5
        assert counts["test"] == 40

    def test_assign_splits_no_val(self):
        df = self._make_df(50)
        result = assign_splits(df, fewshot_size=5, val_size=0)
        assert "val" not in result["split"].values


# ── I/O tests ─────────────────────────────────────────────────────────


class TestIO:
    def test_write_jsonl(self, tmp_path):
        import json
        records = [{"a": 1, "x": "hello"}, {"b": 2, "y": [1, 2, 3]}]
        out = tmp_path / "test.jsonl"
        n = write_jsonl(records, out)
        assert n == 2
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2
        # Validate each line is valid JSON and matches original record
        for line, original in zip(lines, records):
            parsed = json.loads(line)
            assert parsed == original

    def test_write_dataset_metadata(self, tmp_path):
        import json
        out = tmp_path / "meta.json"
        write_dataset_metadata(out, "dc-l1", 100, {"fewshot": 10, "test": 90})
        meta = json.loads(out.read_text())
        assert meta["task"] == "dc-l1"
        assert meta["domain"] == "dc"
        assert meta["n_records"] == 100
