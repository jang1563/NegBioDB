"""Tests for MD LLM dataset builders (L1-L4)."""

import json
from pathlib import Path

import pytest

from negbiodb_md.md_db import get_md_connection, refresh_all_pairs
from negbiodb_md.llm_dataset import (
    L3_RUBRIC_AXES,
    build_l1_dataset,
    build_l2_dataset,
    build_l3_dataset,
    build_l4_dataset,
    load_md_candidate_pool,
    _generate_gold_reasoning,
)

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_md"


def _seed_md_db(conn, n_studies: int = 3, n_metabolites: int = 8):
    """Insert minimal MD data for LLM dataset testing."""
    # Metabolites
    metabolites = [
        ("glucose", "WQZGKKKJIJFFOK-GASJEMHNSA-N", "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "Carbohydrates and Carbohydrate Conjugates"),
        ("alanine", "QNAYBMKLOCPYGJ-REOHCLBHSA-N", "C[C@@H](N)C(O)=O", "Amino Acids and Analogues"),
        ("leucine", "ROHFNLRQFUQHCH-YFKPBYRVSA-N", "CC(C)C[C@@H](N)C(O)=O", "Amino Acids and Analogues"),
        ("palmitate", "IPCSVZSSVZVIGE-UHFFFAOYSA-N", "CCCCCCCCCCCCCCCC(O)=O", "Lipids and Lipid-Like Molecules"),
        ("pyruvate", "LCTONWCANYUPML-UHFFFAOYSA-M", "CC(=O)C([O-])=O", "Organic Acids and Derivatives"),
        ("urea", "XSQUKJJJFZCRTK-UHFFFAOYSA-N", "NC(N)=O", "Organonitrogen Compounds"),
        ("creatinine", "DDRJAANPRJIHGJ-UHFFFAOYSA-N", "CN1CC(=O)NC1=N", "Organonitrogen Compounds"),
        ("phenylalanine", "COLNVLDHVKWLRT-QMMMGPOBSA-N", "N[C@@H](Cc1ccccc1)C(O)=O", "Amino Acids and Analogues"),
    ]
    for name, inchikey, smiles, met_class in metabolites[:n_metabolites]:
        conn.execute(
            """INSERT OR IGNORE INTO md_metabolites
               (name, inchikey, canonical_smiles, metabolite_class)
               VALUES (?,?,?,?)""",
            (name, inchikey, smiles, met_class),
        )

    # Diseases
    diseases = [
        ("type 2 diabetes mellitus", "metabolic", "MONDO:0005148"),
        ("colorectal cancer", "cancer", "MONDO:0005575"),
        ("Alzheimer disease", "neurological", "MONDO:0004975"),
    ]
    for name, category, mondo_id in diseases:
        conn.execute(
            """INSERT OR IGNORE INTO md_diseases (name, disease_category, mondo_id)
               VALUES (?,?,?)""",
            (name, category, mondo_id),
        )

    # Studies
    studies = [
        ("metabolights", "MTBLS1", "Serum metabolomics in T2D patients", "blood", "lc_ms", 60, 55),
        ("nmdr", "ST000001", "Urine NMR metabolomics in colorectal cancer", "urine", "nmr", 30, 28),
        ("metabolights", "MTBLS2", "Plasma metabolomics in Alzheimer disease", "blood", "lc_ms", 45, 40),
    ]
    for src, eid, title, bf, plat, nd, nc in studies[:n_studies]:
        conn.execute(
            """INSERT OR IGNORE INTO md_studies
               (source, external_id, title, description, biofluid, platform,
                comparison, n_disease, n_control)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (src, eid, title, f"Description for {title}", bf, plat,
             "disease_vs_healthy", nd, nc),
        )

    # Results: mix of significant and non-significant
    # Study 1, Disease 1 (T2D): glucose significant, others not
    study_ids = [r[0] for r in conn.execute("SELECT study_id FROM md_studies ORDER BY study_id").fetchall()]
    dis_ids = [r[0] for r in conn.execute("SELECT disease_id FROM md_diseases ORDER BY disease_id").fetchall()]
    met_ids = [r[0] for r in conn.execute("SELECT metabolite_id FROM md_metabolites ORDER BY metabolite_id").fetchall()]

    if len(study_ids) >= 1 and len(dis_ids) >= 1 and len(met_ids) >= 1:
        # glucose significant in T2D (study 1)
        conn.execute(
            """INSERT INTO md_biomarker_results
               (metabolite_id, disease_id, study_id, p_value, fdr, is_significant)
               VALUES (?,?,?,?,?,?)""",
            (met_ids[0], dis_ids[0], study_ids[0], 0.001, 0.005, 1),
        )
        # alanine, leucine, palmitate NOT significant in T2D (distractors for L1)
        for i in [1, 2, 3]:
            if i < len(met_ids):
                conn.execute(
                    """INSERT INTO md_biomarker_results
                       (metabolite_id, disease_id, study_id, p_value, fdr, is_significant, tier)
                       VALUES (?,?,?,?,?,?,?)""",
                    (met_ids[i], dis_ids[0], study_ids[0], 0.3, 0.45, 0, "silver"),
                )

    conn.commit()
    refresh_all_pairs(conn)


@pytest.fixture
def md_conn(tmp_path):
    db_path = tmp_path / "md_test.db"
    conn = get_md_connection(db_path)
    _seed_md_db(conn)
    return conn


# ── load_md_candidate_pool ────────────────────────────────────────────────────

def test_load_candidate_pool_returns_dataframe(md_conn):
    df = load_md_candidate_pool(md_conn)
    assert len(df) > 0
    assert "metabolite_name" in df.columns
    assert "disease_name" in df.columns
    assert "is_significant" in df.columns


def test_load_candidate_pool_filter_negatives(md_conn):
    df = load_md_candidate_pool(md_conn, is_significant=0)
    assert all(df["is_significant"] == 0)


def test_load_candidate_pool_filter_positives(md_conn):
    df = load_md_candidate_pool(md_conn, is_significant=1)
    assert all(df["is_significant"] == 1)


# ── build_l1_dataset ──────────────────────────────────────────────────────────

def test_build_l1_dataset_creates_file(md_conn, tmp_path):
    out = build_l1_dataset(md_conn, n_records=5, seed=42, output_dir=tmp_path)
    assert out.exists()


def test_build_l1_dataset_valid_records(md_conn, tmp_path):
    out = build_l1_dataset(md_conn, n_records=5, seed=42, output_dir=tmp_path)
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    for rec in records:
        assert rec["task"] == "md_l1"
        assert "choices" in rec
        assert len(rec["choices"]) == 4
        assert rec["gold_answer"] in "ABCD"
        # Correct answer must be in choices
        assert rec["gold_answer"] in rec["choices"]


def test_build_l1_dataset_correct_answer_is_nonsignificant(md_conn, tmp_path):
    """The correct answer metabolite must be non-significant in the study."""
    out = build_l1_dataset(md_conn, n_records=5, seed=42, output_dir=tmp_path)
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    for rec in records:
        correct_metabolite = rec["choices"][rec["gold_answer"]]
        # Verify by checking metadata
        assert rec["metadata"]["p_value"] is None or rec["metadata"]["p_value"] > 0.05 or rec["metadata"]["fdr"] is None or rec["metadata"]["fdr"] > 0.05 or rec["metadata"]["tier"] is not None


# ── build_l2_dataset ──────────────────────────────────────────────────────────

def test_build_l2_dataset_creates_file(md_conn, tmp_path):
    out = build_l2_dataset(md_conn, n_records=5, seed=42, output_dir=tmp_path)
    assert out.exists()


def test_build_l2_dataset_valid_records(md_conn, tmp_path):
    out = build_l2_dataset(md_conn, n_records=5, seed=42, output_dir=tmp_path)
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    for rec in records:
        assert rec["task"] == "md_l2"
        assert "gold_fields" in rec
        gf = rec["gold_fields"]
        assert "metabolite" in gf
        assert "disease" in gf
        assert "outcome" in gf
        assert gf["outcome"] in ("significant", "not_significant")


# ── build_l3_dataset ──────────────────────────────────────────────────────────

def test_build_l3_dataset_creates_file(md_conn, tmp_path):
    out = build_l3_dataset(md_conn, n_records=5, seed=42, output_dir=tmp_path)
    assert out.exists()


def test_build_l3_dataset_has_gold_reasoning(md_conn, tmp_path):
    out = build_l3_dataset(md_conn, n_records=5, seed=42, output_dir=tmp_path)
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    for rec in records:
        assert rec["task"] == "md_l3"
        assert "gold_reasoning" in rec
        assert len(rec["gold_reasoning"]) > 20
        assert rec["rubric_axes"] == L3_RUBRIC_AXES


# ── build_l4_dataset ──────────────────────────────────────────────────────────

def test_build_l4_dataset_creates_file(md_conn, tmp_path):
    out = build_l4_dataset(md_conn, n_records=6, seed=42, output_dir=tmp_path)
    assert out.exists()


def test_build_l4_dataset_has_both_labels(md_conn, tmp_path):
    out = build_l4_dataset(md_conn, n_records=6, seed=42, output_dir=tmp_path)
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    labels = {rec["label"] for rec in records}
    assert 0 in labels  # synthetic
    assert 1 in labels  # real


def test_build_l4_dataset_valid_structure(md_conn, tmp_path):
    out = build_l4_dataset(md_conn, n_records=6, seed=42, output_dir=tmp_path)
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    for rec in records:
        assert rec["task"] == "md_l4"
        assert rec["label"] in (0, 1)
        assert rec["label_text"] in ("real", "synthetic")
        assert "context" in rec


# ── _generate_gold_reasoning ──────────────────────────────────────────────────

def test_generate_gold_reasoning_contains_key_info():
    import pandas as pd
    row = pd.Series({
        "metabolite_name": "glucose",
        "metabolite_class": "Carbohydrates",
        "disease_name": "type 2 diabetes mellitus",
        "disease_category": "metabolic",
        "platform": "lc_ms",
        "biofluid": "blood",
        "p_value": 0.25,
        "fdr": 0.40,
        "n_disease": 60,
    })
    reasoning = _generate_gold_reasoning(row)
    assert "glucose" in reasoning
    assert "type 2 diabetes" in reasoning.lower()
    assert len(reasoning) > 100
