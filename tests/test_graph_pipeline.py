"""Tests for the NegBioGraph build, contradiction, and query pipeline."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from negbiodb.db import create_database, get_connection as get_dti_connection, refresh_all_pairs
from negbiodb_ct.ct_db import create_ct_database, get_connection as get_ct_connection, refresh_all_ct_pairs
from negbiodb_ppi.ppi_db import create_ppi_database, get_connection as get_ppi_connection, refresh_all_ppi_pairs
from negbiodb_depmap.depmap_db import create_ge_database, get_connection as get_ge_connection, refresh_all_ge_pairs
from negbiodb_vp.vp_db import create_vp_database, get_connection as get_vp_connection, refresh_all_vp_pairs
from negbiodb_md.md_db import create_md_database, get_connection as get_md_connection, refresh_all_pairs as refresh_all_md_pairs
from negbiodb_dc.dc_db import create_dc_database, get_connection as get_dc_connection, refresh_all_drug_pairs
from negbiodb_graph.builder import build_graph
from negbiodb_graph.contradictions import build_contradictions
from negbiodb_graph.db import create_graph_database, get_connection as get_graph_connection
from negbiodb_graph.duckdb_marts import materialize_duckdb
from negbiodb_graph.queries import run_example_queries
from scripts_graph.build_contradictions import main as build_contradictions_main
from scripts_graph.build_graph import main as build_graph_main
from scripts_graph.materialize_duckdb import main as materialize_duckdb_main
from scripts_graph.run_example_queries import main as run_example_queries_main
from tests.cp_test_utils import create_seeded_cp_database


ROOT = Path(__file__).resolve().parents[1]
DTI_MIGRATIONS = ROOT / "migrations"
CT_MIGRATIONS = ROOT / "migrations_ct"
PPI_MIGRATIONS = ROOT / "migrations_ppi"
GE_MIGRATIONS = ROOT / "migrations_depmap"
VP_MIGRATIONS = ROOT / "migrations_vp"
MD_MIGRATIONS = ROOT / "migrations_md"
DC_MIGRATIONS = ROOT / "migrations_dc"

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
ASPIRIN_INCHIKEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
IBUPROFEN_SMILES = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
IBUPROFEN_INCHIKEY = "HEFNNWSXXWATRW-UHFFFAOYSA-N"


def _seed_dti_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "dti.db"
    create_database(db_path, DTI_MIGRATIONS)
    conn = get_dti_connection(db_path)
    try:
        conn.execute(
            """INSERT INTO compounds
               (canonical_smiles, inchikey, inchikey_connectivity, pubchem_cid, chembl_id)
               VALUES (?, ?, ?, ?, ?)""",
            (ASPIRIN_SMILES, ASPIRIN_INCHIKEY, ASPIRIN_INCHIKEY[:14], 2244, "CHEMBL25"),
        )
        compound_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO targets
               (uniprot_accession, gene_symbol, amino_acid_sequence, ncbi_gene_id)
               VALUES (?, ?, ?, ?)""",
            ("P00533", "EGFR", "MSEQEGFR", 1956),
        )
        target_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO assays
               (source_db, source_assay_id, assay_type, assay_format, screen_type)
               VALUES (?, ?, ?, ?, ?)""",
            ("chembl", "A1", "binding", "biochemical", "confirmatory_dose_response"),
        )
        assay_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for source_db, record_id in [("chembl", "DTI-NEG-1"), ("pubchem", "DTI-NEG-2")]:
            conn.execute(
                """INSERT INTO negative_results
                   (compound_id, target_id, assay_id, result_type, confidence_tier,
                    source_db, source_record_id, extraction_method, publication_year)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    compound_id,
                    target_id,
                    assay_id,
                    "hard_negative",
                    "gold",
                    source_db,
                    record_id,
                    "database_direct",
                    2021,
                ),
            )
        refresh_all_pairs(conn)
        conn.commit()
    finally:
        conn.close()
    return db_path


def _seed_ct_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "ct.db"
    create_ct_database(db_path, CT_MIGRATIONS)
    conn = get_ct_connection(db_path)
    try:
        conn.execute(
            """INSERT INTO interventions
               (intervention_type, intervention_name, canonical_name, chembl_id,
                pubchem_cid, canonical_smiles, inchikey, inchikey_connectivity,
                molecular_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "drug",
                "Aspirin",
                "aspirin",
                "CHEMBL25",
                2244,
                ASPIRIN_SMILES,
                ASPIRIN_INCHIKEY,
                ASPIRIN_INCHIKEY[:14],
                "small_molecule",
            ),
        )
        intervention_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO conditions
               (condition_name, canonical_name, mesh_id, do_id, icd10_code)
               VALUES (?, ?, ?, ?, ?)""",
            ("Lung Cancer", "lung cancer", "D008175", "DOID:1324", "C34"),
        )
        condition_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO clinical_trials
               (source_db, source_trial_id, overall_status, trial_phase, primary_completion_date)
               VALUES (?, ?, ?, ?, ?)""",
            ("clinicaltrials_gov", "NCT00000001", "terminated", "phase_3", "2020-01-01"),
        )
        trial_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO trial_interventions (trial_id, intervention_id, arm_role) VALUES (?, ?, ?)",
            (trial_id, intervention_id, "experimental"),
        )
        conn.execute(
            "INSERT INTO trial_conditions (trial_id, condition_id) VALUES (?, ?)",
            (trial_id, condition_id),
        )
        conn.execute(
            """INSERT INTO intervention_targets
               (intervention_id, uniprot_accession, gene_symbol, source)
               VALUES (?, ?, ?, ?)""",
            (intervention_id, "P00533", "EGFR", "chembl"),
        )
        conn.execute(
            """INSERT INTO trial_failure_results
               (intervention_id, condition_id, trial_id, failure_category,
                confidence_tier, source_db, source_record_id, extraction_method,
                publication_year, highest_phase_reached)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                intervention_id,
                condition_id,
                trial_id,
                "efficacy",
                "gold",
                "clinicaltrials_gov",
                "CT-FAIL-1",
                "database_direct",
                2020,
                "phase_3",
            ),
        )
        refresh_all_ct_pairs(conn)
        conn.commit()
    finally:
        conn.close()
    return db_path


def _seed_ppi_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "ppi.db"
    create_ppi_database(db_path, PPI_MIGRATIONS)
    conn = get_ppi_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO proteins (uniprot_accession, gene_symbol, amino_acid_sequence) VALUES (?, ?, ?)",
            ("P00533", "EGFR", "MSEQEGFR"),
        )
        p1 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO proteins (uniprot_accession, gene_symbol, amino_acid_sequence) VALUES (?, ?, ?)",
            ("Q99999", "GENE2", "MSEQGENE2"),
        )
        p2 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO ppi_negative_results
               (protein1_id, protein2_id, experiment_id, evidence_type,
                confidence_tier, source_db, source_record_id, extraction_method,
                publication_year)
               VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)""",
            (p1, p2, "experimental_non_interaction", "gold", "intact", "PPI-NEG-1", "database_direct", 2021),
        )
        refresh_all_ppi_pairs(conn)
        conn.commit()
    finally:
        conn.close()
    return db_path


def _seed_ge_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "ge.db"
    create_ge_database(db_path, GE_MIGRATIONS)
    conn = get_ge_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO genes (entrez_id, gene_symbol, ensembl_id) VALUES (?, ?, ?)",
            (1956, "EGFR", "ENSG00000146648"),
        )
        gene_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO cell_lines
               (model_id, ccle_name, stripped_name, lineage, primary_disease)
               VALUES (?, ?, ?, ?, ?)""",
            ("ACH-000001", "U2OS", "u2os", "Bone", "Osteosarcoma"),
        )
        cell_line_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO ge_screens
               (source_db, depmap_release, screen_type, library, algorithm)
               VALUES (?, ?, ?, ?, ?)""",
            ("depmap", "24Q1", "crispr", "Avana", "Chronos"),
        )
        screen_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO ge_negative_results
               (gene_id, cell_line_id, screen_id, gene_effect_score, dependency_probability,
                evidence_type, confidence_tier, source_db, source_record_id, extraction_method)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                gene_id,
                cell_line_id,
                screen_id,
                0.15,
                0.05,
                "crispr_nonessential",
                "gold",
                "depmap",
                "GE-NEG-1",
                "score_threshold",
            ),
        )
        refresh_all_ge_pairs(conn)
        conn.execute(
            """INSERT INTO prism_compounds
               (broad_id, name, smiles, inchikey, chembl_id, pubchem_cid)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("BRD-A", "Aspirin", ASPIRIN_SMILES, ASPIRIN_INCHIKEY, "CHEMBL25", 2244),
        )
        prism_compound_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO prism_sensitivity
               (compound_id, cell_line_id, screen_type, auc, ic50, depmap_release)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (prism_compound_id, cell_line_id, "primary", 0.42, 1.2, "24Q1"),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _seed_vp_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "vp.db"
    create_vp_database(db_path, VP_MIGRATIONS)
    conn = get_vp_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO genes (entrez_id, gene_symbol, hgnc_id, ensembl_id) VALUES (?, ?, ?, ?)",
            (1956, "EGFR", "HGNC:3236", "ENSG00000146648"),
        )
        gene_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO variants
               (chromosome, position, ref_allele, alt_allele, variant_type,
                gene_id, hgvs_coding, hgvs_protein, consequence_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("7", 55181378, "G", "A", "single nucleotide variant", gene_id, "c.2573T>G", "p.Leu858Arg", "missense"),
        )
        variant_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO diseases
               (canonical_name, mondo_id, omim_id, medgen_cui)
               VALUES (?, ?, ?, ?)""",
            ("Lung Cancer", "MONDO:0004992", "211980", "C0242379"),
        )
        disease_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO vp_negative_results
               (variant_id, disease_id, classification, evidence_type, confidence_tier,
                source_db, source_record_id, extraction_method, submission_year,
                has_conflict, num_benign_criteria)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                variant_id,
                disease_id,
                "benign",
                "expert_reviewed",
                "gold",
                "clinvar",
                "VCV000001",
                "review_status",
                2022,
                1,
                2,
            ),
        )
        refresh_all_vp_pairs(conn)
        conn.execute(
            "INSERT INTO vp_cross_domain_genes (gene_id, domain, external_id) VALUES (?, ?, ?)",
            (gene_id, "ppi", "P00533"),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _seed_md_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "md.db"
    create_md_database(db_path, MD_MIGRATIONS)
    conn = get_md_connection(db_path)
    try:
        conn.execute(
            """INSERT INTO md_metabolites
               (name, pubchem_cid, inchikey, canonical_smiles)
               VALUES (?, ?, ?, ?)""",
            ("Aspirin", 2244, ASPIRIN_INCHIKEY, ASPIRIN_SMILES),
        )
        metabolite_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO md_diseases
               (name, mondo_id, mesh_id, doid, disease_category)
               VALUES (?, ?, ?, ?, ?)""",
            ("Lung Cancer", "MONDO:0004992", "D008175", "DOID:1324", "cancer"),
        )
        disease_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO md_studies
               (source, external_id, title, biofluid, platform, n_disease, n_control)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("metabolights", "MTBLS1", "Study 1", "blood", "lc_ms", 30, 30),
        )
        study_1 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO md_studies
               (source, external_id, title, biofluid, platform, n_disease, n_control)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("nmdr", "ST0001", "Study 2", "blood", "lc_ms", 30, 30),
        )
        study_2 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO md_biomarker_results
               (metabolite_id, disease_id, study_id, p_value, fdr, is_significant, tier)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (metabolite_id, disease_id, study_1, 0.2, 0.3, 0, "silver"),
        )
        conn.execute(
            """INSERT INTO md_biomarker_results
               (metabolite_id, disease_id, study_id, p_value, fdr, is_significant)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (metabolite_id, disease_id, study_2, 0.001, 0.01, 1),
        )
        refresh_all_md_pairs(conn)
        conn.commit()
    finally:
        conn.close()
    return db_path


def _seed_dc_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "dc.db"
    create_dc_database(db_path, DC_MIGRATIONS)
    conn = get_dc_connection(db_path)
    try:
        conn.execute(
            """INSERT INTO compounds
               (drug_name, pubchem_cid, inchikey, canonical_smiles, chembl_id)
               VALUES (?, ?, ?, ?, ?)""",
            ("Aspirin", 2244, ASPIRIN_INCHIKEY, ASPIRIN_SMILES, "CHEMBL25"),
        )
        compound_a = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO compounds
               (drug_name, pubchem_cid, inchikey, canonical_smiles, chembl_id)
               VALUES (?, ?, ?, ?, ?)""",
            ("Ibuprofen", 3672, IBUPROFEN_INCHIKEY, IBUPROFEN_SMILES, "CHEMBL521"),
        )
        compound_b = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO cell_lines
               (cell_line_name, depmap_model_id, cosmic_id, lineage, tissue, primary_disease)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("U2OS", "ACH-000001", 111, "Bone", "Bone", "Osteosarcoma"),
        )
        cell_line_1 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO cell_lines
               (cell_line_name, depmap_model_id, cosmic_id, lineage, tissue, primary_disease)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("A549", "ACH-000002", 222, "Lung", "Lung", "Lung Adenocarcinoma"),
        )
        cell_line_2 = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """INSERT INTO dc_synergy_results
               (compound_a_id, compound_b_id, cell_line_id, zip_score, bliss_score,
                synergy_class, confidence_tier, evidence_type, source_db, source_study_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (compound_a, compound_b, cell_line_1, 12.0, 10.0, "synergistic", "gold", "multi_cell_line", "drugcomb", "DC-1"),
        )
        conn.execute(
            """INSERT INTO dc_synergy_results
               (compound_a_id, compound_b_id, cell_line_id, zip_score, bliss_score,
                synergy_class, confidence_tier, evidence_type, source_db, source_study_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (compound_a, compound_b, cell_line_2, -12.0, -10.0, "antagonistic", "gold", "multi_cell_line", "nci_almanac", "DC-2"),
        )
        refresh_all_drug_pairs(conn)
        conn.execute(
            """INSERT INTO dc_cross_domain_cell_lines
               (cell_line_id, domain, external_id)
               VALUES (?, ?, ?)""",
            (cell_line_1, "ge", "ACH-000001"),
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _write_manifest(tmp_path: Path) -> Path:
    dti_feed = tmp_path / "dti_positive_pairs.parquet"
    pd.DataFrame(
        [
            {
                "inchikey": ASPIRIN_INCHIKEY,
                "uniprot_id": "P00533",
                "target_sequence": "MSEQEGFR",
                "publication_year": 2022,
            }
        ]
    ).to_parquet(dti_feed, index=False)

    ppi_feed = tmp_path / "ppi_positive_pairs.parquet"
    pd.DataFrame(
        [
            {
                "uniprot_id_1": "P00533",
                "uniprot_id_2": "Q99999",
                "sequence_1": "MSEQEGFR",
                "sequence_2": "MSEQGENE2",
            }
        ]
    ).to_parquet(ppi_feed, index=False)

    ge_feed = tmp_path / "ge_essential_pairs.parquet"
    pd.DataFrame(
        [
            {
                "entrez_id": 1956,
                "gene_symbol": "EGFR",
                "model_id": "ACH-000001",
                "ccle_name": "U2OS",
                "lineage": "Bone",
            }
        ]
    ).to_parquet(ge_feed, index=False)

    ct_feed = tmp_path / "ct_success_pairs.parquet"
    pd.DataFrame(
        [
            {
                "intervention_id": 1,
                "intervention_name": "Aspirin",
                "condition_name": "Lung Cancer",
                "mesh_id": "D008175",
                "do_id": "DOID:1324",
                "icd10_code": "C34",
                "inchikey": ASPIRIN_INCHIKEY,
                "inchikey_connectivity": ASPIRIN_INCHIKEY[:14],
                "chembl_id": "CHEMBL25",
                "pubchem_cid": 2244,
                "smiles": ASPIRIN_SMILES,
            }
        ]
    ).to_parquet(ct_feed, index=False)

    manifest_path = tmp_path / "reference-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "feeds": [
                    {"name": "dti_positives", "kind": "dti_positive_pairs", "path": str(dti_feed)},
                    {"name": "ppi_positives", "kind": "ppi_positive_pairs", "path": str(ppi_feed)},
                    {"name": "ge_positives", "kind": "ge_essential_pairs", "path": str(ge_feed)},
                    {"name": "ct_successes", "kind": "ct_success_pairs", "path": str(ct_feed)},
                ]
            }
        )
    )
    return manifest_path


@pytest.fixture
def seeded_graph_inputs(tmp_path: Path) -> dict[str, Path]:
    inputs = {
        "dti": _seed_dti_db(tmp_path),
        "ct": _seed_ct_db(tmp_path),
        "ppi": _seed_ppi_db(tmp_path),
        "ge": _seed_ge_db(tmp_path),
        "vp": _seed_vp_db(tmp_path),
        "md": _seed_md_db(tmp_path),
        "dc": _seed_dc_db(tmp_path),
        "cp": create_seeded_cp_database(tmp_path),
    }
    inputs["manifest"] = _write_manifest(tmp_path)
    return inputs


def test_graph_migrations_create_tables(tmp_path: Path):
    graph_db = tmp_path / "graph.db"
    create_graph_database(graph_db)
    conn = get_graph_connection(graph_db)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
    finally:
        conn.close()
    expected = {
        "schema_migrations",
        "graph_builds",
        "graph_build_inputs",
        "graph_entities",
        "graph_entity_aliases",
        "graph_bridges",
        "graph_claims",
        "graph_claim_entities",
        "graph_evidence",
        "graph_claim_rollups",
        "graph_contradiction_groups",
        "graph_contradiction_members",
    }
    assert expected.issubset(tables)


def test_build_graph_ingests_domains_and_reference_feeds(seeded_graph_inputs: dict[str, Path], tmp_path: Path):
    graph_db = tmp_path / "graph.db"
    result = build_graph(
        graph_db,
        domain_paths={code: path for code, path in seeded_graph_inputs.items() if code in {"dti", "ct", "ppi", "ge", "vp", "md", "dc", "cp"}},
        manifest_path=seeded_graph_inputs["manifest"],
    )
    assert result["graph_entities"] > 0
    assert result["graph_claims"] > 0
    assert result["graph_evidence"] > 0
    assert result["graph_claim_rollups"] > 0

    conn = get_graph_connection(graph_db)
    try:
        counts = conn.execute(
            "SELECT COUNT(*) FROM graph_claims WHERE claim_family='binding' AND claim_label='inactive_against'"
        ).fetchone()[0]
        evidence = conn.execute(
            "SELECT COUNT(*) FROM graph_evidence WHERE source_table='negative_results'"
        ).fetchone()[0]
        active_refs = conn.execute(
            "SELECT COUNT(*) FROM graph_claims WHERE claim_family='binding' AND claim_label='active_against'"
        ).fetchone()[0]
        available_inputs = conn.execute(
            "SELECT COUNT(*) FROM graph_build_inputs WHERE is_available = 1"
        ).fetchone()[0]
    finally:
        conn.close()

    assert counts >= 1
    assert evidence >= 2
    assert active_refs == 1
    assert available_inputs >= 12


def test_graph_contradictions_duckdb_and_queries(seeded_graph_inputs: dict[str, Path], tmp_path: Path):
    graph_db = tmp_path / "graph.db"
    duckdb_path = tmp_path / "graph.duckdb"
    output_path = tmp_path / "queries.json"
    build_graph(
        graph_db,
        domain_paths={code: path for code, path in seeded_graph_inputs.items() if code in {"dti", "ct", "ppi", "ge", "vp", "md", "dc", "cp"}},
        manifest_path=seeded_graph_inputs["manifest"],
    )
    contradiction_result = build_contradictions(graph_db)
    assert contradiction_result["group_count"] >= 4

    conn = get_graph_connection(graph_db)
    try:
        contradiction_types = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT contradiction_type FROM graph_contradiction_groups"
            ).fetchall()
        }
    finally:
        conn.close()

    assert "direct_label_conflict" in contradiction_types
    assert "context_specificity" in contradiction_types
    assert "mixed_consensus" in contradiction_types
    assert "submission_conflict" in contradiction_types

    pytest.importorskip("duckdb")
    marts = materialize_duckdb(graph_db, duckdb_path)
    assert marts["mart_claims_rows"] > 0
    assert marts["mart_cross_domain_paths_rows"] > 0

    queries = run_example_queries(graph_db, output_path)
    assert queries["dti_ct_ge_paths"]
    assert queries["dti_cp_paths"]
    assert queries["ppi_direct_conflicts"]
    assert queries["md_mixed_consensus"]
    assert queries["dc_context_specificity"]
    assert queries["vp_submission_conflicts"]
    assert output_path.exists()


def test_graph_cli_smoke(seeded_graph_inputs: dict[str, Path], tmp_path: Path):
    graph_db = tmp_path / "graph.db"
    duckdb_path = tmp_path / "graph.duckdb"
    output_path = tmp_path / "queries.json"
    build_args = [
        "--graph-db",
        str(graph_db),
        "--manifest",
        str(seeded_graph_inputs["manifest"]),
        "--dti-db",
        str(seeded_graph_inputs["dti"]),
        "--ct-db",
        str(seeded_graph_inputs["ct"]),
        "--ppi-db",
        str(seeded_graph_inputs["ppi"]),
        "--ge-db",
        str(seeded_graph_inputs["ge"]),
        "--vp-db",
        str(seeded_graph_inputs["vp"]),
        "--md-db",
        str(seeded_graph_inputs["md"]),
        "--dc-db",
        str(seeded_graph_inputs["dc"]),
        "--cp-db",
        str(seeded_graph_inputs["cp"]),
    ]
    assert build_graph_main(build_args) == 0
    assert build_contradictions_main(["--graph-db", str(graph_db)]) == 0
    pytest.importorskip("duckdb")
    assert materialize_duckdb_main(["--graph-db", str(graph_db), "--duckdb", str(duckdb_path)]) == 0
    assert run_example_queries_main(["--graph-db", str(graph_db), "--output", str(output_path)]) == 0
    assert graph_db.exists()
    assert duckdb_path.exists()
    assert output_path.exists()
