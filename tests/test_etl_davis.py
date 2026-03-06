"""Tests for DAVIS ETL pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from negbiodb.db import connect, create_database
from negbiodb.etl_davis import (
    classify_affinities,
    insert_compounds,
    insert_negative_results,
    insert_target_variants,
    insert_targets,
    load_davis_csvs,
    parse_gene_name,
    standardize_all_compounds,
    standardize_compound,
    standardize_all_targets,
)
from negbiodb.db import refresh_all_pairs

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def migrated_db(tmp_path):
    """Create a fresh migrated database."""
    db_path = tmp_path / "test.db"
    create_database(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def sample_drugs_df():
    return pd.DataFrame({
        "Drug_Index": [0, 1],
        "CID": [11314340, 24889392],
        "Canonical_SMILES": [
            "CC1=C2C=C(C=CC2=NN1)C3=CC(=CN=C3)OCC(CC4=CC=CC=C4)N",
            "CC(C)(C)C1=CC(=NO1)NC(=O)NC2=CC=C(C=C2)C3=CN4C5=C(C=C(C=C5)OCCN6CCOCC6)SC4=N3",
        ],
        "Isomeric_SMILES": ["", ""],
    })


@pytest.fixture
def sample_proteins_df():
    return pd.DataFrame({
        "Protein_Index": [0, 1, 3],
        "Accession_Number": ["NP_055726.3", "NP_005148.2", "NP_005148.2"],
        "Gene_Name": ["AAK1", "ABL1(E255K)-phosphorylated", "ABL1(F317I)-phosphorylated"],
        "Sequence": ["MKKFF" * 20, "MLEICL" * 20, "MLEICL" * 20],
    })


@pytest.fixture
def sample_affinities_df():
    return pd.DataFrame({
        "Drug_Index": [0, 0, 0, 1, 1, 1],
        "Protein_Index": [0, 1, 3, 0, 1, 3],
        "Affinity": [7.37, 5.0, 5.0, 5.0, 6.5, 5.0],
    })


@pytest.fixture
def mock_refseq_mapping():
    return {
        "NP_055726.3": "Q2M2I8",
        "NP_005148.2": "P00519",
    }


# ============================================================
# TestParseGeneName
# ============================================================


class TestParseGeneName:

    def test_simple_gene(self):
        assert parse_gene_name("AAK1") == ("AAK1", None)

    def test_mutation(self):
        assert parse_gene_name("ABL1(E255K)-phosphorylated") == ("ABL1", "E255K")

    def test_mutation_no_suffix(self):
        assert parse_gene_name("ABL1(T315I)") == ("ABL1", "T315I")

    def test_phosphorylated_only(self):
        assert parse_gene_name("ABL1-phosphorylated") == ("ABL1", None)

    def test_nonphosphorylated(self):
        assert parse_gene_name("ABL1-nonphosphorylated") == ("ABL1", None)

    def test_domain_selector_jh1(self):
        assert parse_gene_name("JAK1(JH1domain-catalytic)") == ("JAK1", None)

    def test_domain_selector_jh2(self):
        assert parse_gene_name("TYK2(JH2domain-pseudokinase)") == ("TYK2", None)

    def test_domain_selector_kin_dom(self):
        assert parse_gene_name("RPS6KA4(Kin.Dom.1-N-terminal)") == ("RPS6KA4", None)

    def test_species_selector(self):
        assert parse_gene_name("PFCDPK1(P.falciparum)") == ("PFCDPK1", None)

    def test_deletion_mutation(self):
        assert parse_gene_name("EGFR(E746-A750del)") == ("EGFR", "E746-A750del")


# ============================================================
# TestStandardizeCompound
# ============================================================


class TestStandardizeCompound:

    def test_valid_smiles(self):
        result = standardize_compound("c1ccccc1", 123)
        assert result is not None
        assert result["inchikey"].startswith("UHOVQNZJYSORNB")

    def test_inchikey_format(self):
        result = standardize_compound("CC(=O)O", 176)
        assert len(result["inchikey"]) == 27
        assert result["inchikey"].count("-") == 2

    def test_computes_properties(self):
        result = standardize_compound("c1ccccc1", 123)
        assert result["molecular_weight"] > 0
        assert result["num_heavy_atoms"] == 6
        assert result["pubchem_cid"] == 123

    def test_invalid_smiles(self):
        result = standardize_compound("not_a_smiles", 0)
        assert result is None


# ============================================================
# TestClassifyAffinities
# ============================================================


class TestClassifyAffinities:

    def test_inactive_at_5(self):
        df = pd.DataFrame({"Affinity": [5.0]})
        result = classify_affinities(df)
        assert result["classification"].iloc[0] == "inactive"

    def test_active_at_7(self):
        df = pd.DataFrame({"Affinity": [7.0]})
        result = classify_affinities(df)
        assert result["classification"].iloc[0] == "active"

    def test_borderline(self):
        df = pd.DataFrame({"Affinity": [6.0]})
        result = classify_affinities(df)
        assert result["classification"].iloc[0] == "borderline"

    def test_distribution(self, sample_affinities_df):
        result = classify_affinities(sample_affinities_df)
        counts = result["classification"].value_counts()
        assert counts["inactive"] == 4
        assert counts["active"] == 1
        assert counts["borderline"] == 1


# ============================================================
# TestInsertCompounds
# ============================================================


class TestInsertCompounds:

    def test_insert_new(self, migrated_db):
        compounds = [standardize_compound("c1ccccc1", 241)]
        compounds[0]["drug_index"] = 0
        with connect(migrated_db) as conn:
            mapping = insert_compounds(conn, compounds)
            conn.commit()
        assert 0 in mapping
        assert mapping[0] > 0

    def test_idempotent(self, migrated_db):
        compounds = [standardize_compound("c1ccccc1", 241)]
        compounds[0]["drug_index"] = 0
        with connect(migrated_db) as conn:
            m1 = insert_compounds(conn, compounds)
            m2 = insert_compounds(conn, compounds)
            conn.commit()
        assert m1[0] == m2[0]

    def test_returns_mapping(self, migrated_db, sample_drugs_df):
        compounds = standardize_all_compounds(sample_drugs_df)
        with connect(migrated_db) as conn:
            mapping = insert_compounds(conn, compounds)
            conn.commit()
        assert len(mapping) == 2
        assert 0 in mapping
        assert 1 in mapping


# ============================================================
# TestInsertTargets
# ============================================================


class TestInsertTargets:

    def test_insert_new(self, migrated_db):
        targets = [{
            "protein_index": 0,
            "uniprot_accession": "Q2M2I8",
            "gene_symbol": "AAK1",
            "amino_acid_sequence": "MKKFF" * 20,
            "sequence_length": 100,
            "target_family": "kinase",
        }]
        with connect(migrated_db) as conn:
            mapping = insert_targets(conn, targets)
            conn.commit()
        assert 0 in mapping

    def test_idempotent(self, migrated_db):
        targets = [{
            "protein_index": 0,
            "uniprot_accession": "Q2M2I8",
            "gene_symbol": "AAK1",
            "amino_acid_sequence": "MKKFF" * 20,
            "sequence_length": 100,
            "target_family": "kinase",
        }]
        with connect(migrated_db) as conn:
            m1 = insert_targets(conn, targets)
            m2 = insert_targets(conn, targets)
            conn.commit()
        assert m1[0] == m2[0]

    def test_stores_sequence(self, migrated_db):
        seq = "MKKFF" * 20
        targets = [{
            "protein_index": 0,
            "uniprot_accession": "Q2M2I8",
            "gene_symbol": "AAK1",
            "amino_acid_sequence": seq,
            "sequence_length": len(seq),
            "target_family": "kinase",
        }]
        with connect(migrated_db) as conn:
            insert_targets(conn, targets)
            conn.commit()
            row = conn.execute(
                "SELECT amino_acid_sequence, sequence_length FROM targets WHERE uniprot_accession='Q2M2I8'"
            ).fetchone()
        assert row[0] == seq
        assert row[1] == len(seq)


# ============================================================
# TestTargetVariants
# ============================================================


class TestTargetVariants:

    def test_standardize_targets_uses_canonical_uniprot(
        self,
        sample_proteins_df,
        mock_refseq_mapping,
    ):
        targets = standardize_all_targets(sample_proteins_df, mock_refseq_mapping)
        accessions = {t["uniprot_accession"] for t in targets}

        # Canonical only: no mutation suffix in uniprot_accession
        assert accessions == {"Q2M2I8", "P00519"}
        assert all("_" not in acc for acc in accessions)

        # Variant labels are kept separately for later target_variants insertion
        labels = {t["variant_label"] for t in targets if t["variant_label"] is not None}
        assert labels == {"E255K", "F317I"}

    def test_insert_target_variants(self, migrated_db):
        targets = [
            {
                "protein_index": 0,
                "uniprot_accession": "Q2M2I8",
                "gene_symbol": "AAK1",
                "amino_acid_sequence": "MKKFF" * 20,
                "sequence_length": 100,
                "target_family": "kinase",
                "variant_label": None,
                "raw_gene_name": "AAK1",
            },
            {
                "protein_index": 1,
                "uniprot_accession": "P00519",
                "gene_symbol": "ABL1",
                "amino_acid_sequence": "MLEICL" * 20,
                "sequence_length": 120,
                "target_family": "kinase",
                "variant_label": "E255K",
                "raw_gene_name": "ABL1(E255K)-phosphorylated",
            },
            {
                "protein_index": 3,
                "uniprot_accession": "P00519",
                "gene_symbol": "ABL1",
                "amino_acid_sequence": "MLEICL" * 20,
                "sequence_length": 120,
                "target_family": "kinase",
                "variant_label": "F317I",
                "raw_gene_name": "ABL1(F317I)-phosphorylated",
            },
        ]

        with connect(migrated_db) as conn:
            target_map = insert_targets(conn, targets)
            prot_to_variant, n_variants = insert_target_variants(conn, targets, target_map)
            conn.commit()

            assert n_variants == 2
            assert 1 in prot_to_variant
            assert 3 in prot_to_variant
            row = conn.execute(
                "SELECT COUNT(*) FROM target_variants WHERE source_db='davis'"
            ).fetchone()
            assert row[0] == 2


# ============================================================
# TestInsertNegativeResults
# ============================================================


class TestInsertNegativeResults:

    def _setup_data(self, conn):
        """Insert minimal compound and target for testing."""
        conn.execute(
            """INSERT INTO compounds
            (canonical_smiles, inchikey, inchikey_connectivity, pubchem_cid)
            VALUES ('c1ccccc1', 'IMNFDUFMRHMDMM-UHFFFAOYSA-N', 'IMNFDUFMRHMDMM', 241)"""
        )
        cid = conn.execute("SELECT compound_id FROM compounds").fetchone()[0]
        conn.execute(
            """INSERT INTO targets (uniprot_accession, gene_symbol, sequence_length)
            VALUES ('Q2M2I8', 'AAK1', 100)"""
        )
        tid = conn.execute("SELECT target_id FROM targets").fetchone()[0]
        return cid, tid

    def test_insert(self, migrated_db):
        with connect(migrated_db) as conn:
            cid, tid = self._setup_data(conn)
            inactive_df = pd.DataFrame({
                "Drug_Index": [0],
                "Protein_Index": [0],
                "Affinity": [5.0],
            })
            total, skipped = insert_negative_results(
                conn, inactive_df, {0: cid}, {0: tid},
            )
            conn.commit()
        assert total == 1
        assert skipped == 0

    def test_confidence_bronze(self, migrated_db):
        with connect(migrated_db) as conn:
            cid, tid = self._setup_data(conn)
            inactive_df = pd.DataFrame({
                "Drug_Index": [0],
                "Protein_Index": [0],
                "Affinity": [5.0],
            })
            insert_negative_results(conn, inactive_df, {0: cid}, {0: tid})
            conn.commit()
            row = conn.execute(
                "SELECT confidence_tier FROM negative_results"
            ).fetchone()
        assert row[0] == "bronze"

    def test_result_type_hard_negative(self, migrated_db):
        with connect(migrated_db) as conn:
            cid, tid = self._setup_data(conn)
            inactive_df = pd.DataFrame({
                "Drug_Index": [0],
                "Protein_Index": [0],
                "Affinity": [5.0],
            })
            insert_negative_results(conn, inactive_df, {0: cid}, {0: tid})
            conn.commit()
            row = conn.execute(
                "SELECT result_type FROM negative_results"
            ).fetchone()
        assert row[0] == "hard_negative"

    def test_skips_unmapped(self, migrated_db):
        with connect(migrated_db) as conn:
            cid, tid = self._setup_data(conn)
            inactive_df = pd.DataFrame({
                "Drug_Index": [0, 99],
                "Protein_Index": [0, 0],
                "Affinity": [5.0, 5.0],
            })
            total, skipped = insert_negative_results(
                conn, inactive_df, {0: cid}, {0: tid},
            )
            conn.commit()
        assert total == 1
        assert skipped == 1

    def test_records_variant_id_when_provided(self, migrated_db):
        with connect(migrated_db) as conn:
            cid, tid = self._setup_data(conn)
            conn.execute(
                """INSERT INTO target_variants
                (target_id, variant_label, raw_gene_name, source_db, source_record_id)
                VALUES (?, 'E255K', 'ABL1(E255K)-phosphorylated', 'davis', 'DAVIS:PROTEIN:1')""",
                (tid,),
            )
            variant_id = conn.execute(
                "SELECT variant_id FROM target_variants WHERE source_record_id='DAVIS:PROTEIN:1'"
            ).fetchone()[0]

            inactive_df = pd.DataFrame({
                "Drug_Index": [0],
                "Protein_Index": [1],
                "Affinity": [5.0],
            })
            total, skipped = insert_negative_results(
                conn,
                inactive_df,
                {0: cid},
                {1: tid},
                variant_map={1: variant_id},
            )
            conn.commit()

            assert total == 1
            assert skipped == 0
            row = conn.execute(
                "SELECT variant_id FROM negative_results WHERE source_record_id='DAVIS:0_1'"
            ).fetchone()
            assert row[0] == variant_id


# ============================================================
# TestRefreshPairs
# ============================================================


class TestRefreshPairs:

    def test_creates_pair(self, migrated_db):
        with connect(migrated_db) as conn:
            conn.execute(
                """INSERT INTO compounds
                (canonical_smiles, inchikey, inchikey_connectivity)
                VALUES ('c1ccccc1', 'IMNFDUFMRHMDMM-UHFFFAOYSA-N', 'IMNFDUFMRHMDMM')"""
            )
            conn.execute(
                """INSERT INTO targets (uniprot_accession, gene_symbol, sequence_length)
                VALUES ('Q2M2I8', 'AAK1', 100)"""
            )
            conn.execute(
                """INSERT INTO negative_results
                (compound_id, target_id, result_type, confidence_tier,
                 activity_type, activity_value, activity_unit, pchembl_value,
                 inactivity_threshold, source_db, source_record_id, extraction_method,
                 publication_year)
                VALUES (1, 1, 'hard_negative', 'bronze',
                        'Kd', 10000.0, 'nM', 5.0,
                        10000.0, 'davis', 'DAVIS:0_0', 'database_direct', 2011)"""
            )
            count = refresh_all_pairs(conn)
            conn.commit()
        assert count == 1

    def test_pair_confidence(self, migrated_db):
        with connect(migrated_db) as conn:
            conn.execute(
                """INSERT INTO compounds
                (canonical_smiles, inchikey, inchikey_connectivity)
                VALUES ('c1ccccc1', 'IMNFDUFMRHMDMM-UHFFFAOYSA-N', 'IMNFDUFMRHMDMM')"""
            )
            conn.execute(
                """INSERT INTO targets (uniprot_accession, gene_symbol, sequence_length)
                VALUES ('Q2M2I8', 'AAK1', 100)"""
            )
            conn.execute(
                """INSERT INTO negative_results
                (compound_id, target_id, result_type, confidence_tier,
                 activity_type, activity_value, activity_unit, pchembl_value,
                 inactivity_threshold, source_db, source_record_id, extraction_method,
                 publication_year)
                VALUES (1, 1, 'hard_negative', 'bronze',
                        'Kd', 10000.0, 'nM', 5.0,
                        10000.0, 'davis', 'DAVIS:0_0', 'database_direct', 2011)"""
            )
            refresh_all_pairs(conn)
            conn.commit()
            row = conn.execute(
                "SELECT best_confidence FROM compound_target_pairs"
            ).fetchone()
        assert row[0] == "bronze"


# ============================================================
# Integration Test
# ============================================================


@pytest.mark.integration
class TestRunDavisETL:

    def test_full_etl(self, tmp_path):
        from negbiodb.etl_davis import run_davis_etl

        davis_dir = Path(__file__).resolve().parent.parent / "data" / "davis"
        if not davis_dir.exists():
            pytest.skip("DAVIS data not downloaded")

        db_path = tmp_path / "test.db"
        # Use skip_api=True with a pre-built cache or accept partial results
        stats = run_davis_etl(db_path, data_dir=davis_dir, skip_api=True)

        # Should have standardized all 68 compounds
        assert stats["compounds_inserted"] == 68

        # Results should only be inactive (pKd <= 5.0)
        with connect(db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM negative_results WHERE source_db='davis' AND pchembl_value > 5.0"
            ).fetchone()
            assert row[0] == 0

            tier = conn.execute(
                "SELECT DISTINCT confidence_tier FROM negative_results WHERE source_db='davis'"
            ).fetchone()
            if tier:
                assert tier[0] == "bronze"
