"""Tests for HuRI Y2H-derived negative PPI ETL."""

from pathlib import Path

import pytest

from negbiodb_ppi.etl_huri import (
    derive_huri_negatives,
    get_y2h_viable_proteins,
    load_huri_positives,
    load_orfeome_proteins,
    run_huri_etl,
)
from negbiodb_ppi.ppi_db import get_connection, run_ppi_migrations
from negbiodb_ppi.protein_mapper import load_ensg_mapping

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ppi"


@pytest.fixture
def ppi_db(tmp_path):
    db_path = tmp_path / "test_ppi.db"
    run_ppi_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def huri_data_dir(tmp_path):
    """Create mock HuRI data files."""
    data_dir = tmp_path / "huri"
    data_dir.mkdir()

    # Mock HI-union.tsv: 3 positive PPIs among 4 proteins
    # A-B, A-C, B-D (canonical pairs)
    ppi_file = data_dir / "HI-union.tsv"
    ppi_file.write_text(
        "P00001\tP00002\n"
        "P00001\tP00003\n"
        "P00002\tP00004\n"
    )

    # Mock ORFeome: 5 proteins (A,B,C,D,E — E has no positives)
    orfeome_file = data_dir / "orfeome.txt"
    orfeome_file.write_text(
        "P00001\n"
        "P00002\n"
        "P00003\n"
        "P00004\n"
        "P00005\n"
    )

    return data_dir


class TestLoadHuriPositives:
    def test_basic_parsing(self, tmp_path):
        f = tmp_path / "ppi.tsv"
        f.write_text("P00001\tP00002\nP00003\tP00004\n")
        pairs = load_huri_positives(f)
        assert pairs == {("P00001", "P00002"), ("P00003", "P00004")}

    def test_canonical_ordering(self, tmp_path):
        f = tmp_path / "ppi.tsv"
        f.write_text("P00002\tP00001\n")
        pairs = load_huri_positives(f)
        assert pairs == {("P00001", "P00002")}

    def test_skip_comments(self, tmp_path):
        f = tmp_path / "ppi.tsv"
        f.write_text("# header\nP00001\tP00002\n")
        pairs = load_huri_positives(f)
        assert len(pairs) == 1

    def test_skip_empty_lines(self, tmp_path):
        f = tmp_path / "ppi.tsv"
        f.write_text("P00001\tP00002\n\n\nP00003\tP00004\n")
        pairs = load_huri_positives(f)
        assert len(pairs) == 2

    def test_skip_invalid_uniprot(self, tmp_path):
        f = tmp_path / "ppi.tsv"
        f.write_text("P00001\tP00002\ninvalid\tP00003\n")
        pairs = load_huri_positives(f)
        assert len(pairs) == 1

    def test_skip_self_interactions(self, tmp_path):
        f = tmp_path / "ppi.tsv"
        f.write_text("P00001\tP00001\n")
        pairs = load_huri_positives(f)
        assert len(pairs) == 0

    def test_deduplicate(self, tmp_path):
        f = tmp_path / "ppi.tsv"
        f.write_text("P00001\tP00002\nP00002\tP00001\n")
        pairs = load_huri_positives(f)
        assert len(pairs) == 1

    def test_empty_file(self, tmp_path):
        f = tmp_path / "ppi.tsv"
        f.write_text("")
        pairs = load_huri_positives(f)
        assert len(pairs) == 0


class TestLoadOrfeomeProteins:
    def test_basic_parsing(self, tmp_path):
        f = tmp_path / "orfeome.txt"
        f.write_text("P00001\nP00002\nP00003\n")
        proteins = load_orfeome_proteins(f)
        assert proteins == {"P00001", "P00002", "P00003"}

    def test_skip_comments(self, tmp_path):
        f = tmp_path / "orfeome.txt"
        f.write_text("# header\nP00001\n")
        proteins = load_orfeome_proteins(f)
        assert proteins == {"P00001"}

    def test_skip_invalid(self, tmp_path):
        f = tmp_path / "orfeome.txt"
        f.write_text("P00001\ninvalid\nP00002\n")
        proteins = load_orfeome_proteins(f)
        assert proteins == {"P00001", "P00002"}

    def test_tab_separated(self, tmp_path):
        f = tmp_path / "orfeome.txt"
        f.write_text("P00001\tGene1\nP00002\tGene2\n")
        proteins = load_orfeome_proteins(f)
        assert proteins == {"P00001", "P00002"}


class TestGetY2hViableProteins:
    def test_basic_viable(self):
        orfeome = {"P00001", "P00002", "P00003", "P00004"}
        positives = {("P00001", "P00002"), ("P00001", "P00003")}
        viable = get_y2h_viable_proteins(orfeome, positives)
        assert viable == {"P00001", "P00002", "P00003"}

    def test_non_orfeome_excluded(self):
        orfeome = {"P00001", "P00002"}
        positives = {("P00001", "P00003")}  # P00003 not in ORFeome
        viable = get_y2h_viable_proteins(orfeome, positives)
        assert viable == {"P00001"}

    def test_no_positives(self):
        orfeome = {"P00001", "P00002"}
        positives = set()
        viable = get_y2h_viable_proteins(orfeome, positives)
        assert viable == set()

    def test_all_viable(self):
        orfeome = {"P00001", "P00002"}
        positives = {("P00001", "P00002")}
        viable = get_y2h_viable_proteins(orfeome, positives)
        assert viable == {"P00001", "P00002"}


class TestDeriveHuriNegatives:
    def test_basic_derivation(self):
        viable = {"P00001", "P00002", "P00003"}
        positives = {("P00001", "P00002")}
        negatives = derive_huri_negatives(viable, positives)
        # 3 choose 2 = 3 pairs, minus 1 positive = 2 negatives
        assert len(negatives) == 2
        assert ("P00001", "P00002") not in negatives
        assert ("P00001", "P00003") in negatives
        assert ("P00002", "P00003") in negatives

    def test_canonical_ordering(self):
        viable = {"P00002", "P00001"}
        positives = set()
        negatives = derive_huri_negatives(viable, positives)
        assert negatives == [("P00001", "P00002")]

    def test_max_pairs_cap(self):
        viable = {"P00001", "P00002", "P00003", "P00004"}
        positives = set()
        # 4 choose 2 = 6 pairs, cap at 3
        negatives = derive_huri_negatives(viable, positives, max_pairs=3)
        assert len(negatives) == 3

    def test_max_pairs_deterministic(self):
        viable = {"P00001", "P00002", "P00003", "P00004", "P00005"}
        positives = set()
        neg1 = derive_huri_negatives(viable, positives, max_pairs=3, random_seed=42)
        neg2 = derive_huri_negatives(viable, positives, max_pairs=3, random_seed=42)
        assert neg1 == neg2

    def test_empty_viable(self):
        negatives = derive_huri_negatives(set(), set())
        assert negatives == []

    def test_single_protein(self):
        negatives = derive_huri_negatives({"P00001"}, set())
        assert negatives == []

    def test_all_positives(self):
        viable = {"P00001", "P00002"}
        positives = {("P00001", "P00002")}
        negatives = derive_huri_negatives(viable, positives)
        assert negatives == []

    def test_sorted_output(self):
        viable = {"P00003", "P00001", "P00002"}
        positives = set()
        negatives = derive_huri_negatives(viable, positives)
        assert negatives == sorted(negatives)


class TestRunHuriEtl:
    def test_basic_etl(self, ppi_db, huri_data_dir):
        stats = run_huri_etl(
            db_path=ppi_db,
            data_dir=huri_data_dir,
            orfeome_file="orfeome.txt",
        )

        assert stats["orfeome_total"] == 5
        assert stats["positive_pairs"] == 3
        # Viable: P00001, P00002, P00003, P00004 (P00005 has no positives)
        assert stats["viable_proteins"] == 4
        # 4 choose 2 = 6 candidate pairs
        assert stats["candidate_pairs"] == 6
        # 6 - 3 positives = 3 negatives
        assert stats["negative_pairs_derived"] == 3

        # Verify DB contents
        conn = get_connection(ppi_db)
        try:
            # Check proteins inserted
            protein_count = conn.execute(
                "SELECT COUNT(*) FROM proteins"
            ).fetchone()[0]
            assert protein_count == 4  # only viable proteins

            # Check all negatives are gold tier
            tiers = conn.execute(
                "SELECT DISTINCT confidence_tier FROM ppi_negative_results"
            ).fetchall()
            assert tiers == [("gold",)]

            # Check experiment record
            exp = conn.execute(
                "SELECT source_db, experiment_type "
                "FROM ppi_experiments WHERE source_db = 'huri'"
            ).fetchone()
            assert exp[0] == "huri"
            assert exp[1] == "systematic_y2h"

            # Check dataset_versions
            dv = conn.execute(
                "SELECT name FROM dataset_versions WHERE name = 'huri'"
            ).fetchone()
            assert dv is not None
        finally:
            conn.close()

    def test_etl_without_orfeome(self, ppi_db, huri_data_dir):
        """Without ORFeome file, all proteins from positives are used."""
        stats = run_huri_etl(
            db_path=ppi_db,
            data_dir=huri_data_dir,
            orfeome_file=None,  # No ORFeome → derive from positives
        )

        # Without ORFeome, all 4 proteins from positives are viable
        assert stats["viable_proteins"] == 4
        assert stats["positive_pairs"] == 3
        assert stats["negative_pairs_derived"] == 3

    def test_etl_with_max_pairs(self, ppi_db, huri_data_dir):
        stats = run_huri_etl(
            db_path=ppi_db,
            data_dir=huri_data_dir,
            orfeome_file="orfeome.txt",
            max_pairs=2,
        )

        assert stats["negative_pairs_derived"] == 2

        conn = get_connection(ppi_db)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM ppi_negative_results"
            ).fetchone()[0]
            assert count == 2
        finally:
            conn.close()

    def test_etl_idempotent(self, ppi_db, huri_data_dir):
        """Running ETL twice should not duplicate records (INSERT OR IGNORE)."""
        stats1 = run_huri_etl(
            db_path=ppi_db,
            data_dir=huri_data_dir,
            orfeome_file="orfeome.txt",
        )
        stats2 = run_huri_etl(
            db_path=ppi_db,
            data_dir=huri_data_dir,
            orfeome_file="orfeome.txt",
        )

        conn = get_connection(ppi_db)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM ppi_negative_results"
            ).fetchone()[0]
            # Should be 3, not 6
            assert count == stats1["negative_pairs_derived"]
        finally:
            conn.close()


class TestLoadEnsgMapping:
    def test_basic(self, tmp_path):
        f = tmp_path / "mapping.tsv"
        f.write_text("ENSG00000000003\tP12345\nENSG00000000005\tQ9UHC1\n")
        mapping = load_ensg_mapping(f)
        assert mapping == {"ENSG00000000003": "P12345", "ENSG00000000005": "Q9UHC1"}

    def test_skip_comments(self, tmp_path):
        f = tmp_path / "mapping.tsv"
        f.write_text("# header\nENSG00000000003\tP12345\n")
        mapping = load_ensg_mapping(f)
        assert len(mapping) == 1

    def test_skip_invalid_uniprot(self, tmp_path):
        f = tmp_path / "mapping.tsv"
        f.write_text("ENSG00000000003\tinvalid\nENSG00000000005\tP12345\n")
        mapping = load_ensg_mapping(f)
        assert mapping == {"ENSG00000000005": "P12345"}

    def test_first_occurrence_wins(self, tmp_path):
        """One-to-many: first occurrence kept (SwissProt before TrEMBL)."""
        f = tmp_path / "mapping.tsv"
        f.write_text("ENSG00000000003\tP12345\nENSG00000000003\tA0A0K9P0T2\n")
        mapping = load_ensg_mapping(f)
        assert mapping["ENSG00000000003"] == "P12345"

    def test_empty_file(self, tmp_path):
        f = tmp_path / "mapping.tsv"
        f.write_text("")
        assert load_ensg_mapping(f) == {}


class TestLoadHuriPositivesWithEnsg:
    def test_ensg_ids_with_mapping(self, tmp_path):
        """ENSG IDs are translated to UniProt via mapping."""
        f = tmp_path / "ppi.tsv"
        f.write_text("ENSG00000000003\tENSG00000000005\n")
        mapping = {"ENSG00000000003": "P12345", "ENSG00000000005": "Q9UHC1"}
        pairs = load_huri_positives(f, ensg_mapping=mapping)
        assert pairs == {("P12345", "Q9UHC1")}

    def test_ensg_without_mapping_skipped(self, tmp_path):
        """ENSG IDs without mapping are skipped."""
        f = tmp_path / "ppi.tsv"
        f.write_text("ENSG00000000003\tENSG00000000005\n")
        pairs = load_huri_positives(f)  # No mapping provided
        assert len(pairs) == 0

    def test_mixed_uniprot_and_ensg(self, tmp_path):
        """One UniProt + one ENSG in same line works."""
        f = tmp_path / "ppi.tsv"
        f.write_text("P12345\tENSG00000000005\n")
        mapping = {"ENSG00000000005": "Q9UHC1"}
        pairs = load_huri_positives(f, ensg_mapping=mapping)
        assert pairs == {("P12345", "Q9UHC1")}

    def test_unmapped_ensg_counted(self, tmp_path):
        """Partial mapping: one mapped, one not → pair skipped."""
        f = tmp_path / "ppi.tsv"
        f.write_text(
            "ENSG00000000003\tENSG00000000005\n"
            "ENSG00000000007\tENSG00000000009\n"
        )
        mapping = {"ENSG00000000003": "P12345", "ENSG00000000005": "Q9UHC1"}
        pairs = load_huri_positives(f, ensg_mapping=mapping)
        # Only first line maps, second line both unmapped
        assert len(pairs) == 1


class TestRunHuriEtlWithEnsg:
    def test_etl_with_ensg_mapping(self, ppi_db, tmp_path):
        """Full ETL with ENSG-format HI-union.tsv and mapping file."""
        data_dir = tmp_path / "huri_ensg"
        data_dir.mkdir()

        # ENSG-format positive PPIs: 3 pairs among 4 genes
        ppi_file = data_dir / "HI-union.tsv"
        ppi_file.write_text(
            "ENSG00000000001\tENSG00000000002\n"
            "ENSG00000000001\tENSG00000000003\n"
            "ENSG00000000002\tENSG00000000004\n"
        )

        # ENSG→UniProt mapping
        mapping_file = data_dir / "ensg_to_uniprot.tsv"
        mapping_file.write_text(
            "ENSG00000000001\tP00001\n"
            "ENSG00000000002\tP00002\n"
            "ENSG00000000003\tP00003\n"
            "ENSG00000000004\tP00004\n"
        )

        stats = run_huri_etl(
            db_path=ppi_db,
            data_dir=data_dir,
        )

        # All 4 proteins viable, 4C2=6, minus 3 positives = 3 negatives
        assert stats["viable_proteins"] == 4
        assert stats["positive_pairs"] == 3
        assert stats["negative_pairs_derived"] == 3
