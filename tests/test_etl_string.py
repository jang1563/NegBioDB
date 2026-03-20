"""Tests for STRING v12.0 negative PPI ETL."""

from pathlib import Path

import pytest

from negbiodb_ppi.etl_string import (
    compute_protein_degrees,
    extract_zero_score_pairs,
    load_linked_pairs,
    load_string_mapping,
    run_string_etl,
)
from negbiodb_ppi.ppi_db import get_connection, run_ppi_migrations

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ppi"


@pytest.fixture
def ppi_db(tmp_path):
    db_path = tmp_path / "test_ppi.db"
    run_ppi_migrations(db_path, MIGRATIONS_DIR)
    return db_path


# --- Mapping file fixtures ---

MAPPING_LINES = """\
9606\tP00001|PROT1_HUMAN\t9606.ENSP00000000001\t100.0\t500.0
9606\tP00002|PROT2_HUMAN\t9606.ENSP00000000002\t100.0\t480.0
9606\tP00003|PROT3_HUMAN\t9606.ENSP00000000003\t100.0\t460.0
9606\tP00004|PROT4_HUMAN\t9606.ENSP00000000004\t100.0\t440.0
9606\tP00005|PROT5_HUMAN\t9606.ENSP00000000005\t100.0\t420.0
9606\tP00006|PROT6_HUMAN\t9606.ENSP00000000006\t100.0\t400.0
"""

# 6 proteins, links forming a triangle (P1-P2, P1-P3, P2-P3) + P4-P5, P4-P6, P5-P6
# Each of P1-P6 has degree >= 2
LINKS_LINES = """\
protein1 protein2 combined_score
9606.ENSP00000000001 9606.ENSP00000000002 900
9606.ENSP00000000001 9606.ENSP00000000003 800
9606.ENSP00000000002 9606.ENSP00000000003 700
9606.ENSP00000000004 9606.ENSP00000000005 600
9606.ENSP00000000004 9606.ENSP00000000006 500
9606.ENSP00000000005 9606.ENSP00000000006 400
"""


@pytest.fixture
def mapping_file(tmp_path):
    p = tmp_path / "human.uniprot_2_string.2018.tsv"
    p.write_text(MAPPING_LINES)
    return p


@pytest.fixture
def links_file(tmp_path):
    p = tmp_path / "9606.protein.links.v12.0.txt"
    p.write_text(LINKS_LINES)
    return p


class TestLoadStringMapping:
    def test_basic(self, mapping_file):
        m = load_string_mapping(mapping_file)
        assert m["9606.ENSP00000000001"] == "P00001"
        assert m["9606.ENSP00000000006"] == "P00006"
        assert len(m) == 6

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.tsv"
        p.write_text("")
        assert load_string_mapping(p) == {}

    def test_comment_lines(self, tmp_path):
        p = tmp_path / "commented.tsv"
        p.write_text("# header comment\n9606\tP00001|X\t9606.ENSP1\t100\t500\n")
        m = load_string_mapping(p)
        assert len(m) == 1

    def test_invalid_uniprot_skipped(self, tmp_path):
        p = tmp_path / "bad.tsv"
        p.write_text("9606\tinvalid|X\t9606.ENSP1\t100\t500\n")
        assert load_string_mapping(p) == {}

    def test_short_line_skipped(self, tmp_path):
        p = tmp_path / "short.tsv"
        p.write_text("9606\tP00001|X\n")  # Only 2 columns
        assert load_string_mapping(p) == {}

    def test_gzipped(self, tmp_path):
        import gzip

        p = tmp_path / "mapping.tsv.gz"
        with gzip.open(p, "wt") as f:
            f.write("9606\tP00001|X\t9606.ENSP1\t100\t500\n")
        m = load_string_mapping(p)
        assert m["9606.ENSP1"] == "P00001"


class TestComputeProteinDegrees:
    def test_basic_ensp(self, links_file):
        degrees = compute_protein_degrees(links_file)
        assert degrees["9606.ENSP00000000001"] == 2
        assert degrees["9606.ENSP00000000004"] == 2

    def test_with_uniprot_mapping(self, links_file, mapping_file):
        mapping = load_string_mapping(mapping_file)
        degrees = compute_protein_degrees(links_file, mapping)
        assert degrees["P00001"] == 2
        assert degrees["P00002"] == 2
        assert degrees["P00004"] == 2
        assert len(degrees) == 6

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.protein.links.txt"
        p.write_text("protein1 protein2 combined_score\n")
        assert compute_protein_degrees(p) == {}

    def test_header_skipped(self, links_file):
        degrees = compute_protein_degrees(links_file)
        assert "protein1" not in degrees


class TestLoadLinkedPairs:
    def test_basic(self, links_file, mapping_file):
        mapping = load_string_mapping(mapping_file)
        pairs = load_linked_pairs(links_file, mapping)
        assert len(pairs) == 6
        # All canonical ordered
        for a, b in pairs:
            assert a < b

    def test_unmapped_skipped(self, tmp_path):
        links = tmp_path / "links.txt"
        links.write_text(
            "protein1 protein2 combined_score\n"
            "9606.ENSP1 9606.ENSP2 900\n"
        )
        # Empty mapping → no pairs
        assert load_linked_pairs(links, {}) == set()

    def test_self_pair_skipped(self, tmp_path):
        links = tmp_path / "links.txt"
        links.write_text(
            "protein1 protein2 combined_score\n"
            "9606.ENSP1 9606.ENSP1 900\n"
        )
        mapping = {"9606.ENSP1": "P00001"}
        assert load_linked_pairs(links, mapping) == set()


class TestExtractZeroScorePairs:
    def test_basic_subtraction(self):
        linked = {("P00001", "P00002"), ("P00001", "P00003")}
        degrees = {"P00001": 5, "P00002": 5, "P00003": 5}
        result = extract_zero_score_pairs(linked, degrees, min_degree=5)
        # 3 choose 2 = 3 pairs, minus 2 linked = 1 pair
        assert result == [("P00002", "P00003")]

    def test_degree_filter(self):
        linked = set()
        degrees = {"P00001": 10, "P00002": 10, "P00003": 1}  # P3 low degree
        result = extract_zero_score_pairs(linked, degrees, min_degree=5)
        # Only P1 and P2 qualify → 1 pair
        assert result == [("P00001", "P00002")]

    def test_universe_filter(self):
        linked = set()
        degrees = {"P00001": 10, "P00002": 10, "P00003": 10}
        universe = {"P00001", "P00002"}  # P3 not in universe
        result = extract_zero_score_pairs(
            linked, degrees, min_degree=5, protein_universe=universe
        )
        assert result == [("P00001", "P00002")]

    def test_max_pairs_cap(self):
        linked = set()
        # 5 proteins → 10 pairs
        degrees = {f"P0000{i}": 10 for i in range(1, 6)}
        result = extract_zero_score_pairs(
            linked, degrees, min_degree=5, max_pairs=3
        )
        assert len(result) == 3

    def test_max_pairs_deterministic(self):
        linked = set()
        degrees = {f"P0000{i}": 10 for i in range(1, 6)}
        r1 = extract_zero_score_pairs(
            linked, degrees, min_degree=5, max_pairs=3, random_seed=42
        )
        r2 = extract_zero_score_pairs(
            linked, degrees, min_degree=5, max_pairs=3, random_seed=42
        )
        assert r1 == r2

    def test_empty_candidates(self):
        result = extract_zero_score_pairs(set(), {}, min_degree=5)
        assert result == []

    def test_invalid_uniprot_filtered(self):
        linked = set()
        degrees = {"P00001": 10, "invalid": 10, "P00002": 10}
        result = extract_zero_score_pairs(linked, degrees, min_degree=5)
        # Only P00001, P00002 pass validate_uniprot
        assert result == [("P00001", "P00002")]

    def test_results_sorted(self):
        linked = set()
        degrees = {f"P0000{i}": 10 for i in range(1, 5)}
        result = extract_zero_score_pairs(linked, degrees, min_degree=5)
        assert result == sorted(result)


class TestRunStringEtl:
    @pytest.fixture
    def string_data_dir(self, tmp_path):
        data_dir = tmp_path / "string"
        data_dir.mkdir()
        (data_dir / "human.uniprot_2_string.2018.tsv").write_text(MAPPING_LINES)
        (data_dir / "9606.protein.links.v12.0.txt").write_text(LINKS_LINES)
        return data_dir

    def test_basic_etl(self, ppi_db, string_data_dir):
        stats = run_string_etl(
            db_path=ppi_db,
            data_dir=string_data_dir,
            min_degree=2,
            max_pairs=500_000,
        )
        assert stats["mapping_entries"] == 6
        assert stats["linked_pairs"] == 6
        # 6 choose 2 = 15, minus 6 linked = 9
        assert stats["negative_pairs_derived"] == 9
        assert stats["negative_pairs_inserted"] == 9

    def test_all_bronze_tier(self, ppi_db, string_data_dir):
        run_string_etl(db_path=ppi_db, data_dir=string_data_dir, min_degree=2)
        conn = get_connection(ppi_db)
        try:
            tiers = conn.execute(
                "SELECT DISTINCT confidence_tier FROM ppi_negative_results"
            ).fetchall()
            assert tiers == [("bronze",)]

            evidence = conn.execute(
                "SELECT DISTINCT evidence_type FROM ppi_negative_results"
            ).fetchall()
            assert evidence == [("low_score_negative",)]
        finally:
            conn.close()

    def test_canonical_ordering(self, ppi_db, string_data_dir):
        run_string_etl(db_path=ppi_db, data_dir=string_data_dir, min_degree=2)
        conn = get_connection(ppi_db)
        try:
            rows = conn.execute(
                "SELECT protein1_id, protein2_id FROM ppi_negative_results"
            ).fetchall()
            for p1, p2 in rows:
                assert p1 < p2
        finally:
            conn.close()

    def test_experiment_record(self, ppi_db, string_data_dir):
        run_string_etl(db_path=ppi_db, data_dir=string_data_dir, min_degree=2)
        conn = get_connection(ppi_db)
        try:
            exp = conn.execute(
                "SELECT source_db, source_experiment_id FROM ppi_experiments "
                "WHERE source_db = 'string'"
            ).fetchone()
            assert exp is not None
            assert exp[1] == "string-v12.0-zero-score"
        finally:
            conn.close()

    def test_dataset_version(self, ppi_db, string_data_dir):
        run_string_etl(db_path=ppi_db, data_dir=string_data_dir, min_degree=2)
        conn = get_connection(ppi_db)
        try:
            dv = conn.execute(
                "SELECT name, version FROM dataset_versions WHERE name = 'string'"
            ).fetchone()
            assert dv[0] == "string"
            assert dv[1] == "v12.0"
        finally:
            conn.close()

    def test_proteins_inserted(self, ppi_db, string_data_dir):
        run_string_etl(db_path=ppi_db, data_dir=string_data_dir, min_degree=2)
        conn = get_connection(ppi_db)
        try:
            count = conn.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]
            assert count == 6
        finally:
            conn.close()

    def test_missing_mapping_file(self, ppi_db, tmp_path):
        empty_dir = tmp_path / "empty_string"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="mapping"):
            run_string_etl(db_path=ppi_db, data_dir=empty_dir)

    def test_missing_links_file(self, ppi_db, tmp_path):
        no_links = tmp_path / "no_links"
        no_links.mkdir()
        (no_links / "human.uniprot_2_string.2018.tsv").write_text(MAPPING_LINES)
        with pytest.raises(FileNotFoundError, match="links"):
            run_string_etl(db_path=ppi_db, data_dir=no_links)

    def test_protein_universe_restricts(self, ppi_db, string_data_dir):
        # Only allow P00001 and P00004 — they're not linked to each other
        stats = run_string_etl(
            db_path=ppi_db,
            data_dir=string_data_dir,
            min_degree=2,
            protein_universe={"P00001", "P00004"},
        )
        # 2 choose 2 = 1 pair, minus 0 linked between them = 1
        assert stats["negative_pairs_derived"] == 1
        assert stats["negative_pairs_inserted"] == 1

    def test_stats_keys(self, ppi_db, string_data_dir):
        stats = run_string_etl(
            db_path=ppi_db, data_dir=string_data_dir, min_degree=2
        )
        expected_keys = {
            "mapping_entries",
            "linked_pairs",
            "proteins_with_degree",
            "well_studied_proteins",
            "negative_pairs_derived",
            "negative_pairs_inserted",
        }
        assert set(stats.keys()) == expected_keys
