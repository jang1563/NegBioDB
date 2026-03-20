"""Tests for IntAct negative PPI ETL."""

from pathlib import Path

import pytest

from negbiodb_ppi.etl_intact import (
    _parse_mi_id,
    _parse_mi_term,
    _parse_miscore,
    _parse_pubmed,
    _parse_taxon_id,
    _parse_uniprot_id,
    classify_tier,
    parse_mitab_line,
    run_intact_etl,
)
from negbiodb_ppi.ppi_db import get_connection, run_ppi_migrations

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations_ppi"


@pytest.fixture
def ppi_db(tmp_path):
    db_path = tmp_path / "test_ppi.db"
    run_ppi_migrations(db_path, MIGRATIONS_DIR)
    return db_path


def _make_mitab_line(
    uniprot_a="uniprotkb:P00001",
    uniprot_b="uniprotkb:P00002",
    detection="psi-mi:\"MI:0019\"(coimmunoprecipitation)",
    pubmed="pubmed:12345678",
    taxon_a="taxid:9606(Homo sapiens)",
    taxon_b="taxid:9606(Homo sapiens)",
    interaction_type="psi-mi:\"MI:0914\"(association)",
    interaction_id="EBI-12345",
    confidence="intact-miscore:0.56",
    negative="true",
):
    """Build a mock PSI-MI TAB 2.7 line (36+ columns)."""
    cols = [""] * 42
    cols[0] = uniprot_a
    cols[1] = uniprot_b
    cols[6] = detection
    cols[8] = pubmed
    cols[9] = taxon_a
    cols[10] = taxon_b
    cols[11] = interaction_type
    cols[13] = interaction_id
    cols[14] = confidence
    cols[35] = negative
    return "\t".join(cols)


class TestParseUniprotId:
    def test_basic(self):
        assert _parse_uniprot_id("uniprotkb:P12346") == "P12346"

    def test_isoform(self):
        assert _parse_uniprot_id("uniprotkb:P12346-2") == "P12346"

    def test_multi_value(self):
        assert _parse_uniprot_id("uniprotkb:P12346|chebi:12345") == "P12346"

    def test_non_uniprotkb(self):
        assert _parse_uniprot_id("chebi:12345") is None

    def test_empty(self):
        assert _parse_uniprot_id("") is None

    def test_dash_only(self):
        assert _parse_uniprot_id("-") is None


class TestParseTaxonId:
    def test_human(self):
        assert _parse_taxon_id("taxid:9606(Homo sapiens)") == 9606

    def test_mouse(self):
        assert _parse_taxon_id("taxid:10090(mouse)") == 10090

    def test_empty(self):
        assert _parse_taxon_id("-") is None


class TestParseMiId:
    def test_basic(self):
        assert _parse_mi_id('psi-mi:"MI:0018"(two hybrid)') == "MI:0018"

    def test_coip(self):
        assert _parse_mi_id('psi-mi:"MI:0019"(coimmunoprecipitation)') == "MI:0019"

    def test_no_match(self):
        assert _parse_mi_id("-") is None


class TestParseMiTerm:
    def test_basic(self):
        assert _parse_mi_term('psi-mi:"MI:0018"(two hybrid)') == "two hybrid"

    def test_coip(self):
        assert (
            _parse_mi_term('psi-mi:"MI:0019"(coimmunoprecipitation)')
            == "coimmunoprecipitation"
        )


class TestParsePubmed:
    def test_basic(self):
        assert _parse_pubmed("pubmed:12345678") == 12345678

    def test_multi(self):
        assert _parse_pubmed("pubmed:12345678|pubmed:99999") == 12345678

    def test_no_pubmed(self):
        assert _parse_pubmed("-") is None


class TestParseMiscore:
    def test_basic(self):
        assert _parse_miscore("intact-miscore:0.56") == pytest.approx(0.56)

    def test_no_score(self):
        assert _parse_miscore("-") is None


class TestClassifyTier:
    def test_gold_coip(self):
        assert classify_tier("MI:0019") == "gold"

    def test_gold_pulldown(self):
        assert classify_tier("MI:0096") == "gold"

    def test_gold_xray(self):
        assert classify_tier("MI:0114") == "gold"

    def test_gold_crosslink(self):
        assert classify_tier("MI:0030") == "gold"

    def test_silver_y2h(self):
        assert classify_tier("MI:0018") == "silver"

    def test_silver_unknown(self):
        assert classify_tier("MI:9999") == "silver"

    def test_silver_none(self):
        assert classify_tier(None) == "silver"


class TestParseMitabLine:
    def test_valid_negative(self):
        line = _make_mitab_line()
        result = parse_mitab_line(line)
        assert result is not None
        assert result["uniprot_a"] == "P00001"
        assert result["uniprot_b"] == "P00002"
        assert result["detection_method_id"] == "MI:0019"
        assert result["taxon_a"] == 9606

    def test_positive_rejected(self):
        line = _make_mitab_line(negative="false")
        assert parse_mitab_line(line) is None

    def test_short_line_rejected(self):
        line = "\t".join(["a", "b", "c"])  # Only 3 columns
        assert parse_mitab_line(line) is None

    def test_non_uniprot_rejected(self):
        line = _make_mitab_line(uniprot_a="chebi:12345")
        assert parse_mitab_line(line) is None

    def test_pubmed_parsed(self):
        line = _make_mitab_line(pubmed="pubmed:99999")
        result = parse_mitab_line(line)
        assert result["pubmed_id"] == 99999


class TestRunIntactEtl:
    @pytest.fixture
    def intact_data_dir(self, tmp_path):
        data_dir = tmp_path / "intact"
        data_dir.mkdir()
        # Mock intact_negative.txt with 3 lines: 2 human, 1 mouse
        lines = [
            _make_mitab_line(
                uniprot_a="uniprotkb:P00001",
                uniprot_b="uniprotkb:P00002",
                detection='psi-mi:"MI:0019"(coimmunoprecipitation)',
                interaction_id="EBI-001",
            ),
            _make_mitab_line(
                uniprot_a="uniprotkb:P00003",
                uniprot_b="uniprotkb:P00004",
                detection='psi-mi:"MI:0018"(two hybrid)',
                interaction_id="EBI-002",
            ),
            _make_mitab_line(
                uniprot_a="uniprotkb:P00005",
                uniprot_b="uniprotkb:P00006",
                taxon_a="taxid:10090(mouse)",
                taxon_b="taxid:10090(mouse)",
                interaction_id="EBI-003",
            ),
        ]
        (data_dir / "intact_negative.txt").write_text("\n".join(lines) + "\n")
        return data_dir

    def test_basic_etl(self, ppi_db, intact_data_dir):
        stats = run_intact_etl(db_path=ppi_db, data_dir=intact_data_dir)

        # 3 lines total, 2 human, 1 mouse filtered
        assert stats["lines_total"] == 3
        assert stats["lines_parsed"] == 2
        assert stats["lines_skipped_non_human"] == 1
        assert stats["pairs_gold"] == 1  # MI:0019 = gold
        assert stats["pairs_silver"] == 1  # MI:0018 = silver
        assert stats["pairs_inserted"] == 2

    def test_db_contents(self, ppi_db, intact_data_dir):
        run_intact_etl(db_path=ppi_db, data_dir=intact_data_dir)

        conn = get_connection(ppi_db)
        try:
            # 4 human proteins
            protein_count = conn.execute(
                "SELECT COUNT(*) FROM proteins"
            ).fetchone()[0]
            assert protein_count == 4

            # 2 negative results
            result_count = conn.execute(
                "SELECT COUNT(*) FROM ppi_negative_results"
            ).fetchone()[0]
            assert result_count == 2

            # Both gold and silver tiers present
            tiers = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT confidence_tier FROM ppi_negative_results"
                ).fetchall()
            }
            assert tiers == {"gold", "silver"}

            # Dataset version recorded
            dv = conn.execute(
                "SELECT name FROM dataset_versions WHERE name = 'intact_negative'"
            ).fetchone()
            assert dv is not None
        finally:
            conn.close()

    def test_human_only_false(self, ppi_db, intact_data_dir):
        """With human_only=False, mouse interactions are included."""
        stats = run_intact_etl(
            db_path=ppi_db, data_dir=intact_data_dir, human_only=False
        )
        assert stats["lines_parsed"] == 3
        assert stats["pairs_inserted"] == 3

    def test_etl_idempotent(self, ppi_db, intact_data_dir):
        """Running ETL twice should not duplicate records."""
        stats1 = run_intact_etl(db_path=ppi_db, data_dir=intact_data_dir)
        stats2 = run_intact_etl(db_path=ppi_db, data_dir=intact_data_dir)
        assert stats1["pairs_inserted"] == stats2["pairs_inserted"]
        conn = get_connection(ppi_db)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM ppi_negative_results"
            ).fetchone()[0]
            assert count == stats1["pairs_inserted"]
            dv_count = conn.execute(
                "SELECT COUNT(*) FROM dataset_versions "
                "WHERE name = 'intact_negative'"
            ).fetchone()[0]
            assert dv_count == 1
        finally:
            conn.close()

    def test_comment_lines_skipped(self, ppi_db, tmp_path):
        """Comment/header lines starting with # are counted and skipped."""
        data_dir = tmp_path / "intact_comment"
        data_dir.mkdir()
        lines = [
            "#" + "\t".join(["col" + str(i) for i in range(42)]),
            _make_mitab_line(
                uniprot_a="uniprotkb:P00001",
                uniprot_b="uniprotkb:P00002",
                interaction_id="EBI-100",
            ),
        ]
        (data_dir / "intact_negative.txt").write_text(
            "\n".join(lines) + "\n"
        )
        stats = run_intact_etl(db_path=ppi_db, data_dir=data_dir)
        assert stats["lines_total"] == 2
        assert stats["lines_skipped_comment"] == 1
        assert stats["lines_parsed"] == 1

    def test_dash_interaction_id_generates_unique(self, ppi_db, tmp_path):
        """Dash '-' in interaction_id column generates per-pair ID."""
        data_dir = tmp_path / "intact_dash"
        data_dir.mkdir()
        lines = [
            _make_mitab_line(
                uniprot_a="uniprotkb:P00001",
                uniprot_b="uniprotkb:P00002",
                interaction_id="-",
            ),
            _make_mitab_line(
                uniprot_a="uniprotkb:P00003",
                uniprot_b="uniprotkb:P00004",
                interaction_id="-",
            ),
        ]
        (data_dir / "intact_negative.txt").write_text(
            "\n".join(lines) + "\n"
        )
        stats = run_intact_etl(db_path=ppi_db, data_dir=data_dir)
        assert stats["pairs_inserted"] == 2
        # Each pair should have its own experiment record
        conn = get_connection(ppi_db)
        try:
            exp_count = conn.execute(
                "SELECT COUNT(*) FROM ppi_experiments WHERE source_db = 'intact'"
            ).fetchone()[0]
            assert exp_count == 2
        finally:
            conn.close()
