"""Tests for computational score ETL module."""

from pathlib import Path

import pytest

from negbiodb_vp.etl_scores import annotate_scores
from negbiodb_vp.vp_db import get_connection, run_vp_migrations

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_vp"


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test_vp.db"
    run_vp_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    c = get_connection(tmp_db)
    yield c
    c.close()


def _seed_data(conn):
    conn.execute("INSERT INTO genes (gene_id, gene_symbol) VALUES (1, 'TP53')")
    conn.execute(
        """INSERT INTO variants (variant_id, chromosome, position, ref_allele, alt_allele,
         variant_type, gene_id, consequence_type)
        VALUES (1, '17', 7579472, 'G', 'A', 'single nucleotide variant', 1, 'missense')"""
    )
    conn.execute(
        """INSERT INTO variants (variant_id, chromosome, position, ref_allele, alt_allele,
         variant_type, gene_id, consequence_type)
        VALUES (2, '17', 7579500, 'C', 'T', 'single nucleotide variant', 1, 'synonymous')"""
    )
    conn.commit()


class TestAnnotateScores:
    def test_annotates_missense(self, conn, tmp_path):
        _seed_data(conn)
        tsv = tmp_path / "scores.tsv"
        tsv.write_text(
            "chromosome\tposition\tref\talt\tcadd_phred\trevel_score\t"
            "alphamissense_score\talphamissense_class\tphylop_score\t"
            "gerp_score\tsift_score\tpolyphen2_score\n"
            "17\t7579472\tG\tA\t28.5\t0.92\t0.85\tlikely_pathogenic\t"
            "5.2\t4.8\t0.001\t0.995\n"
        )
        stats = annotate_scores(conn, tsv)
        assert stats["variants_annotated"] == 1

        row = conn.execute(
            """SELECT cadd_phred, revel_score, alphamissense_score,
                      alphamissense_class, phylop_score, gerp_score,
                      sift_score, polyphen2_score
            FROM variants WHERE variant_id = 1"""
        ).fetchone()
        assert row[0] == pytest.approx(28.5)
        assert row[1] == pytest.approx(0.92)
        assert row[2] == pytest.approx(0.85)
        assert row[3] == "likely_pathogenic"
        assert row[4] == pytest.approx(5.2)
        assert row[5] == pytest.approx(4.8)
        assert row[6] == pytest.approx(0.001)
        assert row[7] == pytest.approx(0.995)

    def test_null_for_non_missense(self, conn, tmp_path):
        _seed_data(conn)
        tsv = tmp_path / "scores.tsv"
        tsv.write_text(
            "chromosome\tposition\tref\talt\tcadd_phred\trevel_score\t"
            "alphamissense_score\talphamissense_class\tphylop_score\t"
            "gerp_score\tsift_score\tpolyphen2_score\n"
            "17\t7579500\tC\tT\t12.3\t\t\t\t3.1\t2.5\t\t\n"
        )
        stats = annotate_scores(conn, tsv)
        assert stats["variants_annotated"] == 1

        row = conn.execute(
            "SELECT cadd_phred, revel_score, alphamissense_score FROM variants WHERE variant_id = 2"
        ).fetchone()
        assert row[0] == pytest.approx(12.3)
        assert row[1] is None  # REVEL: missense only
        assert row[2] is None  # AlphaMissense: missense only

    def test_not_found(self, conn, tmp_path):
        _seed_data(conn)
        tsv = tmp_path / "scores.tsv"
        tsv.write_text(
            "chromosome\tposition\tref\talt\tcadd_phred\trevel_score\t"
            "alphamissense_score\talphamissense_class\tphylop_score\t"
            "gerp_score\tsift_score\tpolyphen2_score\n"
            "1\t999999\tA\tG\t10.0\t0.5\t0.3\tambiguous\t1.0\t1.0\t0.1\t0.5\n"
        )
        stats = annotate_scores(conn, tsv)
        assert stats["variants_not_found"] == 1
        assert stats["variants_annotated"] == 0
