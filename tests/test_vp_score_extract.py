"""Tests for VP HPC score extraction helpers."""

import csv
import gzip
import zipfile
from pathlib import Path

import pytest

from negbiodb_vp.etl_scores import annotate_scores
from negbiodb_vp.score_extract import export_score_targets, extract_scores_for_targets
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


class TestScoreExtract:
    def test_export_and_extract_scores(self, conn, tmp_path):
        _seed_data(conn)

        targets_tsv = tmp_path / "vp_score_targets.tsv"
        export_stats = export_score_targets(conn, targets_tsv)
        assert export_stats["targets_exported"] == 2

        cadd_tsv = tmp_path / "whole_genome_SNVs.tsv"
        cadd_tsv.write_text(
            "#Chrom\tPos\tRef\tAlt\tPHRED\tSIFTval\tPolyPhenVal\tPhyloP100way_vertebrate\tGerpRS\n"
            "17\t7579472\tG\tA\t28.5\t0.001\t0.995\t5.2\t4.8\n"
            "17\t7579500\tC\tT\t12.3\t\t\t3.1\t2.5\n"
        )

        revel_zip = tmp_path / "revel.zip"
        revel_inner = "revel.tsv"
        with zipfile.ZipFile(revel_zip, "w") as zf:
            zf.writestr(
                revel_inner,
                "chr\tgrch38_pos\tref\talt\tREVEL\n"
                "17\t7579472\tG\tA\t0.92\n",
            )

        alphamissense_gz = tmp_path / "AlphaMissense_hg38.tsv.gz"
        with gzip.open(alphamissense_gz, "wt") as f:
            f.write(
                "CHROM\tPOS\tREF\tALT\tgenome\tam_pathogenicity\tam_class\n"
                "17\t7579472\tG\tA\thg38\t0.85\tlikely_pathogenic\n"
                "17\t7579472\tG\tA\thg19\t0.12\tlikely_benign\n"
            )

        merged_tsv = tmp_path / "merged_scores.tsv"
        extract_stats = extract_scores_for_targets(
            targets_tsv=targets_tsv,
            output_tsv=merged_tsv,
            cadd_tsv=cadd_tsv,
            revel_tsv=revel_zip,
            alphamissense_tsv=alphamissense_gz,
        )

        assert extract_stats["targets_loaded"] == 2
        assert extract_stats["cadd_matches"] == 2
        assert extract_stats["revel_matches"] == 1
        assert extract_stats["alphamissense_matches"] == 1
        assert extract_stats["rows_written"] == 2

        with open(merged_tsv) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert len(rows) == 2
        missense = next(r for r in rows if r["position"] == "7579472")
        assert missense["cadd_phred"] == "28.5"
        assert missense["revel_score"] == "0.92"
        assert missense["alphamissense_score"] == "0.85"
        assert missense["alphamissense_class"] == "likely_pathogenic"

        annotate_stats = annotate_scores(conn, merged_tsv)
        assert annotate_stats["variants_annotated"] == 2

        row1 = conn.execute(
            """SELECT cadd_phred, revel_score, alphamissense_score, alphamissense_class
               FROM variants WHERE variant_id = 1"""
        ).fetchone()
        assert row1[0] == pytest.approx(28.5)
        assert row1[1] == pytest.approx(0.92)
        assert row1[2] == pytest.approx(0.85)
        assert row1[3] == "likely_pathogenic"

        row2 = conn.execute(
            """SELECT cadd_phred, revel_score, alphamissense_score, phylop_score, gerp_score
               FROM variants WHERE variant_id = 2"""
        ).fetchone()
        assert row2[0] == pytest.approx(12.3)
        assert row2[1] is None
        assert row2[2] is None
        assert row2[3] == pytest.approx(3.1)
        assert row2[4] == pytest.approx(2.5)
