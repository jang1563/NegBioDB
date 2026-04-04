"""Tests for gnomAD ETL module."""

from pathlib import Path

import pytest

from negbiodb_vp.etl_gnomad import (
    annotate_variant_frequencies,
    export_gnomad_site_tsvs,
    generate_copper_tier,
    load_gene_constraints,
    merge_gnomad_tsv_shards,
)
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


def _seed_genes(conn):
    """Insert test genes."""
    conn.execute("INSERT INTO genes (gene_id, entrez_id, gene_symbol) VALUES (1, 7157, 'TP53')")
    conn.execute("INSERT INTO genes (gene_id, entrez_id, gene_symbol) VALUES (2, 672, 'BRCA1')")
    conn.execute("INSERT INTO genes (gene_id, entrez_id, gene_symbol) VALUES (3, 675, 'BRCA2')")
    conn.commit()


def _seed_variants(conn):
    """Insert test variants."""
    _seed_genes(conn)
    conn.execute(
        """INSERT INTO variants (variant_id, chromosome, position, ref_allele, alt_allele,
         variant_type, gene_id)
        VALUES (1, '17', 7579472, 'G', 'A', 'single nucleotide variant', 1)"""
    )
    conn.execute(
        """INSERT INTO variants (variant_id, chromosome, position, ref_allele, alt_allele,
         variant_type, gene_id)
        VALUES (2, '17', 43092919, 'A', 'G', 'single nucleotide variant', 2)"""
    )
    conn.commit()


class TestLoadGeneConstraints:
    def test_updates_scores(self, conn, tmp_path):
        _seed_genes(conn)
        tsv = tmp_path / "constraint.tsv"
        tsv.write_text(
            "gene\tpLI\toe_lof_upper\tmis_z\n"
            "TP53\t0.99\t0.12\t3.5\n"
            "BRCA1\t0.85\t0.25\t2.1\n"
            "UNKNOWN\t0.5\t0.5\t1.0\n"
        )
        stats = load_gene_constraints(conn, tsv)
        assert stats["genes_updated"] == 2
        assert stats["genes_not_found"] == 1

        row = conn.execute(
            "SELECT pli_score, loeuf_score, missense_z FROM genes WHERE gene_id = 1"
        ).fetchone()
        assert row[0] == pytest.approx(0.99)
        assert row[1] == pytest.approx(0.12)
        assert row[2] == pytest.approx(3.5)

    def test_handles_na(self, conn, tmp_path):
        _seed_genes(conn)
        tsv = tmp_path / "constraint.tsv"
        tsv.write_text("gene\tpLI\toe_lof_upper\tmis_z\nTP53\tNA\t0.12\tnan\n")
        stats = load_gene_constraints(conn, tsv)
        assert stats["genes_updated"] == 1

        row = conn.execute(
            "SELECT pli_score, loeuf_score, missense_z FROM genes WHERE gene_id = 1"
        ).fetchone()
        assert row[0] is None  # NA → NULL
        assert row[1] == pytest.approx(0.12)
        assert row[2] is None  # nan → NULL

    def test_supports_gnomad_v41_columns(self, conn, tmp_path):
        _seed_genes(conn)
        tsv = tmp_path / "constraint_v41.tsv"
        tsv.write_text(
            "gene\tgene_id\ttranscript\tcanonical\tmane_select\tlof.pLI\tlof.oe_ci.upper\tmis.z_score\n"
            "TP53\t7157\tENST00000269305\tfalse\tfalse\t0.11\t0.91\t1.2\n"
            "TP53\t7157\tNM_000546.6\ttrue\ttrue\t0.99\t0.12\t3.5\n"
            "BRCA1\t672\tNM_007294.4\ttrue\ttrue\t0.85\t0.25\t2.1\n"
        )
        stats = load_gene_constraints(conn, tsv)
        assert stats["genes_updated"] == 2
        assert stats["genes_not_found"] == 0

        row = conn.execute(
            "SELECT pli_score, loeuf_score, missense_z FROM genes WHERE gene_id = 1"
        ).fetchone()
        assert row[0] == pytest.approx(0.99)
        assert row[1] == pytest.approx(0.12)
        assert row[2] == pytest.approx(3.5)


class TestAnnotateVariantFrequencies:
    def test_annotates_existing_variants(self, conn, tmp_path):
        _seed_variants(conn)
        tsv = tmp_path / "freq.tsv"
        tsv.write_text(
            "chromosome\tposition\tref\talt\taf_global\taf_afr\taf_amr\t"
            "af_asj\taf_eas\taf_fin\taf_nfe\taf_sas\taf_oth\n"
            "17\t7579472\tG\tA\t0.001\t0.002\t0.001\t0.0005\t0.001\t"
            "0.001\t0.001\t0.001\t0.001\n"
            "17\t99999999\tC\tT\t0.5\t0.5\t0.5\t0.5\t0.5\t0.5\t0.5\t0.5\t0.5\n"
        )
        stats = annotate_variant_frequencies(conn, tsv)
        assert stats["variants_annotated"] == 1
        assert stats["variants_not_found"] == 1

        row = conn.execute(
            "SELECT gnomad_af_global, gnomad_af_afr FROM variants WHERE variant_id = 1"
        ).fetchone()
        assert row[0] == pytest.approx(0.001)
        assert row[1] == pytest.approx(0.002)


class TestGenerateCopperTier:
    def test_inserts_copper_variants(self, conn, tmp_path):
        _seed_variants(conn)
        # Add disease
        conn.execute(
            "INSERT INTO diseases (disease_id, medgen_cui, canonical_name) VALUES (1, 'C0006142', 'Breast cancer')"
        )
        conn.commit()

        tsv = tmp_path / "copper.tsv"
        tsv.write_text(
            "chromosome\tposition\tref\talt\taf_global\tconsequence\tgene_symbol\n"
            "1\t100000\tA\tG\t0.05\tmissense\tFOO\n"
            "17\t7579472\tG\tA\t0.02\tmissense\tTP53\n"  # Already in DB
        )
        stats = generate_copper_tier(conn, tsv, af_threshold=0.01)
        assert stats["variants_inserted"] == 1
        assert stats["results_inserted"] == 1
        assert stats["skipped_already_in_db"] == 1

        # Check copper result exists
        row = conn.execute(
            "SELECT confidence_tier, source_db FROM vp_negative_results WHERE source_db = 'gnomad'"
        ).fetchone()
        assert row[0] == "copper"
        assert row[1] == "gnomad"

    def test_creates_not_provided_disease(self, conn, tmp_path):
        _seed_variants(conn)
        tsv = tmp_path / "copper.tsv"
        tsv.write_text("chromosome\tposition\tref\talt\taf_global\tconsequence\tgene_symbol\n")
        generate_copper_tier(conn, tsv)

        # "not provided" disease should exist
        row = conn.execute(
            "SELECT canonical_name FROM diseases WHERE medgen_cui = 'CN169374'"
        ).fetchone()
        assert row[0] == "not provided"


class TestExportGnomadSites:
    def test_exports_and_loads_frequency_and_copper_tsvs(self, conn, tmp_path):
        _seed_variants(conn)

        vcf = tmp_path / "gnomad_sites.vcf"
        vcf.write_text(
            "##fileformat=VCFv4.2\n"
            '##INFO=<ID=vep,Number=.,Type=String,Description="Consequence annotations from Ensembl VEP. Format: Allele|Consequence|SYMBOL">\n'
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "17\t7579472\t.\tG\tA,T\t.\tPASS\tAF=0.001,0.005;AF_afr=0.002,0.001;AF_nfe=0.0005,0.001;vep=A|missense_variant|TP53,T|synonymous_variant|TP53\n"
            "1\t100000\t.\tA\tG\t.\tPASS\tAF=0.050;AF_remaining=0.040;vep=G|missense_variant|FOO\n"
        )

        freq_tsv = tmp_path / "variant_frequencies.tsv"
        copper_tsv = tmp_path / "copper_variants.tsv"
        stats = export_gnomad_site_tsvs(
            conn,
            [vcf],
            frequencies_out=freq_tsv,
            copper_out=copper_tsv,
            af_threshold=0.01,
        )

        assert stats["frequency_rows_written"] == 1
        assert stats["copper_rows_written"] == 1

        freq_stats = annotate_variant_frequencies(conn, freq_tsv)
        assert freq_stats["variants_annotated"] == 1

        row = conn.execute(
            "SELECT gnomad_af_global, gnomad_af_afr, gnomad_af_nfe FROM variants WHERE variant_id = 1"
        ).fetchone()
        assert row[0] == pytest.approx(0.001)
        assert row[1] == pytest.approx(0.002)
        assert row[2] == pytest.approx(0.0005)

        copper_stats = generate_copper_tier(conn, copper_tsv, af_threshold=0.01)
        assert copper_stats["variants_inserted"] == 1
        assert copper_stats["results_inserted"] == 1

        copper_row = conn.execute(
            """SELECT chromosome, position, ref_allele, alt_allele, gnomad_af_global,
                      consequence_type
               FROM variants
               WHERE chromosome = '1' AND position = 100000 AND ref_allele = 'A' AND alt_allele = 'G'"""
        ).fetchone()
        assert copper_row[0] == "1"
        assert copper_row[1] == 100000
        assert copper_row[2] == "A"
        assert copper_row[3] == "G"
        assert copper_row[4] == pytest.approx(0.05)
        assert copper_row[5] == "missense"


class TestMergeGnomadShards:
    def test_merges_and_deduplicates_shards(self, tmp_path):
        shard1 = tmp_path / "variant_frequencies.chr1.tsv"
        shard1.write_text(
            "chromosome\tposition\tref\talt\taf_global\taf_afr\n"
            "1\t100\tA\tG\t0.01\t0.02\n"
            "1\t101\tC\tT\t0.02\t0.03\n"
        )
        shard2 = tmp_path / "variant_frequencies.chr2.tsv"
        shard2.write_text(
            "chromosome\tposition\tref\talt\taf_global\taf_afr\n"
            "1\t101\tC\tT\t0.02\t0.03\n"
            "2\t200\tG\tA\t0.04\t0.05\n"
        )

        merged = tmp_path / "variant_frequencies.tsv"
        stats = merge_gnomad_tsv_shards([shard2, shard1], merged)

        assert stats["shards_read"] == 2
        assert stats["rows_read"] == 4
        assert stats["rows_written"] == 3
        assert stats["duplicate_rows_skipped"] == 1
        assert merged.read_text().splitlines() == [
            "chromosome\tposition\tref\talt\taf_global\taf_afr",
            "1\t100\tA\tG\t0.01\t0.02",
            "1\t101\tC\tT\t0.02\t0.03",
            "2\t200\tG\tA\t0.04\t0.05",
        ]
