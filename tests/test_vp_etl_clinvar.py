"""Tests for ClinVar ETL module.

Tests parsing, tier assignment, conflict detection, HGVS extraction,
and ACMG criteria extraction.
"""

import gzip
import json
import textwrap
from pathlib import Path

import pytest

from negbiodb_vp.etl_clinvar import (
    _classify_evidence_type,
    _classify_tier,
    _extract_acmg_criteria,
    _normalize_classification,
    _parse_consequence,
    _parse_hgvs_from_name,
    _parse_phenotype_ids,
    _parse_phenotype_names,
    load_clinvar_data,
    parse_submission_summary,
    parse_variant_summary,
)
from negbiodb_vp.vp_db import get_connection, refresh_all_vp_pairs, run_vp_migrations

MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations_vp"


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary VP database."""
    db_path = tmp_path / "test_vp.db"
    run_vp_migrations(db_path, MIGRATIONS_DIR)
    return db_path


@pytest.fixture
def conn(tmp_db):
    c = get_connection(tmp_db)
    yield c
    c.close()


# ── Unit tests: classification normalization ──────────────────────────


class TestNormalizeClassification:
    def test_benign(self):
        assert _normalize_classification("Benign") == "benign"

    def test_likely_benign(self):
        assert _normalize_classification("Likely benign") == "likely_benign"

    def test_benign_likely_benign(self):
        assert _normalize_classification("Benign/Likely benign") == "benign/likely_benign"

    def test_pathogenic(self):
        assert _normalize_classification("Pathogenic") == "pathogenic"

    def test_unknown(self):
        assert _normalize_classification("risk factor") is None

    def test_whitespace(self):
        assert _normalize_classification("  Benign  ") == "benign"


# ── Unit tests: tier assignment ──────────────────────────────────────


class TestClassifyTier:
    def test_expert_panel_gold(self):
        assert _classify_tier("reviewed by expert panel") == "gold"

    def test_practice_guideline_gold(self):
        assert _classify_tier("practice guideline") == "gold"

    def test_multiple_submitters_silver(self):
        assert _classify_tier(
            "criteria provided, multiple submitters, no conflicts"
        ) == "silver"

    def test_single_submitter_bronze(self):
        assert _classify_tier("criteria provided, single submitter") == "bronze"

    def test_conflicting_bronze(self):
        assert _classify_tier(
            "criteria provided, conflicting classifications"
        ) == "bronze"

    def test_fallback_bronze(self):
        assert _classify_tier("no assertion criteria provided") == "bronze"


# ── Unit tests: evidence type ─────────────────────────────────────────


class TestClassifyEvidenceType:
    def test_expert_reviewed(self):
        assert _classify_evidence_type("reviewed by expert panel", 1) == "expert_reviewed"

    def test_multi_submitter_by_status(self):
        assert _classify_evidence_type(
            "criteria provided, multiple submitters, no conflicts", 2
        ) == "multi_submitter_concordant"

    def test_multi_submitter_by_count(self):
        assert _classify_evidence_type(
            "criteria provided, single submitter", 3
        ) == "multi_submitter_concordant"

    def test_single_submitter(self):
        assert _classify_evidence_type(
            "criteria provided, single submitter", 1
        ) == "single_submitter"


# ── Unit tests: consequence parsing ──────────────────────────────────


class TestParseConsequence:
    def test_missense(self):
        assert _parse_consequence("NM_000059.4(BRCA2):c.5123C>A (p.Ala1708Asp)") == "missense"

    def test_nonsense(self):
        assert _parse_consequence("stop gained variant") == "nonsense"

    def test_frameshift(self):
        assert _parse_consequence("frameshift variant") == "frameshift"

    def test_splice(self):
        assert _parse_consequence("splice_donor_variant") == "splice"

    def test_synonymous(self):
        assert _parse_consequence("synonymous variant") == "synonymous"

    def test_intronic(self):
        assert _parse_consequence("intron variant") == "intronic"

    def test_other(self):
        assert _parse_consequence("unknown variant type") == "other"


# ── Unit tests: HGVS extraction ──────────────────────────────────────


class TestParseHgvs:
    def test_full_name(self):
        hgvs_c, hgvs_p = _parse_hgvs_from_name(
            "NM_000059.4(BRCA2):c.5123C>A (p.Ala1708Asp)"
        )
        assert hgvs_c == "NM_000059.4:c.5123C>A"
        assert hgvs_p == "p.Ala1708Asp"

    def test_coding_only(self):
        hgvs_c, hgvs_p = _parse_hgvs_from_name("NM_000059.4:c.5123C>A")
        assert hgvs_c == "NM_000059.4:c.5123C>A"
        assert hgvs_p is None

    def test_empty(self):
        assert _parse_hgvs_from_name("") == (None, None)

    def test_none(self):
        assert _parse_hgvs_from_name(None) == (None, None)


# ── Unit tests: ACMG criteria extraction ─────────────────────────────


class TestExtractAcmgCriteria:
    def test_multiple_criteria(self):
        result = _extract_acmg_criteria("This variant meets BA1, BS1, and BP4 criteria")
        assert result == ["BA1", "BP4", "BS1"]

    def test_single_criterion(self):
        result = _extract_acmg_criteria("Meets BA1 standalone")
        assert result == ["BA1"]

    def test_no_criteria(self):
        assert _extract_acmg_criteria("No criteria mentioned") == []

    def test_empty(self):
        assert _extract_acmg_criteria("") == []

    def test_none(self):
        assert _extract_acmg_criteria(None) == []

    def test_pathogenic_criteria(self):
        result = _extract_acmg_criteria("PVS1 and PM2 apply")
        assert "PVS1" in result
        assert "PM2" in result


# ── Unit tests: phenotype parsing ────────────────────────────────────


class TestParsePhenotypeIds:
    def test_single_disease(self):
        result = _parse_phenotype_ids("MedGen:C0006142,OMIM:114480")
        assert len(result) == 1
        assert result[0]["medgen_cui"] == "C0006142"
        assert result[0]["omim_id"] == "114480"

    def test_multiple_diseases(self):
        result = _parse_phenotype_ids("MedGen:C0006142|MedGen:C0677776")
        assert len(result) == 2

    def test_empty(self):
        assert _parse_phenotype_ids("") == []

    def test_na(self):
        assert _parse_phenotype_ids("na") == []


class TestParsePhenotypeNames:
    def test_single(self):
        assert _parse_phenotype_names("Breast cancer") == ["Breast cancer"]

    def test_multiple(self):
        result = _parse_phenotype_names("Breast cancer|Ovarian cancer")
        assert len(result) == 2

    def test_not_provided(self):
        assert _parse_phenotype_names("not provided") == []

    def test_empty(self):
        assert _parse_phenotype_names("") == []


# ── Integration test: variant_summary parsing ────────────────────────


def _create_variant_summary_gz(tmp_path: Path) -> Path:
    """Create a minimal variant_summary.txt.gz for testing."""
    header = (
        "#AlleleID\tType\tName\tGeneID\tGeneSymbol\tHGNC_ID\t"
        "ClinicalSignificance\tClinSigSimple\tLastEvaluated\t"
        "RS# (dbSNP)\tnsv/esv (dbVar)\tRCVaccession\tPhenotypeIDS\t"
        "PhenotypeList\tOrigin\tOriginSimple\tAssembly\t"
        "ChromosomeAccession\tChromosome\tStart\tStop\t"
        "ReferenceAllele\tAlternateAllele\tCytogenetic\t"
        "ReviewStatus\tNumberSubmitters\tGuidelines\t"
        "TestedInGTR\tOtherIDs\tSubmitterCategories\tVariationID\t"
        "PositionVCF\tReferenceAlleleVCF\tAlternateAlleleVCF"
    )

    row1 = (
        "1234\tsingle nucleotide variant\t"
        "NM_000059.4(BRCA2):c.5123C>A (p.Ala1708Asp)\t"
        "675\tBRCA2\tHGNC:1101\t"
        "Benign\t0\t2020-01-15\t"
        "12345\t-\tRCV000001\t"
        "MedGen:C0006142,OMIM:114480\t"
        "Breast cancer\tgermline\tgermline\tGRCh38\t"
        "NC_000017.11\t17\t43092919\t43092919\t"
        "A\tG\t17q21.31\t"
        "reviewed by expert panel\t3\t-\t"
        "N\t-\t1\t100001\t"
        "43092919\tA\tG"
    )

    row2 = (
        "5678\tsingle nucleotide variant\t"
        "NM_000059.4(BRCA2):c.200A>T (p.Lys67Met)\t"
        "675\tBRCA2\tHGNC:1101\t"
        "Likely benign\t0\t2022-06-01\t"
        "67890\t-\tRCV000002\t"
        "MedGen:C0006142\t"
        "Breast cancer\tgermline\tgermline\tGRCh38\t"
        "NC_000017.11\t17\t43100000\t43100000\t"
        "C\tT\t17q21.31\t"
        "criteria provided, single submitter\t1\t-\t"
        "N\t-\t1\t100002\t"
        "43100000\tC\tT"
    )

    # Pathogenic variant for conflict detection
    row3 = (
        "9999\tsingle nucleotide variant\t"
        "NM_000059.4(BRCA2):c.5123C>A (p.Ala1708Asp)\t"
        "675\tBRCA2\tHGNC:1101\t"
        "Pathogenic\t1\t2021-03-10\t"
        "12345\t-\tRCV000003\t"
        "MedGen:C0006142\t"
        "Breast cancer\tgermline\tgermline\tGRCh38\t"
        "NC_000017.11\t17\t43092919\t43092919\t"
        "A\tG\t17q21.31\t"
        "criteria provided, single submitter\t1\t-\t"
        "N\t-\t1\t100001\t"
        "43092919\tA\tG"
    )

    # GRCh37 row (should be filtered out)
    row4 = (
        "4444\tsingle nucleotide variant\t"
        "NM_000059.4:c.100G>A\t"
        "675\tBRCA2\tHGNC:1101\t"
        "Benign\t0\t2019-01-01\t"
        "-1\t-\tRCV000004\t"
        "MedGen:C0006142\t"
        "Breast cancer\tgermline\tgermline\tGRCh37\t"
        "NC_000017.10\t17\t41244000\t41244000\t"
        "G\tA\t17q21.31\t"
        "criteria provided, single submitter\t1\t-\t"
        "N\t-\t1\t100003\t"
        "41244000\tG\tA"
    )

    content = "\n".join([header, row1, row2, row3, row4]) + "\n"
    gz_path = tmp_path / "variant_summary.txt.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write(content)
    return gz_path


def _create_submission_summary_gz(tmp_path: Path) -> Path:
    """Create a minimal submission_summary.txt.gz for testing."""
    header = (
        "#VariationID\tClinicalSignificance\tDateLastEvaluated\t"
        "Description\tReviewStatus\tCollectionMethod\t"
        "OriginCounts\tSubmitter\tSCV\tSubmittedPhenotypeInfo\t"
        "ReportedPhenotypeInfo\tClinVarAccession\tExplanationOfInterpretation"
    )

    sub1 = (
        "100001\tBenign\t2020-01-15\t"
        "This variant meets BA1 and BS1 criteria.\t"
        "reviewed by expert panel\tresearch\t"
        "germline:1\tLab A\tSCV000001.1\t"
        "Breast cancer\tC0006142:Breast cancer\tSCV000001.1\t"
        "Benign based on BA1 BS1 BP4"
    )

    sub2 = (
        "100001\tBenign\t2021-05-20\t"
        "Population frequency supports benign.\t"
        "criteria provided, single submitter\tclinical testing\t"
        "germline:1\tLab B\tSCV000002.2\t"
        "Breast cancer\tC0006142:Breast cancer\tSCV000002.2\t"
        "Benign"
    )

    sub3 = (
        "100002\tLikely benign\t2022-06-01\t"
        "Low conservation, common in population.\t"
        "criteria provided, single submitter\tclinical testing\t"
        "germline:1\tLab C\tSCV000003.1\t"
        "Breast cancer\tC0006142:Breast cancer\tSCV000003.1\t"
        "Likely benign"
    )

    content = "\n".join([header, sub1, sub2, sub3]) + "\n"
    gz_path = tmp_path / "submission_summary.txt.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write(content)
    return gz_path


class TestParseVariantSummary:
    def test_parse_benign_records(self, tmp_path):
        gz = _create_variant_summary_gz(tmp_path)
        benign, pathogenic = parse_variant_summary(gz)
        # 2 GRCh38 benign rows (row1 + row2), row4 is GRCh37 → filtered
        assert len(benign) == 2

    def test_parse_pathogenic_records(self, tmp_path):
        gz = _create_variant_summary_gz(tmp_path)
        benign, pathogenic = parse_variant_summary(gz)
        assert len(pathogenic) == 1
        assert pathogenic[0]["variation_id"] == 100001

    def test_grch37_filtered(self, tmp_path):
        gz = _create_variant_summary_gz(tmp_path)
        benign, pathogenic = parse_variant_summary(gz)
        # No GRCh37 records should appear
        all_chroms = [r["chromosome"] for r in benign + pathogenic]
        # All should be chr17 GRCh38
        assert all(c == "17" for c in all_chroms)

    def test_gold_tier(self, tmp_path):
        gz = _create_variant_summary_gz(tmp_path)
        benign, _ = parse_variant_summary(gz)
        gold = [r for r in benign if r["confidence_tier"] == "gold"]
        assert len(gold) == 1
        assert gold[0]["evidence_type"] == "expert_reviewed"

    def test_bronze_tier(self, tmp_path):
        gz = _create_variant_summary_gz(tmp_path)
        benign, _ = parse_variant_summary(gz)
        bronze = [r for r in benign if r["confidence_tier"] == "bronze"]
        assert len(bronze) == 1


class TestParseSubmissionSummary:
    def test_parse_submissions(self, tmp_path):
        gz = _create_submission_summary_gz(tmp_path)
        subs = parse_submission_summary(gz)
        assert 100001 in subs
        assert len(subs[100001]) == 2  # Two submissions for VID 100001

    def test_filter_by_variation_ids(self, tmp_path):
        gz = _create_submission_summary_gz(tmp_path)
        subs = parse_submission_summary(gz, variation_ids={100001})
        assert 100001 in subs
        assert 100002 not in subs

    def test_acmg_extraction(self, tmp_path):
        gz = _create_submission_summary_gz(tmp_path)
        subs = parse_submission_summary(gz)
        # First submission for VID 100001 should have BA1, BS1, BP4
        sub1 = subs[100001][0]
        assert "BA1" in sub1["acmg_criteria"]
        assert "BS1" in sub1["acmg_criteria"]
        assert "BP4" in sub1["acmg_criteria"]

    def test_scv_version_stripped(self, tmp_path):
        gz = _create_submission_summary_gz(tmp_path)
        subs = parse_submission_summary(gz)
        for vid_subs in subs.values():
            for s in vid_subs:
                if s["scv_accession"]:
                    assert "." not in s["scv_accession"]


# ── Integration test: full load pipeline ─────────────────────────────


class TestLoadClinvarData:
    def test_full_pipeline(self, conn, tmp_path):
        """End-to-end: parse + load + aggregate pairs."""
        vs_gz = _create_variant_summary_gz(tmp_path)
        ss_gz = _create_submission_summary_gz(tmp_path)

        benign, pathogenic = parse_variant_summary(vs_gz)
        all_vids = {r["variation_id"] for r in benign + pathogenic}
        subs = parse_submission_summary(ss_gz, variation_ids=all_vids)

        stats = load_clinvar_data(conn, benign, pathogenic, subs)
        conn.commit()

        assert stats["genes_inserted"] > 0
        assert stats["variants_inserted"] > 0
        assert stats["diseases_inserted"] > 0
        assert stats["negative_results_inserted"] > 0

    def test_conflict_detection(self, conn, tmp_path):
        """Variant with both benign and pathogenic submissions → has_conflict=1."""
        vs_gz = _create_variant_summary_gz(tmp_path)
        ss_gz = _create_submission_summary_gz(tmp_path)

        benign, pathogenic = parse_variant_summary(vs_gz)
        subs = parse_submission_summary(ss_gz)
        stats = load_clinvar_data(conn, benign, pathogenic, subs)
        conn.commit()

        assert stats["conflicts_detected"] > 0

        # Check has_conflict flag in DB
        conflict_rows = conn.execute(
            "SELECT COUNT(*) FROM vp_negative_results WHERE has_conflict = 1"
        ).fetchone()[0]
        assert conflict_rows > 0

    def test_submission_level_conflict_detection(self, conn, tmp_path):
        """P/LP submission phenotype should trigger conflict even without pathogenic aggregate."""
        vs_gz = _create_variant_summary_gz(tmp_path)
        benign, _ = parse_variant_summary(vs_gz)

        header = (
            "#VariationID\tClinicalSignificance\tDateLastEvaluated\t"
            "Description\tReviewStatus\tCollectionMethod\t"
            "OriginCounts\tSubmitter\tSCV\tSubmittedPhenotypeInfo\t"
            "ReportedPhenotypeInfo\tClinVarAccession\tExplanationOfInterpretation"
        )
        benign_sub = (
            "100001\tBenign\t2020-01-15\t"
            "Population frequency supports benign.\t"
            "reviewed by expert panel\tresearch\t"
            "germline:1\tLab A\tSCV000101.1\t"
            "Breast cancer\tC0006142:Breast cancer\tSCV000101.1\t"
            "Benign based on BA1"
        )
        pathogenic_sub = (
            "100001\tLikely pathogenic\t2021-01-01\t"
            "Segregation and functional evidence.\t"
            "criteria provided, single submitter\tclinical testing\t"
            "germline:1\tLab P\tSCV000102.1\t"
            "Breast cancer\tC0006142:Breast cancer\tSCV000102.1\t"
            "Likely pathogenic"
        )
        ss_gz = tmp_path / "submission_conflict.txt.gz"
        with gzip.open(ss_gz, "wt") as f:
            f.write("\n".join([header, benign_sub, pathogenic_sub]) + "\n")

        subs = parse_submission_summary(ss_gz)
        stats = load_clinvar_data(conn, benign, [], subs)
        conn.commit()

        assert stats["conflicts_detected"] > 0
        conflict_rows = conn.execute(
            "SELECT COUNT(*) FROM vp_negative_results WHERE has_conflict = 1"
        ).fetchone()[0]
        assert conflict_rows > 0

    def test_pair_aggregation(self, conn, tmp_path):
        """Aggregated pairs should be created after refresh."""
        vs_gz = _create_variant_summary_gz(tmp_path)
        ss_gz = _create_submission_summary_gz(tmp_path)

        benign, pathogenic = parse_variant_summary(vs_gz)
        subs = parse_submission_summary(ss_gz)
        load_clinvar_data(conn, benign, pathogenic, subs)
        conn.commit()

        count = refresh_all_vp_pairs(conn)
        conn.commit()
        assert count > 0

        # Check degrees are set
        row = conn.execute(
            "SELECT variant_degree, disease_degree FROM variant_disease_pairs LIMIT 1"
        ).fetchone()
        assert row[0] is not None
        assert row[1] is not None
