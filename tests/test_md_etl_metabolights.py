"""Tests for MetaboLights ETL module."""

import pytest
from negbiodb_md.etl_metabolights import (
    is_human_disease_study,
    detect_platform,
    detect_biofluid,
    extract_disease_terms,
    _fetch_maf_file,
)


# ── is_human_disease_study ────────────────────────────────────────────────────

def test_is_human_disease_study_positive():
    details = {
        "organisms": ["homo sapiens"],
        "title": "Metabolomics of type 2 diabetes patients",
        "description": "Case-control study of healthy controls vs T2D",
        "factors": ["type 2 diabetes"],
    }
    assert is_human_disease_study(details) is True


def test_is_human_disease_study_non_human():
    details = {
        "organisms": ["mus musculus"],
        "title": "Mouse model of diabetes",
        "description": "Mice were used in this study",
        "factors": ["diabetes"],
    }
    assert is_human_disease_study(details) is False


def test_is_human_disease_study_no_disease_keywords():
    details = {
        "organisms": ["homo sapiens"],
        "title": "NMR metabolomics of exercise",
        "description": "Healthy volunteers performed exercise",
        "factors": [],
    }
    assert is_human_disease_study(details) is False


def test_is_human_disease_study_cancer():
    details = {
        "organisms": ["human"],
        "title": "Serum metabolomics in colorectal cancer",
        "description": "Comparison of cancer patients vs healthy controls",
        "factors": ["colorectal cancer"],
    }
    assert is_human_disease_study(details) is True


def test_is_human_disease_study_none():
    assert is_human_disease_study(None) is False


# ── detect_platform ───────────────────────────────────────────────────────────

def test_detect_platform_nmr():
    assert detect_platform("NMR-based metabolomics", "") == "nmr"


def test_detect_platform_lcms():
    assert detect_platform("LC-MS analysis", "") == "lc_ms"


def test_detect_platform_gcms():
    assert detect_platform("", "Gas chromatography mass spectrometry analysis") == "gc_ms"


def test_detect_platform_unknown():
    assert detect_platform("Metabolite profiling study", "") == "other"


# ── detect_biofluid ───────────────────────────────────────────────────────────

def test_detect_biofluid_serum():
    assert detect_biofluid("Serum metabolomics study", "") == "blood"


def test_detect_biofluid_urine():
    assert detect_biofluid("", "Urinary metabolomics analysis") == "urine"


def test_detect_biofluid_csf():
    assert detect_biofluid("CSF metabolomics in Alzheimer's", "") == "csf"


def test_detect_biofluid_unknown():
    assert detect_biofluid("Metabolomics study", "") == "other"


# ── extract_disease_terms ─────────────────────────────────────────────────────

def test_extract_disease_terms_factors():
    terms = extract_disease_terms("Metabolomics study", "", ["type 2 diabetes", "obesity"])
    assert "type 2 diabetes" in terms
    assert "obesity" in terms


def test_extract_disease_terms_title():
    terms = extract_disease_terms("Metabolomics of Alzheimer's disease patients", "", [])
    assert len(terms) > 0


# ── _fetch_maf_file (unit test with mock response) ────────────────────────────

def test_fetch_maf_file_no_pval_cols(monkeypatch):
    """MAF without p-value columns should return empty list."""
    import requests

    class MockResponse:
        status_code = 200
        text = "metabolite_identification\tfold_change\nglucose\t2.5\nleucine\t0.8\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    rows = _fetch_maf_file("http://mock.url/maf.tsv", "MTBLS1", "test.tsv")
    assert rows == []


def test_fetch_maf_file_with_pval(monkeypatch):
    """MAF with p-value columns should return rows."""
    import requests

    class MockResponse:
        status_code = 200
        text = (
            "metabolite_identification\tfolr_change\tp_value\tfdr\n"
            "glucose\t2.5\t0.03\t0.08\n"
            "leucine\t0.8\t0.45\t0.62\n"
        )

        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **kw: MockResponse())
    rows = _fetch_maf_file("http://mock.url/maf.tsv", "MTBLS1", "test.tsv")
    assert len(rows) == 2
    assert rows[0]["p_value"] == 0.03
    assert rows[1]["fdr"] == pytest.approx(0.62)
