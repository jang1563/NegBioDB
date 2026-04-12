"""Tests for NMDR (Metabolomics Workbench) ETL module."""

import pytest

from negbiodb_md.etl_nmdr import (
    list_public_studies,
    fetch_study_details,
    fetch_results,
    is_human_disease_study,
    detect_platform,
    detect_biofluid,
    _parse_pmid,
    _to_float,
)


# ── _parse_pmid ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("12345678", 12345678),
    (12345678, 12345678),
    ("  98765  ", 98765),
    ("", None),
    (None, None),
    ("not-a-pmid", None),
])
def test_parse_pmid(raw, expected):
    assert _parse_pmid(raw) == expected


# ── _to_float ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("val,expected", [
    ("0.05", 0.05),
    (0.001, 0.001),
    ("1.5e-3", 1.5e-3),
    (None, None),
    ("not_a_float", None),
    ("", None),
])
def test_to_float(val, expected):
    result = _to_float(val)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# ── list_public_studies (mocked) ──────────────────────────────────────────────

def test_list_public_studies_dict_format(monkeypatch):
    """NMDR returns nested dict format {"study": {"ST000001": {...}, ...}}."""
    import negbiodb_md.etl_nmdr as nmdr_mod

    def mock_get(endpoint):
        return {
            "study": {
                "ST000001": {"study_title": "Diabetes metabolomics", "subject_species": "Homo sapiens"},
                "ST000002": {"study_title": "Cancer study", "subject_species": "Homo sapiens"},
            }
        }

    monkeypatch.setattr(nmdr_mod, "_get", mock_get)
    studies = list_public_studies()
    assert len(studies) == 2
    ids = {s["study_id"] for s in studies}
    assert "ST000001" in ids
    assert "ST000002" in ids


def test_list_public_studies_list_format(monkeypatch):
    """NMDR can also return a list directly."""
    import negbiodb_md.etl_nmdr as nmdr_mod

    def mock_get(endpoint):
        return [
            {"study_id": "ST000003", "study_title": "List study"},
        ]

    monkeypatch.setattr(nmdr_mod, "_get", mock_get)
    studies = list_public_studies()
    assert len(studies) == 1
    assert studies[0]["study_id"] == "ST000003"


def test_list_public_studies_api_failure(monkeypatch):
    """When API returns None (failure), return empty list."""
    import negbiodb_md.etl_nmdr as nmdr_mod

    monkeypatch.setattr(nmdr_mod, "_get", lambda e: None)
    studies = list_public_studies()
    assert studies == []


# ── fetch_study_details (mocked) ──────────────────────────────────────────────

def test_fetch_study_details_returns_normalized_dict(monkeypatch):
    import negbiodb_md.etl_nmdr as nmdr_mod

    def mock_get(endpoint):
        return {
            "study": {
                "ST000001": {
                    "study_title": "Serum metabolomics in T2D patients",
                    "study_summary": "Case-control study",
                    "subject_species": "Homo sapiens",
                    "analysis_type": "LC-MS",
                    "sample_source": "Serum",
                    "pubmed_id": "12345678",
                }
            }
        }

    monkeypatch.setattr(nmdr_mod, "_get", mock_get)
    details = fetch_study_details("ST000001")
    assert details is not None
    assert details["study_id"] == "ST000001"
    assert "t2d" in details["title"].lower() or "metabolomics" in details["title"].lower()
    assert details["organism"] == "homo sapiens"
    assert details["analysis_type"] == "lc-ms"
    assert details["sample_source"] == "serum"
    assert details["pmid"] == 12345678


def test_fetch_study_details_returns_none_on_failure(monkeypatch):
    import negbiodb_md.etl_nmdr as nmdr_mod

    monkeypatch.setattr(nmdr_mod, "_get", lambda e: None)
    assert fetch_study_details("ST000001") is None


# ── is_human_disease_study ────────────────────────────────────────────────────

def test_is_human_disease_study_positive():
    details = {
        "organism": "homo sapiens",
        "title": "Metabolomics of type 2 diabetes patients",
        "description": "Case-control study vs healthy controls",
        "factors": ["type 2 diabetes"],
    }
    assert is_human_disease_study(details) is True


def test_is_human_disease_study_mouse():
    details = {
        "organism": "mus musculus",
        "title": "Mouse model of obesity",
        "description": "Diet-induced obesity study",
        "factors": ["obesity"],
    }
    assert is_human_disease_study(details) is False


def test_is_human_disease_study_no_disease_terms():
    details = {
        "organism": "homo sapiens",
        "title": "Exercise metabolomics in healthy volunteers",
        "description": "NMR metabolomics after aerobic exercise",
        "factors": [],
    }
    assert is_human_disease_study(details) is False


def test_is_human_disease_study_cancer():
    details = {
        "organism": "human",
        "title": "LC-MS metabolomics in colorectal cancer patients",
        "description": "Tumor vs adjacent normal tissue comparison",
        "factors": ["colorectal cancer"],
    }
    assert is_human_disease_study(details) is True


def test_is_human_disease_study_none():
    assert is_human_disease_study(None) is False


def test_is_human_disease_study_empty_dict():
    assert is_human_disease_study({}) is False


# ── detect_platform ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("analysis_type,expected", [
    ({"analysis_type": "lc-ms"}, "lc_ms"),
    ({"analysis_type": "LC/MS"}, "lc_ms"),
    ({"analysis_type": "nmr"}, "nmr"),
    ({"analysis_type": "gc-ms"}, "gc_ms"),
    ({"analysis_type": "uplc"}, "lc_ms"),
    ({"analysis_type": "unknown method"}, "other"),
    ({}, "other"),
])
def test_detect_platform(analysis_type, expected):
    assert detect_platform(analysis_type) == expected


# ── detect_biofluid ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("details,expected", [
    ({"sample_source": "serum"}, "blood"),
    ({"sample_source": "plasma"}, "blood"),
    ({"sample_source": "urine"}, "urine"),
    ({"sample_source": "cerebrospinal fluid"}, "csf"),
    ({"sample_source": "csf"}, "csf"),
    ({"sample_source": "tissue biopsy"}, "tissue"),
    ({"sample_source": "feces"}, "other"),
    ({"sample_source": "unknown biofluid"}, "other"),
    ({}, "other"),
])
def test_detect_biofluid(details, expected):
    assert detect_biofluid(details) == expected


# ── fetch_results (mocked) ────────────────────────────────────────────────────

def test_fetch_results_returns_rows_with_stats(monkeypatch):
    """fetch_results should return rows that have p-value or FDR."""
    import negbiodb_md.etl_nmdr as nmdr_mod

    def mock_get(endpoint):
        return {
            "metabolites": [
                {"metabolite_name": "glucose", "p_value": "0.003", "fdr": "0.01"},
                {"metabolite_name": "alanine", "p_value": "0.35", "fdr": "0.55"},
                {"metabolite_name": "leucine", "p_value": None, "fdr": None},  # no stats
            ]
        }

    monkeypatch.setattr(nmdr_mod, "_get", mock_get)
    rows = fetch_results("ST000001")
    # Only rows WITH statistics (p_value or fdr) should be returned
    assert len(rows) == 2
    names = [r["metabolite_name"] for r in rows]
    assert "glucose" in names
    assert "alanine" in names
    assert "leucine" not in names


def test_fetch_results_empty_on_no_stats(monkeypatch):
    """fetch_results should return [] if no rows have statistics."""
    import negbiodb_md.etl_nmdr as nmdr_mod

    def mock_get(endpoint):
        return {
            "metabolites": [
                {"metabolite_name": "glucose"},
                {"name": "alanine"},
            ]
        }

    monkeypatch.setattr(nmdr_mod, "_get", mock_get)
    rows = fetch_results("ST000001")
    assert rows == []


def test_fetch_results_parses_p_values(monkeypatch):
    """p_value and fdr should be parsed to float."""
    import negbiodb_md.etl_nmdr as nmdr_mod

    def mock_get(endpoint):
        return {
            "metabolites": [
                {"metabolite_name": "pyruvate", "p_value": "0.003", "fdr": "0.015", "fold_change": "1.8"},
            ]
        }

    monkeypatch.setattr(nmdr_mod, "_get", mock_get)
    rows = fetch_results("ST000001")
    assert len(rows) == 1
    assert rows[0]["p_value"] == pytest.approx(0.003)
    assert rows[0]["fdr"] == pytest.approx(0.015)
    assert rows[0]["fold_change"] == pytest.approx(1.8)


def test_fetch_results_handles_dict_metabolites(monkeypatch):
    """NMDR may return metabolites as a dict keyed by metabolite ID."""
    import negbiodb_md.etl_nmdr as nmdr_mod

    def mock_get(endpoint):
        return {
            "metabolites": {
                "1": {"metabolite_name": "urea", "p_value": "0.42"},
                "2": {"metabolite_name": "creatinine", "fdr": "0.07"},
            }
        }

    monkeypatch.setattr(nmdr_mod, "_get", mock_get)
    rows = fetch_results("ST000001")
    assert len(rows) == 2


def test_fetch_results_api_failure_returns_empty(monkeypatch):
    import negbiodb_md.etl_nmdr as nmdr_mod

    monkeypatch.setattr(nmdr_mod, "_get", lambda e: None)
    assert fetch_results("ST000001") == []
