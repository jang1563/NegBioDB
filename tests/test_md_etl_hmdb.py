"""Tests for HMDB ETL module (internal cache — CC BY-NC 4.0, not redistributed)."""

import gzip
import sqlite3
from pathlib import Path

import pytest

from negbiodb_md.etl_hmdb import (
    build_hmdb_cache,
    get_cache_connection,
    lookup_by_name,
    lookup_by_inchikey,
    parse_hmdb_xml,
    _ns,
    HMDB_NS,
)


# ── Minimal HMDB XML fixture ─────────────────────────────────────────────────

_HMDB_XML_FRAGMENT = """\
<?xml version="1.0" encoding="UTF-8"?>
<hmdb xmlns="http://www.hmdb.ca">
  <metabolite>
    <accession>HMDB0000122</accession>
    <name>Glucose</name>
    <inchikey>WQZGKKKJIJFFOK-GASJEMHNSA-N</inchikey>
    <smiles>OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O</smiles>
    <chemical_formula>C6H12O6</chemical_formula>
    <pubchem_compound_id>5793</pubchem_compound_id>
    <chebi_id>4167</chebi_id>
    <kegg_id>C00031</kegg_id>
    <synonyms>
      <synonym>D-Glucose</synonym>
      <synonym>Blood sugar</synonym>
    </synonyms>
    <taxonomy>
      <super_class>Carbohydrates and Carbohydrate Conjugates</super_class>
      <class>Hexoses</class>
    </taxonomy>
  </metabolite>
  <metabolite>
    <accession>HMDB0000148</accession>
    <name>Alanine</name>
    <inchikey>QNAYBMKLOCPYGJ-REOHCLBHSA-N</inchikey>
    <smiles>C[C@@H](N)C(O)=O</smiles>
    <chemical_formula>C3H7NO2</chemical_formula>
    <pubchem_compound_id>5950</pubchem_compound_id>
    <taxonomy>
      <super_class>Amino Acids and Analogues</super_class>
      <class>L-Amino Acids</class>
    </taxonomy>
  </metabolite>
</hmdb>
"""


@pytest.fixture
def hmdb_xml_gz(tmp_path) -> Path:
    gz_path = tmp_path / "hmdb_metabolites.xml.gz"
    with gzip.open(gz_path, "wb") as f:
        f.write(_HMDB_XML_FRAGMENT.encode())
    return gz_path


@pytest.fixture
def hmdb_cache(tmp_path, hmdb_xml_gz) -> sqlite3.Connection:
    cache_db = tmp_path / "hmdb_cache.db"
    build_hmdb_cache(hmdb_xml_gz, cache_db)
    return get_cache_connection(cache_db)


# ── parse_hmdb_xml ────────────────────────────────────────────────────────────

def test_parse_hmdb_xml_returns_metabolites(hmdb_xml_gz):
    records = parse_hmdb_xml(hmdb_xml_gz)
    assert len(records) == 2


def test_parse_hmdb_xml_fields(hmdb_xml_gz):
    records = parse_hmdb_xml(hmdb_xml_gz)
    glucose = next(r for r in records if r["hmdb_id"] == "HMDB0000122")
    assert glucose["name"] == "Glucose"
    assert glucose["inchikey"] == "WQZGKKKJIJFFOK-GASJEMHNSA-N"
    assert glucose["pubchem_cid"] == 5793
    assert "D-Glucose" in glucose["synonyms"]
    assert "Blood sugar" in glucose["synonyms"]
    assert glucose["classyfire_superclass"] == "Carbohydrates and Carbohydrate Conjugates"


def test_parse_hmdb_xml_limit(hmdb_xml_gz):
    records = parse_hmdb_xml(hmdb_xml_gz, limit=1)
    assert len(records) == 1


# ── build_hmdb_cache ──────────────────────────────────────────────────────────

def test_build_hmdb_cache_inserts_metabolites(tmp_path, hmdb_xml_gz):
    cache_db = tmp_path / "cache.db"
    n = build_hmdb_cache(hmdb_xml_gz, cache_db)
    assert n == 2

    conn = get_cache_connection(cache_db)
    count = conn.execute("SELECT COUNT(*) FROM hmdb_metabolites").fetchone()[0]
    assert count == 2
    conn.close()


def test_build_hmdb_cache_inserts_synonyms(tmp_path, hmdb_xml_gz):
    cache_db = tmp_path / "cache.db"
    build_hmdb_cache(hmdb_xml_gz, cache_db)
    conn = get_cache_connection(cache_db)
    syns = conn.execute(
        "SELECT synonym FROM hmdb_synonyms WHERE hmdb_id = 'HMDB0000122'"
    ).fetchall()
    syn_set = {s[0] for s in syns}
    assert "glucose" in syn_set  # lowercase
    assert "d-glucose" in syn_set
    conn.close()


# ── lookup helpers ────────────────────────────────────────────────────────────

def test_lookup_by_name_exact(hmdb_cache):
    result = lookup_by_name("glucose", hmdb_cache)
    assert result is not None
    assert result["hmdb_id"] == "HMDB0000122"
    assert result["inchikey"] == "WQZGKKKJIJFFOK-GASJEMHNSA-N"


def test_lookup_by_name_synonym(hmdb_cache):
    result = lookup_by_name("D-Glucose", hmdb_cache)
    assert result is not None
    assert result["hmdb_id"] == "HMDB0000122"


def test_lookup_by_name_missing(hmdb_cache):
    result = lookup_by_name("xyznonexistent", hmdb_cache)
    assert result is None


def test_lookup_by_inchikey(hmdb_cache):
    result = lookup_by_inchikey("WQZGKKKJIJFFOK-GASJEMHNSA-N", hmdb_cache)
    assert result is not None
    assert result["hmdb_id"] == "HMDB0000122"


def test_lookup_by_inchikey_missing(hmdb_cache):
    result = lookup_by_inchikey("AAAAAAAAAAAAA-BBBBBBBB-C", hmdb_cache)
    assert result is None
