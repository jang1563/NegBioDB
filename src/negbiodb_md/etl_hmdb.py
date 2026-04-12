"""ETL module for HMDB (Human Metabolome Database) — internal standardization cache.

LICENSE NOTE: HMDB data is CC BY-NC 4.0 (non-commercial). This module builds an
INTERNAL standardization cache only. HMDB-derived data is NEVER exported as part
of the benchmark. All redistributed metabolite identifiers come from PubChem.

Parses:
  - hmdb_metabolites.xml (full HMDB XML download, ~1 GB)
  - Populates: internal hmdb_cache.db for name → HMDB ID → InChIKey mapping

Usage:
  python scripts_md/01_download_hmdb.py  # downloads XML
  python -c "from negbiodb_md.etl_hmdb import build_hmdb_cache; build_hmdb_cache()"
"""

from __future__ import annotations

import gzip
import logging
import sqlite3
from pathlib import Path
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_HMDB_XML = _PROJECT_ROOT / "data" / "hmdb_metabolites.xml.gz"
DEFAULT_CACHE_DB = _PROJECT_ROOT / "data" / "hmdb_cache.db"

# HMDB XML namespace
HMDB_NS = "http://www.hmdb.ca"


def _ns(tag: str) -> str:
    return f"{{{HMDB_NS}}}{tag}"


# ── Cache database ──────────────────────────────────────────────────────────

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS hmdb_metabolites (
    hmdb_id         TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    inchikey        TEXT,
    smiles          TEXT,
    formula         TEXT,
    pubchem_cid     INTEGER,
    chebi_id        TEXT,
    kegg_id         TEXT,
    classyfire_superclass TEXT,
    classyfire_class TEXT
);
CREATE TABLE IF NOT EXISTS hmdb_synonyms (
    synonym TEXT NOT NULL,
    hmdb_id TEXT NOT NULL REFERENCES hmdb_metabolites(hmdb_id),
    PRIMARY KEY (synonym, hmdb_id)
);
CREATE INDEX IF NOT EXISTS idx_syn_hmdb ON hmdb_synonyms(hmdb_id);
CREATE INDEX IF NOT EXISTS idx_hmdb_inchikey ON hmdb_metabolites(inchikey) WHERE inchikey IS NOT NULL;
"""


def get_cache_connection(cache_db: str | Path | None = None) -> sqlite3.Connection:
    """Return a sqlite3 connection to the HMDB cache (creates schema if needed)."""
    if cache_db is None:
        cache_db = DEFAULT_CACHE_DB
    cache_db = Path(cache_db)
    cache_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_db)
    conn.executescript(CACHE_SCHEMA)
    conn.commit()
    return conn


# ── XML parsing ─────────────────────────────────────────────────────────────

def _text(elem, tag: str) -> str | None:
    """Safe child element text extraction."""
    child = elem.find(_ns(tag))
    if child is not None and child.text:
        return child.text.strip() or None
    return None


def _int_or_none(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def parse_hmdb_xml(
    xml_path: str | Path | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Parse HMDB XML and return list of metabolite dicts.

    Each dict has keys:
        hmdb_id, name, inchikey, smiles, formula, pubchem_cid,
        chebi_id, kegg_id, synonyms (list[str])

    Args:
        xml_path: Path to hmdb_metabolites.xml or .xml.gz. Defaults to data dir.
        limit:    Stop after this many metabolites (for testing).
    """
    if xml_path is None:
        xml_path = DEFAULT_HMDB_XML
    xml_path = Path(xml_path)

    open_fn = gzip.open if xml_path.suffix == ".gz" else open
    records: list[dict] = []
    count = 0

    logger.info("Parsing HMDB XML: %s", xml_path)
    with open_fn(xml_path, "rb") as fh:
        for event, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag != _ns("metabolite"):
                continue

            hmdb_id = _text(elem, "accession")
            if not hmdb_id:
                elem.clear()
                continue

            name = _text(elem, "name") or hmdb_id
            inchikey = _text(elem, "inchikey")
            smiles = _text(elem, "smiles")
            formula = _text(elem, "chemical_formula")

            # External IDs
            pubchem_cid = _int_or_none(_text(elem, "pubchem_compound_id"))
            chebi_id = _text(elem, "chebi_id")
            kegg_id = _text(elem, "kegg_id")

            # Synonyms
            synonyms: list[str] = [name]
            syn_el = elem.find(_ns("synonyms"))
            if syn_el is not None:
                for s in syn_el.findall(_ns("synonym")):
                    if s.text and s.text.strip():
                        synonyms.append(s.text.strip())

            # ClassyFire taxonomy
            cf = elem.find(_ns("taxonomy"))
            superclass = None
            cf_class = None
            if cf is not None:
                superclass = _text(cf, "super_class")
                cf_class = _text(cf, "class")

            records.append({
                "hmdb_id": hmdb_id,
                "name": name,
                "inchikey": inchikey,
                "smiles": smiles,
                "formula": formula,
                "pubchem_cid": pubchem_cid,
                "chebi_id": chebi_id,
                "kegg_id": kegg_id,
                "classyfire_superclass": superclass,
                "classyfire_class": cf_class,
                "synonyms": synonyms,
            })
            elem.clear()  # free memory
            count += 1

            if limit and count >= limit:
                break

    logger.info("Parsed %d HMDB metabolites", len(records))
    return records


# ── Cache builder ────────────────────────────────────────────────────────────

def build_hmdb_cache(
    xml_path: str | Path | None = None,
    cache_db: str | Path | None = None,
    limit: int | None = None,
) -> int:
    """Parse HMDB XML and build the internal standardization cache.

    Returns number of metabolites inserted.
    """
    records = parse_hmdb_xml(xml_path, limit=limit)
    conn = get_cache_connection(cache_db)

    inserted = 0
    with conn:
        for rec in records:
            conn.execute(
                """INSERT OR REPLACE INTO hmdb_metabolites
                    (hmdb_id, name, inchikey, smiles, formula,
                     pubchem_cid, chebi_id, kegg_id,
                     classyfire_superclass, classyfire_class)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec["hmdb_id"], rec["name"], rec["inchikey"],
                    rec["smiles"], rec["formula"], rec["pubchem_cid"],
                    rec["chebi_id"], rec["kegg_id"],
                    rec["classyfire_superclass"], rec["classyfire_class"],
                ),
            )
            for syn in set(rec["synonyms"]):
                if syn:
                    conn.execute(
                        "INSERT OR IGNORE INTO hmdb_synonyms (synonym, hmdb_id) VALUES (?,?)",
                        (syn.lower(), rec["hmdb_id"]),
                    )
            inserted += 1

    logger.info("HMDB cache: %d metabolites in %s", inserted, cache_db or DEFAULT_CACHE_DB)
    return inserted


# ── Lookup helpers (used by etl_standardize.py) ──────────────────────────────

def lookup_by_name(name: str, conn: sqlite3.Connection) -> dict | None:
    """Look up a metabolite by exact or synonym name (case-insensitive).

    Returns dict with hmdb_id, inchikey, smiles, pubchem_cid or None.
    """
    row = conn.execute(
        """SELECT m.hmdb_id, m.inchikey, m.smiles, m.pubchem_cid,
                  m.classyfire_superclass, m.classyfire_class
           FROM hmdb_synonyms s
           JOIN hmdb_metabolites m ON s.hmdb_id = m.hmdb_id
           WHERE s.synonym = ?
           LIMIT 1""",
        (name.lower().strip(),),
    ).fetchone()
    if row is None:
        return None
    cols = ["hmdb_id", "inchikey", "smiles", "pubchem_cid",
            "classyfire_superclass", "classyfire_class"]
    return dict(zip(cols, row))


def lookup_by_inchikey(inchikey: str, conn: sqlite3.Connection) -> dict | None:
    """Look up a metabolite by InChIKey."""
    row = conn.execute(
        """SELECT hmdb_id, inchikey, smiles, pubchem_cid,
                  classyfire_superclass, classyfire_class
           FROM hmdb_metabolites WHERE inchikey = ? LIMIT 1""",
        (inchikey,),
    ).fetchone()
    if row is None:
        return None
    cols = ["hmdb_id", "inchikey", "smiles", "pubchem_cid",
            "classyfire_superclass", "classyfire_class"]
    return dict(zip(cols, row))
