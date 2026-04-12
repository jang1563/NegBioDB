"""Standardization utilities for MD domain.

Handles:
  1. Metabolite name → InChIKey/PubChem CID:
       a. Exact HMDB synonym lookup (internal cache, CC BY-NC 4.0 — not redistributed)
       b. PubChem name search (redistributable)
       c. rapidfuzz fuzzy match against HMDB synonyms (fallback)

  2. Disease term → MONDO/MeSH:
       a. MONDO OWL exact label match (CC0)
       b. rapidfuzz fuzzy match against MONDO labels
       c. Manual mapping for top-50 disease terms

  3. ClassyFire API lookup for metabolite class (CC BY 4.0)

Results are cached in the MD database to avoid redundant API calls.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_CLASSYFIRE_BASE = "https://classyfire.wishartlab.com"
_REQUEST_TIMEOUT = 20
_RETRY_DELAY = 1

# ── Manual disease mapping for common metabolomics study terms ───────────────
# Covers the most frequent disease labels seen in MetaboLights/NMDR to avoid
# API dependency for high-frequency lookups.
MANUAL_DISEASE_MAP: dict[str, dict[str, str]] = {
    "type 2 diabetes": {"name": "type 2 diabetes mellitus", "mondo_id": "MONDO:0005148", "disease_category": "metabolic"},
    "t2d": {"name": "type 2 diabetes mellitus", "mondo_id": "MONDO:0005148", "disease_category": "metabolic"},
    "type 2 diabetes mellitus": {"name": "type 2 diabetes mellitus", "mondo_id": "MONDO:0005148", "disease_category": "metabolic"},
    "type 1 diabetes": {"name": "type 1 diabetes mellitus", "mondo_id": "MONDO:0005147", "disease_category": "metabolic"},
    "obesity": {"name": "obesity", "mondo_id": "MONDO:0011122", "disease_category": "metabolic"},
    "colorectal cancer": {"name": "colorectal cancer", "mondo_id": "MONDO:0005575", "disease_category": "cancer"},
    "breast cancer": {"name": "breast cancer", "mondo_id": "MONDO:0007254", "disease_category": "cancer"},
    "lung cancer": {"name": "lung cancer", "mondo_id": "MONDO:0008903", "disease_category": "cancer"},
    "prostate cancer": {"name": "prostate cancer", "mondo_id": "MONDO:0008315", "disease_category": "cancer"},
    "hepatocellular carcinoma": {"name": "hepatocellular carcinoma", "mondo_id": "MONDO:0007256", "disease_category": "cancer"},
    "alzheimer's disease": {"name": "Alzheimer disease", "mondo_id": "MONDO:0004975", "disease_category": "neurological"},
    "alzheimer disease": {"name": "Alzheimer disease", "mondo_id": "MONDO:0004975", "disease_category": "neurological"},
    "parkinson's disease": {"name": "Parkinson disease", "mondo_id": "MONDO:0005180", "disease_category": "neurological"},
    "parkinson disease": {"name": "Parkinson disease", "mondo_id": "MONDO:0005180", "disease_category": "neurological"},
    "cardiovascular disease": {"name": "cardiovascular disease", "mondo_id": "MONDO:0004995", "disease_category": "cardiovascular"},
    "coronary artery disease": {"name": "coronary artery disease", "mondo_id": "MONDO:0005010", "disease_category": "cardiovascular"},
    "hypertension": {"name": "hypertensive disorder", "mondo_id": "MONDO:0004995", "disease_category": "cardiovascular"},
    "non-alcoholic fatty liver disease": {"name": "non-alcoholic fatty liver disease", "mondo_id": "MONDO:0016264", "disease_category": "metabolic"},
    "nafld": {"name": "non-alcoholic fatty liver disease", "mondo_id": "MONDO:0016264", "disease_category": "metabolic"},
    "chronic kidney disease": {"name": "chronic kidney disease", "mondo_id": "MONDO:0005300", "disease_category": "metabolic"},
    "multiple sclerosis": {"name": "multiple sclerosis", "mondo_id": "MONDO:0005301", "disease_category": "neurological"},
    "rheumatoid arthritis": {"name": "rheumatoid arthritis", "mondo_id": "MONDO:0008383", "disease_category": "other"},
    "inflammatory bowel disease": {"name": "inflammatory bowel disease", "mondo_id": "MONDO:0005265", "disease_category": "other"},
    "crohn's disease": {"name": "Crohn disease", "mondo_id": "MONDO:0005265", "disease_category": "other"},
    "ulcerative colitis": {"name": "ulcerative colitis", "mondo_id": "MONDO:0005101", "disease_category": "other"},
    "depression": {"name": "major depressive disorder", "mondo_id": "MONDO:0002050", "disease_category": "neurological"},
    "schizophrenia": {"name": "schizophrenia", "mondo_id": "MONDO:0005090", "disease_category": "neurological"},
    "metabolic syndrome": {"name": "metabolic syndrome", "mondo_id": "MONDO:0024491", "disease_category": "metabolic"},
    "sepsis": {"name": "sepsis", "mondo_id": "MONDO:0021881", "disease_category": "other"},
    "covid-19": {"name": "COVID-19", "mondo_id": "MONDO:0100096", "disease_category": "other"},
}


class Standardizer:
    """Handles metabolite and disease standardization with caching.

    Uses HMDB internal cache for name resolution (CC BY-NC — internal only),
    PubChem for redistributable identifiers, and MONDO for disease terms.
    """

    def __init__(
        self,
        hmdb_cache_conn=None,
        pubchem_delay: float = 0.2,
        fuzzy_threshold: float = 0.85,
    ):
        """
        Args:
            hmdb_cache_conn: sqlite3 connection to hmdb_cache.db (internal only).
                             If None, HMDB lookup is skipped (PubChem only).
            pubchem_delay:   Seconds between PubChem API calls (rate limit).
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0-1).
        """
        self.hmdb_cache = hmdb_cache_conn
        self.pubchem_delay = pubchem_delay
        self.fuzzy_threshold = fuzzy_threshold
        self._pubchem_last_call = 0.0

        # In-memory caches to reduce DB round-trips
        self._metabolite_cache: dict[str, int | None] = {}   # name → metabolite_id
        self._disease_cache: dict[str, int | None] = {}       # name → disease_id

    # ── Metabolite standardization ───────────────────────────────────────────

    def resolve_metabolite(self, name: str) -> dict | None:
        """Resolve a metabolite name to standardized identifiers.

        Resolution order:
          1. HMDB internal cache (exact synonym match)
          2. PubChem name search (primary redistributable source)
          3. HMDB fuzzy match (if rapidfuzz available)

        Returns dict with inchikey, pubchem_cid, smiles, metabolite_class
        or None if unresolvable.
        """
        if not name or not name.strip():
            return None

        # 1. HMDB exact synonym (internal cache only)
        if self.hmdb_cache is not None:
            from negbiodb_md.etl_hmdb import lookup_by_name
            hit = lookup_by_name(name, self.hmdb_cache)
            if hit and hit.get("inchikey"):
                return {
                    "inchikey": hit["inchikey"],
                    "pubchem_cid": hit["pubchem_cid"],
                    "canonical_smiles": hit.get("smiles"),
                    "metabolite_class": hit.get("classyfire_superclass"),
                    "metabolite_subclass": hit.get("classyfire_class"),
                }

        # 2. PubChem name search
        result = self._pubchem_lookup(name)
        if result:
            return result

        # 3. Fuzzy HMDB fallback
        if self.hmdb_cache is not None:
            result = self._hmdb_fuzzy(name)
            if result:
                return result

        return None

    def _pubchem_lookup(self, name: str) -> dict | None:
        """Look up a compound by name via PubChem PUG REST API."""
        # Rate limiting
        elapsed = time.time() - self._pubchem_last_call
        if elapsed < self.pubchem_delay:
            time.sleep(self.pubchem_delay - elapsed)

        try:
            url = f"{_PUBCHEM_BASE}/compound/name/{requests.utils.quote(name)}/property/InChIKey,CanonicalSMILES,MolecularFormula,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount/JSON"
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            self._pubchem_last_call = time.time()
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [{}])[0]
            inchikey = props.get("InChIKey")
            if not inchikey:
                return None
            return {
                "inchikey": inchikey,
                "pubchem_cid": props.get("CID"),
                "canonical_smiles": props.get("CanonicalSMILES"),
                "formula": props.get("MolecularFormula"),
                "molecular_weight": props.get("MolecularWeight"),
                "logp": props.get("XLogP"),
                "tpsa": props.get("TPSA"),
                "hbd": props.get("HBondDonorCount"),
                "hba": props.get("HBondAcceptorCount"),
                "metabolite_class": None,
                "metabolite_subclass": None,
            }
        except requests.RequestException as exc:
            logger.debug("PubChem lookup failed for '%s': %s", name, exc)
            self._pubchem_last_call = time.time()
            return None

    def _hmdb_fuzzy(self, name: str) -> dict | None:
        """Fuzzy match name against HMDB synonyms using rapidfuzz."""
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            return None

        rows = self.hmdb_cache.execute(
            "SELECT synonym, hmdb_id FROM hmdb_synonyms LIMIT 100000"
        ).fetchall()
        synonyms = [r[0] for r in rows]
        hmdb_ids = {r[0]: r[1] for r in rows}

        match = process.extractOne(
            name.lower(), synonyms, scorer=fuzz.ratio, score_cutoff=self.fuzzy_threshold * 100
        )
        if match is None:
            return None

        matched_syn, score, _ = match
        hmdb_id = hmdb_ids[matched_syn]
        from negbiodb_md.etl_hmdb import lookup_by_name
        hit = lookup_by_name(matched_syn, self.hmdb_cache)
        if hit and hit.get("inchikey"):
            return {
                "inchikey": hit["inchikey"],
                "pubchem_cid": hit["pubchem_cid"],
                "canonical_smiles": hit.get("smiles"),
                "metabolite_class": hit.get("classyfire_superclass"),
                "metabolite_subclass": hit.get("classyfire_class"),
            }
        return None

    def get_or_create_metabolite(self, conn, name: str) -> int | None:
        """Get or insert metabolite into md_metabolites. Returns metabolite_id or None."""
        if name in self._metabolite_cache:
            return self._metabolite_cache[name]

        # Check if already in DB by name
        row = conn.execute(
            "SELECT metabolite_id FROM md_metabolites WHERE name = ? LIMIT 1", (name,)
        ).fetchone()
        if row:
            self._metabolite_cache[name] = row[0]
            return row[0]

        resolved = self.resolve_metabolite(name)
        if resolved and resolved.get("inchikey"):
            # Check by InChIKey (may already exist from another name)
            row = conn.execute(
                "SELECT metabolite_id FROM md_metabolites WHERE inchikey = ? LIMIT 1",
                (resolved["inchikey"],),
            ).fetchone()
            if row:
                self._metabolite_cache[name] = row[0]
                return row[0]

        # Insert new metabolite
        r = resolved or {}
        conn.execute(
            """INSERT OR IGNORE INTO md_metabolites
               (name, pubchem_cid, inchikey, canonical_smiles, formula,
                metabolite_class, metabolite_subclass,
                molecular_weight, logp, tpsa, hbd, hba)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                name,
                r.get("pubchem_cid"),
                r.get("inchikey"),
                r.get("canonical_smiles"),
                r.get("formula"),
                r.get("metabolite_class"),
                r.get("metabolite_subclass"),
                r.get("molecular_weight"),
                r.get("logp"),
                r.get("tpsa"),
                r.get("hbd"),
                r.get("hba"),
            ),
        )
        row = conn.execute(
            "SELECT metabolite_id FROM md_metabolites WHERE name = ? LIMIT 1", (name,)
        ).fetchone()
        mid = row[0] if row else None
        self._metabolite_cache[name] = mid
        return mid

    # ── Disease standardization ──────────────────────────────────────────────

    def resolve_disease(self, terms: list[str]) -> dict | None:
        """Resolve a list of disease term candidates to standardized MONDO entry.

        Resolution order:
          1. Manual mapping (high-confidence, covers top-50 diseases)
          2. Fuzzy match against manual map keys

        Returns dict with name, mondo_id, disease_category or None.
        """
        for term in terms:
            if not term:
                continue
            term_lower = term.lower().strip()

            # Exact manual match
            if term_lower in MANUAL_DISEASE_MAP:
                return MANUAL_DISEASE_MAP[term_lower]

        # Fuzzy match against manual map keys
        try:
            from rapidfuzz import process, fuzz
            all_terms = [t.lower().strip() for t in terms if t]
            best_match = None
            best_score = 0.0
            for term in all_terms:
                result = process.extractOne(
                    term, list(MANUAL_DISEASE_MAP.keys()),
                    scorer=fuzz.token_sort_ratio, score_cutoff=75
                )
                if result and result[1] > best_score:
                    best_score = result[1]
                    best_match = result[0]
            if best_match:
                return MANUAL_DISEASE_MAP[best_match]
        except ImportError:
            pass

        # Fallback: create a minimal disease entry from the longest term
        best_term = max(terms, key=lambda t: len(t) if t else 0, default=None)
        if best_term and len(best_term) > 3:
            return {
                "name": best_term.strip(),
                "mondo_id": None,
                "disease_category": "other",
            }
        return None

    def get_or_create_disease(self, conn, terms: list[str]) -> int | None:
        """Get or insert disease into md_diseases. Returns disease_id or None."""
        cache_key = "|".join(sorted(t.lower() for t in terms if t))
        if cache_key in self._disease_cache:
            return self._disease_cache[cache_key]

        resolved = self.resolve_disease(terms)
        if resolved is None:
            self._disease_cache[cache_key] = None
            return None

        name = resolved["name"]

        # Check if already in DB
        row = conn.execute(
            "SELECT disease_id FROM md_diseases WHERE name = ? LIMIT 1", (name,)
        ).fetchone()
        if row:
            self._disease_cache[cache_key] = row[0]
            return row[0]

        # Insert
        conn.execute(
            """INSERT OR IGNORE INTO md_diseases
               (name, mondo_id, disease_category)
               VALUES (?,?,?)""",
            (name, resolved.get("mondo_id"), resolved.get("disease_category", "other")),
        )
        row = conn.execute(
            "SELECT disease_id FROM md_diseases WHERE name = ? LIMIT 1", (name,)
        ).fetchone()
        did = row[0] if row else None
        self._disease_cache[cache_key] = did
        return did

    # ── ClassyFire lookup ────────────────────────────────────────────────────

    @staticmethod
    def get_classyfire_class(inchikey: str) -> tuple[str | None, str | None]:
        """Look up ClassyFire superclass and class for an InChIKey.

        Returns (superclass, class) tuple. Both may be None on failure.
        ClassyFire is CC BY 4.0 — results are redistributable.
        """
        try:
            url = f"{_CLASSYFIRE_BASE}/entities/{inchikey}.json"
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 404:
                return None, None
            resp.raise_for_status()
            data = resp.json()
            superclass = (data.get("superclass") or {}).get("name")
            cls = (data.get("class") or {}).get("name")
            return superclass, cls
        except requests.RequestException:
            return None, None

    def enrich_metabolite_classes(self, conn) -> int:
        """Fill NULL metabolite_class values using ClassyFire API.

        Returns number of metabolites updated.
        """
        rows = conn.execute(
            """SELECT metabolite_id, inchikey FROM md_metabolites
               WHERE metabolite_class IS NULL AND inchikey IS NOT NULL"""
        ).fetchall()

        updated = 0
        for mid, inchikey in rows:
            superclass, cls = self.get_classyfire_class(inchikey)
            if superclass:
                conn.execute(
                    """UPDATE md_metabolites
                       SET metabolite_class = ?, metabolite_subclass = ?
                       WHERE metabolite_id = ?""",
                    (superclass, cls, mid),
                )
                updated += 1
            time.sleep(0.1)  # ClassyFire rate limit

        conn.commit()
        logger.info("ClassyFire: updated %d metabolite classes", updated)
        return updated
