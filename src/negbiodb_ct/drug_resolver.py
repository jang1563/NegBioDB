"""Drug name resolution cascade for NegBioDB-CT.

Resolves intervention names to ChEMBL IDs, SMILES, InChIKey, and targets
via a 4-step cascade:

  Step 1: ChEMBL molecule_synonyms exact match (in-memory index)
  Step 2: PubChem REST API name lookup (rate-limited, cached)
  Step 3: Fuzzy match via rapidfuzz (Jaro-Winkler > threshold)
  Step 4: Manual override CSV (high-frequency biologics)

After resolution, queries ChEMBL drug_mechanism for target mapping
(UniProt accessions) and inserts into intervention_targets.
"""

import json
import logging
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from negbiodb_ct.ct_db import (
    get_connection,
    DEFAULT_CT_DB_PATH,
)
from negbiodb.download import load_config

logger = logging.getLogger(__name__)

BATCH_SIZE = 500

# Pre-processing patterns to strip dosage/salt info
_DOSAGE_PAT = re.compile(r"\s*[\(\[]\s*\d+\s*(?:mg|ug|µg|ml|g|mcg|iu|%)\b[^\)\]]*[\)\]]", re.IGNORECASE)
_SALT_PAT = re.compile(
    r"\s+(?:hydrochloride|hcl|sodium|potassium|acetate|sulfate|"
    r"mesylate|maleate|fumarate|tartrate|citrate|besylate|tosylate|"
    r"dihydrochloride|monohydrate|succinate|phosphate|nitrate|"
    r"chloride|bromide|calcium|magnesium|disodium|dipotassium)\s*$",
    re.IGNORECASE,
)
_TRAILING_PAREN = re.compile(r"\s*\(.*\)\s*$")

# Patterns for non-drug names (placebos, controls, generic terms)
_NON_DRUG_PATTERNS = re.compile(
    r"\bplacebo\b|\bsaline\b|\bsugar\s*pill\b|\bsham\b"
    r"|\bstandard\s+(?:of\s+)?care\b|\bstandard\s+treatment\b"
    r"|\bno\s+intervention\b|\bobservation\s+only\b"
    r"|\bwatchful\s+wait\b|\bbest\s+supportive\b"
    r"|\bvehicle\s+(?:cream|gel|ointment|solution|patch|tablet|capsule)\b"
    r"|\bblood\s+sample\b|\bblood\s+draw\b",
    re.IGNORECASE,
)


def is_non_drug_name(name: str) -> bool:
    """Check if an intervention name is a placebo, control, or non-drug term.

    Returns True for names that should NOT be sent to drug resolution APIs.
    """
    if not name:
        return False
    return _NON_DRUG_PATTERNS.search(name) is not None


def clean_drug_name(name: str) -> str:
    """Clean intervention name for matching.

    Strips dosage info, salt suffixes, trailing parentheticals, lowercases.
    """
    if not name:
        return ""
    cleaned = name.strip()
    cleaned = _DOSAGE_PAT.sub("", cleaned)
    cleaned = _TRAILING_PAREN.sub("", cleaned)
    cleaned = _SALT_PAT.sub("", cleaned)
    cleaned = cleaned.strip().lower()
    return cleaned


# ============================================================
# STEP 1: CHEMBL SYNONYM INDEX
# ============================================================


def build_chembl_synonym_index(chembl_db_path: Path) -> dict[str, str]:
    """Build in-memory index: lowered_synonym -> chembl_id.

    Reads from ChEMBL SQLite molecule_synonyms + molecule_dictionary.
    Returns dict with ~500K entries.
    """
    if not chembl_db_path.exists():
        logger.warning("ChEMBL SQLite not found: %s", chembl_db_path)
        return {}

    conn = sqlite3.connect(str(chembl_db_path))
    try:
        # molecule_synonyms: molregno, syn_type, synonyms
        # molecule_dictionary: molregno, chembl_id, pref_name
        cursor = conn.execute("""
            SELECT LOWER(ms.synonyms), md.chembl_id
            FROM molecule_synonyms ms
            JOIN molecule_dictionary md ON ms.molregno = md.molregno
            WHERE ms.synonyms IS NOT NULL
              AND md.chembl_id IS NOT NULL
        """)
        index: dict[str, str] = {}
        for syn, chembl_id in cursor:
            if syn and syn.strip():
                index.setdefault(syn.strip(), chembl_id)

        # Also add pref_name entries
        cursor = conn.execute("""
            SELECT LOWER(pref_name), chembl_id
            FROM molecule_dictionary
            WHERE pref_name IS NOT NULL
        """)
        for name, chembl_id in cursor:
            if name and name.strip():
                index.setdefault(name.strip(), chembl_id)

        logger.info("ChEMBL synonym index: %d entries", len(index))
        return index
    finally:
        conn.close()


def resolve_step1_chembl(
    names: list[str],
    synonym_index: dict[str, str],
) -> dict[str, str]:
    """Step 1: Exact match against ChEMBL synonym index.

    Returns dict mapping cleaned_name -> chembl_id for matches.
    """
    resolved: dict[str, str] = {}
    for name in names:
        cleaned = clean_drug_name(name)
        if cleaned in synonym_index:
            resolved[name] = synonym_index[cleaned]
    logger.info("Step 1 (ChEMBL exact): %d / %d resolved", len(resolved), len(names))
    return resolved


# ============================================================
# STEP 2: PUBCHEM REST API
# ============================================================

_PUG_XML_TEMPLATE = """\
<?xml version="1.0"?>
<PCT-Data>
  <PCT-Data_input>
    <PCT-InputData>
      <PCT-InputData_query>
        <PCT-Query>
          <PCT-Query_type>
            <PCT-QueryType>
              <PCT-QueryType_id-exchange>
                <PCT-QueryIDExchange>
                  <PCT-QueryIDExchange_input>
                    <PCT-QueryUids>
                      <PCT-QueryUids_synonyms>
{synonym_elements}
                      </PCT-QueryUids_synonyms>
                    </PCT-QueryUids>
                  </PCT-QueryIDExchange_input>
                  <PCT-QueryIDExchange_operation-type value="same"/>
                  <PCT-QueryIDExchange_output-type value="cid"/>
                  <PCT-QueryIDExchange_output-method value="file-pair"/>
                  <PCT-QueryIDExchange_compression value="none"/>
                </PCT-QueryIDExchange>
              </PCT-QueryType_id-exchange>
            </PCT-QueryType>
          </PCT-Query_type>
        </PCT-Query>
      </PCT-InputData_query>
    </PCT-InputData>
  </PCT-Data_input>
</PCT-Data>"""

_PUG_POLL_TEMPLATE = """\
<?xml version="1.0"?>
<PCT-Data>
  <PCT-Data_input>
    <PCT-InputData>
      <PCT-InputData_request>
        <PCT-Request>
          <PCT-Request_reqid>{reqid}</PCT-Request_reqid>
          <PCT-Request_type value="status"/>
        </PCT-Request>
      </PCT-InputData_request>
    </PCT-InputData>
  </PCT-Data_input>
</PCT-Data>"""

PUG_URL = "https://pubchem.ncbi.nlm.nih.gov/pug/pug.cgi"


def _pubchem_name_lookup(
    name: str,
    rate_limit: float = 5.0,
    last_call_time: list[float] | None = None,
) -> dict | None:
    """Look up a compound name via PubChem REST API (single name).

    Returns dict with CID, CanonicalSMILES, InChIKey or None.
    Rate-limited to `rate_limit` requests/sec.
    """
    import urllib.request
    import urllib.error
    import urllib.parse

    if last_call_time is not None and last_call_time:
        elapsed = time.time() - last_call_time[0]
        wait = (1.0 / rate_limit) - elapsed
        if wait > 0:
            time.sleep(wait)

    encoded = urllib.parse.quote(name)
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{encoded}/property/IsomericSMILES,InChIKey/JSON"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NegBioDB/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        if last_call_time is not None:
            last_call_time.clear()
            last_call_time.append(time.time())
        props = data.get("PropertyTable", {}).get("Properties", [])
        if props:
            p = props[0]
            return {
                "cid": p.get("CID"),
                "smiles": p.get("IsomericSMILES") or p.get("SMILES") or p.get("CanonicalSMILES"),
                "inchikey": p.get("InChIKey"),
            }
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        pass
    except Exception as e:
        logger.debug("PubChem lookup failed for %s: %s", name, e)

    if last_call_time is not None:
        last_call_time.clear()
        last_call_time.append(time.time())
    return None


def _xml_escape(s: str) -> str:
    """Escape XML special characters."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _pug_submit_batch(names: list[str], max_retries: int = 3) -> str | None:
    """Submit batch name→CID request to PUG XML. Returns request ID."""
    import urllib.request
    import urllib.error
    import xml.etree.ElementTree as ET

    elements = "\n".join(
        f"                        <PCT-QueryUids_synonyms_E>{_xml_escape(n)}</PCT-QueryUids_synonyms_E>"
        for n in names
    )
    xml_body = _PUG_XML_TEMPLATE.replace("{synonym_elements}", elements)

    for attempt in range(max_retries):
        req = urllib.request.Request(
            PUG_URL,
            data=xml_body.encode("utf-8"),
            headers={"Content-Type": "application/xml", "User-Agent": "NegBioDB/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                root = ET.fromstring(resp.read())
            # Extract reqid from response
            for elem in root.iter():
                if "Waiting_reqid" in elem.tag and elem.text:
                    return elem.text.strip()
            # Check for immediate result
            for elem in root.iter():
                if "Status" in elem.tag:
                    val = elem.get("value", "")
                    if val in ("success", "hit"):
                        for dl in root.iter():
                            if "Download-URL_url" in dl.tag and dl.text:
                                return f"DIRECT:{dl.text.strip()}"
            # No reqid and no success — likely a server issue
            return None
        except Exception as e:
            logger.warning("PUG batch submit attempt %d/%d failed: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def _pug_poll(reqid: str, max_wait: int = 300) -> str | None:
    """Poll PUG for batch job completion. Returns download URL."""
    import urllib.request
    import xml.etree.ElementTree as ET

    xml_body = _PUG_POLL_TEMPLATE.replace("{reqid}", _xml_escape(reqid))
    deadline = time.time() + max_wait

    while time.time() < deadline:
        time.sleep(5)
        req = urllib.request.Request(
            PUG_URL,
            data=xml_body.encode("utf-8"),
            headers={"Content-Type": "application/xml", "User-Agent": "NegBioDB/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                root = ET.fromstring(resp.read())

            for elem in root.iter():
                if "Status" in elem.tag:
                    val = elem.get("value", "")
                    if val == "success" or val == "hit":
                        # Find download URL
                        for dl in root.iter():
                            if "Download-URL_url" in dl.tag and dl.text:
                                return dl.text.strip()
                    elif val in ("server-error", "data-error", "input-error"):
                        logger.warning("PUG batch job failed with status: %s", val)
                        return None
            # Still running — continue polling
        except Exception as e:
            logger.debug("PUG poll error: %s", e)

    logger.warning("PUG batch job timed out after %ds", max_wait)
    return None


def _pug_download_results(url: str) -> dict[str, int]:
    """Download PUG batch result TSV and parse name→CID mapping."""
    import urllib.request

    # PUG sometimes returns ftp:// URLs; convert to https
    if url.startswith("ftp://"):
        url = url.replace("ftp://", "https://", 1)
    # Normalize host to ftp.ncbi.nlm.nih.gov if needed
    if "ftp.ncbi.nlm.nih.gov" not in url and "pubchem.ncbi.nlm.nih.gov" not in url:
        url = re.sub(r"https?://[^/]+", "https://ftp.ncbi.nlm.nih.gov", url)

    req = urllib.request.Request(url, headers={"User-Agent": "NegBioDB/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning("PUG download failed: %s", e)
        return {}

    result: dict[str, int] = {}
    for line in data.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip():
            name = parts[0].strip()
            try:
                cid = int(parts[1].strip())
                result[name.lower()] = cid
            except ValueError:
                continue
    return result


def _pubchem_batch_names_to_cids(
    names: list[str], batch_size: int = 5000,
) -> dict[str, int]:
    """Batch-resolve compound names → CIDs via PUG XML interface.

    Returns dict mapping lowercase name → CID.
    """
    result: dict[str, int] = {}
    batches = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]

    for i, batch in enumerate(batches):
        logger.info("PUG batch %d/%d (%d names)...", i + 1, len(batches), len(batch))
        reqid = _pug_submit_batch(batch)
        if reqid is None:
            logger.warning("PUG batch %d submit failed, skipping", i + 1)
            continue

        if reqid.startswith("DIRECT:"):
            url = reqid[7:]
        else:
            url = _pug_poll(reqid, max_wait=600)

        if url:
            batch_result = _pug_download_results(url)
            result.update(batch_result)
            logger.info("PUG batch %d: %d / %d resolved", i + 1, len(batch_result), len(batch))
        else:
            logger.warning("PUG batch %d: no results (poll failed or timed out)", i + 1)

        # Small delay between batches
        if i < len(batches) - 1:
            time.sleep(2)

    return result


def _pubchem_batch_cid_to_properties(
    cids: list[int], batch_size: int = 500, max_retries: int = 3,
) -> dict[int, dict]:
    """Batch-fetch CID → {smiles, inchikey} via PUG REST POST.

    Returns dict mapping CID → {smiles, inchikey}.
    """
    import urllib.request
    import urllib.error

    result: dict[int, dict] = {}
    batches = [cids[i:i + batch_size] for i in range(0, len(cids), batch_size)]

    for i, batch in enumerate(batches):
        cid_str = ",".join(str(c) for c in batch)
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/IsomericSMILES,CanonicalSMILES,InChIKey/JSON"
        body = f"cid={cid_str}".encode("utf-8")

        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "NegBioDB/1.0"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())

                for p in data.get("PropertyTable", {}).get("Properties", []):
                    cid = p.get("CID")
                    if cid:
                        # POST response uses varying key names for SMILES;
                        # prefer IsomericSMILES (preserves stereochemistry)
                        smiles = (
                            p.get("IsomericSMILES")
                            or p.get("SMILES")
                            or p.get("CanonicalSMILES")
                            or p.get("ConnectivitySMILES")
                        )
                        result[cid] = {
                            "smiles": smiles,
                            "inchikey": p.get("InChIKey"),
                        }
                break  # Success — no retry needed
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                logger.warning("PUG REST CID batch %d/%d attempt %d/%d failed: %s",
                               i + 1, len(batches), attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning("PUG REST CID batch non-retryable error: %s", e)
                break

        # Rate limit: ~5 req/sec
        time.sleep(0.25)

    return result


def resolve_step2_pubchem(
    names: list[str],
    cache_path: Path | None = None,
    rate_limit: float = 5.0,
    use_batch: bool = True,
) -> dict[str, dict]:
    """Step 2: PubChem name lookup with caching.

    Uses batch PUG XML + REST POST for large sets (>100 names),
    falls back to single-name REST for small sets or testing.

    Returns dict mapping name -> {cid, smiles, inchikey}.
    """
    # Load cache (backward-compatible: old int values become CID-only dicts)
    raw_cache: dict = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            raw_cache = json.load(f)
        logger.info("PubChem cache loaded: %d entries", len(raw_cache))

    cache: dict[str, dict | None] = {}
    for k, v in raw_cache.items():
        if v is None:
            cache[k] = None
        elif isinstance(v, int):
            # Backward-compat: old cache stored just CID as int
            cache[k] = {"cid": v, "smiles": None, "inchikey": None}
        elif isinstance(v, dict):
            cache[k] = v
        else:
            cache[k] = None

    resolved: dict[str, dict] = {}
    uncached_names: list[str] = []
    uncached_cleaned: list[str] = []
    name_to_cleaned: dict[str, str] = {}

    for name in names:
        cleaned = clean_drug_name(name)
        name_to_cleaned[name] = cleaned
        if cleaned in cache:
            entry = cache[cleaned]
            if entry is not None:
                resolved[name] = entry
        else:
            uncached_names.append(name)
            uncached_cleaned.append(cleaned)

    logger.info("PubChem: %d cached hits, %d uncached to resolve", len(resolved), len(uncached_names))

    if uncached_cleaned:
        unique_cleaned = list(dict.fromkeys(uncached_cleaned))

        if use_batch and len(unique_cleaned) > 100:
            # BATCH: PUG XML name→CID, then PUG REST CID→properties
            logger.info("Using PUG XML batch for %d unique names", len(unique_cleaned))
            name_to_cid = _pubchem_batch_names_to_cids(unique_cleaned)

            resolved_cids = list(set(name_to_cid.values()))
            logger.info("PUG batch: %d names → %d unique CIDs", len(name_to_cid), len(resolved_cids))

            cid_to_props = _pubchem_batch_cid_to_properties(resolved_cids) if resolved_cids else {}
            logger.info("CID→properties: %d / %d fetched", len(cid_to_props), len(resolved_cids))

            # Merge into cache
            for cleaned in unique_cleaned:
                cid = name_to_cid.get(cleaned)
                if cid is not None:
                    props = cid_to_props.get(cid, {})
                    entry = {
                        "cid": cid,
                        "smiles": props.get("smiles"),
                        "inchikey": props.get("inchikey"),
                    }
                    cache[cleaned] = entry
                else:
                    cache[cleaned] = None

            # Map back to original names
            for name in uncached_names:
                cleaned = name_to_cleaned[name]
                entry = cache.get(cleaned)
                if entry is not None:
                    resolved[name] = entry
        else:
            # SINGLE: Original per-name REST API
            last_call = [0.0]
            for name in tqdm(uncached_names, desc="Step 2: PubChem API"):
                cleaned = name_to_cleaned[name]
                if cleaned in cache:
                    entry = cache[cleaned]
                    if entry is not None:
                        resolved[name] = entry
                    continue

                result = _pubchem_name_lookup(cleaned, rate_limit, last_call)
                cache[cleaned] = result
                if result is not None:
                    resolved[name] = result

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f)
        logger.info("PubChem cache saved: %d entries", len(cache))

    logger.info("Step 2 (PubChem): %d / %d resolved", len(resolved), len(names))
    return resolved


# ============================================================
# STEP 3: FUZZY MATCH
# ============================================================


def resolve_step3_fuzzy(
    names: list[str],
    synonym_index: dict[str, str],
    threshold: float = 0.90,
) -> dict[str, str]:
    """Step 3: Fuzzy match using Jaro-Winkler similarity.

    Pre-filters candidates by name length (±5 chars) for speed.
    Returns dict mapping name -> chembl_id.
    """
    from rapidfuzz.distance import JaroWinkler

    # Build length-bucketed index for fast pre-filtering
    by_length: dict[int, list[tuple[str, str]]] = {}
    for syn, chembl_id in synonym_index.items():
        slen = len(syn)
        by_length.setdefault(slen, []).append((syn, chembl_id))

    resolved: dict[str, str] = {}
    for name in tqdm(names, desc="Step 3: Fuzzy match"):
        cleaned = clean_drug_name(name)
        if not cleaned:
            continue

        target_len = len(cleaned)
        best_score = 0.0
        best_chembl = None

        for delta in range(-5, 6):
            bucket = by_length.get(target_len + delta, [])
            for syn, chembl_id in bucket:
                score = JaroWinkler.similarity(cleaned, syn)
                if score > best_score:
                    best_score = score
                    best_chembl = chembl_id

        if best_score >= threshold and best_chembl is not None:
            resolved[name] = best_chembl

    logger.info("Step 3 (fuzzy): %d / %d resolved", len(resolved), len(names))
    return resolved


# ============================================================
# STEP 4: MANUAL OVERRIDES
# ============================================================


def load_overrides(csv_path: Path) -> dict[str, dict]:
    """Load manual drug name override CSV.

    CSV columns: intervention_name, chembl_id, canonical_smiles, molecular_type
    Returns dict mapping lowered_name -> override dict.
    """
    if not csv_path.exists():
        logger.info("No override file at %s", csv_path)
        return {}

    df = pd.read_csv(csv_path)
    overrides: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = str(row["intervention_name"]).strip().lower()
        overrides[key] = {
            "chembl_id": row.get("chembl_id") if pd.notna(row.get("chembl_id")) else None,
            "canonical_smiles": (
                row.get("canonical_smiles")
                if pd.notna(row.get("canonical_smiles")) else None
            ),
            "molecular_type": (
                row.get("molecular_type")
                if pd.notna(row.get("molecular_type")) else None
            ),
        }

    logger.info("Loaded %d manual overrides", len(overrides))
    return overrides


def resolve_step4_overrides(
    names: list[str],
    overrides: dict[str, dict],
) -> dict[str, dict]:
    """Step 4: Manual override CSV lookup.

    Returns dict mapping name -> override dict (chembl_id, smiles, etc.).
    """
    resolved: dict[str, dict] = {}
    for name in names:
        cleaned = clean_drug_name(name)
        if cleaned in overrides:
            resolved[name] = overrides[cleaned]
    logger.info("Step 4 (overrides): %d / %d resolved", len(resolved), len(names))
    return resolved


# ============================================================
# CHEMBL COMPOUND DETAILS + TARGET MAPPING
# ============================================================


def fetch_chembl_details(
    chembl_ids: list[str],
    chembl_db_path: Path,
) -> dict[str, dict]:
    """Fetch SMILES, InChIKey, molecular_type from ChEMBL SQLite.

    Returns dict mapping chembl_id -> {smiles, inchikey, inchikey_connectivity, mol_type}.
    """
    if not chembl_db_path.exists():
        return {}

    conn = sqlite3.connect(str(chembl_db_path))
    try:
        details: dict[str, dict] = {}
        for i in range(0, len(chembl_ids), BATCH_SIZE):
            batch = chembl_ids[i:i + BATCH_SIZE]
            placeholders = ",".join(["?"] * len(batch))
            cursor = conn.execute(
                f"SELECT md.chembl_id, cs.canonical_smiles, "
                f"  cs.standard_inchi_key, md.molecule_type "
                f"FROM molecule_dictionary md "
                f"LEFT JOIN compound_structures cs ON md.molregno = cs.molregno "
                f"WHERE md.chembl_id IN ({placeholders})",
                batch,
            )
            for row in cursor:
                chembl_id, smiles, inchikey, mol_type = row
                inchikey_conn = inchikey.split("-")[0] if inchikey else None
                details[chembl_id] = {
                    "canonical_smiles": smiles,
                    "inchikey": inchikey,
                    "inchikey_connectivity": inchikey_conn,
                    "molecular_type": _map_mol_type(mol_type),
                }
        return details
    finally:
        conn.close()


def _map_mol_type(chembl_mol_type: str | None) -> str:
    """Map ChEMBL molecule_type to our schema molecular_type CHECK values."""
    if not chembl_mol_type:
        return "unknown"
    mt = str(chembl_mol_type).lower()
    if "small" in mt:
        return "small_molecule"
    elif "antibody" in mt and "drug" in mt:
        return "antibody_drug_conjugate"
    elif "antibody" in mt:
        return "monoclonal_antibody"
    elif "protein" in mt or "peptide" in mt:
        return "peptide"
    elif "oligonucleotide" in mt:
        return "oligonucleotide"
    elif "cell" in mt:
        return "cell_therapy"
    elif "gene" in mt:
        return "gene_therapy"
    elif "enzyme" in mt or "unknown" in mt:
        return "unknown"
    else:
        return "other_biologic"


def fetch_chembl_targets(
    chembl_ids: list[str],
    chembl_db_path: Path,
) -> list[dict]:
    """Fetch drug-target relationships from ChEMBL drug_mechanism.

    Returns list of dicts with: chembl_id, uniprot_accession, gene_symbol,
    action_type.
    """
    if not chembl_db_path.exists():
        return []

    conn = sqlite3.connect(str(chembl_db_path))
    try:
        targets: list[dict] = []
        for i in range(0, len(chembl_ids), BATCH_SIZE):
            batch = chembl_ids[i:i + BATCH_SIZE]
            placeholders = ",".join(["?"] * len(batch))
            cursor = conn.execute(
                f"SELECT md.chembl_id, cs.accession, "
                f"  td.pref_name, dm.action_type "
                f"FROM drug_mechanism dm "
                f"JOIN molecule_dictionary md ON dm.molregno = md.molregno "
                f"JOIN target_dictionary td ON dm.tid = td.tid "
                f"JOIN target_components tc ON td.tid = tc.tid "
                f"JOIN component_sequences cs ON tc.component_id = cs.component_id "
                f"WHERE md.chembl_id IN ({placeholders}) "
                f"  AND cs.accession IS NOT NULL",
                batch,
            )
            for row in cursor:
                chembl_id, accession, target_name, action_type = row
                targets.append({
                    "chembl_id": chembl_id,
                    "uniprot_accession": accession,
                    "gene_symbol": target_name,
                    "action_type": action_type,
                })
        logger.info("ChEMBL targets found: %d", len(targets))
        return targets
    finally:
        conn.close()


# ============================================================
# PUBCHEM → CHEMBL CROSS-REFERENCE
# ============================================================


def crossref_inchikey_to_chembl(
    inchikeys: list[str],
    chembl_db_path: Path,
) -> dict[str, str]:
    """Look up InChIKeys in ChEMBL compound_structures → chembl_id.

    Returns dict mapping inchikey -> chembl_id for matches.
    """
    if not chembl_db_path.exists() or not inchikeys:
        return {}

    conn = sqlite3.connect(str(chembl_db_path))
    try:
        result: dict[str, str] = {}
        for i in range(0, len(inchikeys), BATCH_SIZE):
            batch = inchikeys[i:i + BATCH_SIZE]
            placeholders = ",".join(["?"] * len(batch))
            cursor = conn.execute(
                f"SELECT cs.standard_inchi_key, md.chembl_id "
                f"FROM compound_structures cs "
                f"JOIN molecule_dictionary md ON cs.molregno = md.molregno "
                f"WHERE cs.standard_inchi_key IN ({placeholders})",
                batch,
            )
            for inchikey, chembl_id in cursor:
                result.setdefault(inchikey, chembl_id)
        logger.info("InChIKey→ChEMBL cross-ref: %d / %d matched",
                     len(result), len(inchikeys))
        return result
    finally:
        conn.close()


# ============================================================
# UPDATE DATABASE
# ============================================================


def update_interventions(
    conn: sqlite3.Connection,
    resolutions: dict[int, dict],
) -> int:
    """Update interventions table with resolved identifiers.

    Args:
        conn: Database connection.
        resolutions: dict mapping intervention_id -> {chembl_id, canonical_smiles,
            inchikey, inchikey_connectivity, molecular_type, pubchem_cid}.

    Returns number of rows updated.
    """
    count = 0
    for interv_id, data in tqdm(resolutions.items(), desc="Update interventions"):
        sets = []
        vals = []
        for col in ["chembl_id", "canonical_smiles", "inchikey",
                     "inchikey_connectivity", "molecular_type", "pubchem_cid"]:
            if col in data and data[col] is not None:
                sets.append(f"{col} = ?")
                vals.append(data[col])

        if not sets:
            continue

        vals.append(interv_id)
        conn.execute(
            f"UPDATE interventions SET {', '.join(sets)} WHERE intervention_id = ?",
            vals,
        )
        count += 1

    conn.commit()
    logger.info("Updated %d interventions", count)
    return count


def insert_intervention_targets(
    conn: sqlite3.Connection,
    targets: list[dict],
    chembl_to_interv: dict[str, list[int]],
) -> int:
    """Insert intervention_targets from ChEMBL drug_mechanism data.

    Returns number of rows inserted.
    """
    count = 0
    seen: set[tuple] = set()

    for t in tqdm(targets, desc="Insert intervention targets"):
        chembl_id = t["chembl_id"]
        interv_ids = chembl_to_interv.get(chembl_id, [])

        for interv_id in interv_ids:
            key = (interv_id, t["uniprot_accession"])
            if key in seen:
                continue
            seen.add(key)

            conn.execute(
                "INSERT OR IGNORE INTO intervention_targets "
                "(intervention_id, uniprot_accession, gene_symbol, "
                " action_type, source) "
                "VALUES (?, ?, ?, ?, 'chembl')",
                (interv_id, t["uniprot_accession"],
                 t["gene_symbol"], t["action_type"]),
            )
            count += 1

    conn.commit()
    logger.info("Inserted %d intervention-target links", count)
    return count


# ============================================================
# ORCHESTRATOR
# ============================================================


def run_drug_resolution(
    db_path: Path = DEFAULT_CT_DB_PATH,
    chembl_db_path: Path | None = None,
    skip_pubchem: bool = False,
    skip_fuzzy: bool = False,
) -> dict:
    """Run the full drug name resolution cascade.

    Steps:
      1. Load drug interventions from CT database
      2. Apply overrides (Step 4 first — highest priority)
      3. ChEMBL exact match (Step 1)
      4. PubChem API (Step 2, optional)
      5. Fuzzy match (Step 3, optional)
      6. Fetch compound details from ChEMBL
      7. Update interventions table
      8. Map and insert targets

    Returns dict with resolution statistics.
    """
    cfg = load_config()
    ct_cfg = cfg["ct_domain"]
    dr_cfg = ct_cfg.get("drug_resolution", {})

    if chembl_db_path is None:
        chembl_dir = Path(cfg["downloads"]["chembl"]["dest_dir"])
        # Find the ChEMBL SQLite file
        candidates = sorted(chembl_dir.glob("chembl_*.db"))
        if candidates:
            chembl_db_path = candidates[-1]
        else:
            logger.warning("No ChEMBL SQLite found in %s", chembl_dir)
            chembl_db_path = Path("/dev/null")

    conn = get_connection(db_path)
    stats: dict = {}

    try:
        # Load drug interventions
        cursor = conn.execute(
            "SELECT intervention_id, intervention_name, intervention_type "
            "FROM interventions "
            "WHERE intervention_type IN ('drug', 'biologic', 'combination') "
            "  AND chembl_id IS NULL"
        )
        drugs = cursor.fetchall()
        logger.info("Unresolved drug interventions: %d", len(drugs))
        stats["total_unresolved"] = len(drugs)

        if not drugs:
            return stats

        # name -> list of intervention_ids (multiple interventions can share a name)
        id_map: dict[str, list[int]] = {}
        for row in drugs:
            id_map.setdefault(row[1], []).append(row[0])
        names = list(id_map.keys())

        # Pre-filter: skip non-drug names (placebos, controls, etc.)
        non_drug = [n for n in names if is_non_drug_name(n)]
        drug_names = [n for n in names if not is_non_drug_name(n)]
        stats["non_drug_filtered"] = len(non_drug)
        logger.info("Pre-filter: %d non-drug names skipped", len(non_drug))

        # Step 4: Manual overrides (highest priority)
        overrides_path = Path(dr_cfg.get("overrides_file", "data/ct/drug_name_overrides.csv"))
        overrides = load_overrides(overrides_path)
        step4 = resolve_step4_overrides(drug_names, overrides)
        stats["step4_overrides"] = len(step4)

        remaining = [n for n in drug_names if n not in step4]

        # Step 1: ChEMBL exact match
        synonym_index = build_chembl_synonym_index(chembl_db_path)
        step1 = resolve_step1_chembl(remaining, synonym_index)
        stats["step1_chembl_exact"] = len(step1)

        remaining = [n for n in remaining if n not in step1]

        # Step 2: PubChem API (optional)
        step2: dict[str, dict] = {}
        if not skip_pubchem and remaining:
            cache_path = Path(dr_cfg.get("pubchem_cache", "data/ct/pubchem_name_cache.json"))
            rate_limit = dr_cfg.get("pubchem_rate_limit_per_sec", 5)
            step2 = resolve_step2_pubchem(remaining, cache_path, rate_limit)
        stats["step2_pubchem"] = len(step2)

        # Step 2b: Cross-reference PubChem InChIKeys against ChEMBL
        pubchem_inchikeys = {}  # inchikey -> list of names
        for name, pdata in step2.items():
            ik = pdata.get("inchikey")
            if ik:
                pubchem_inchikeys.setdefault(ik, []).append(name)

        crossref = crossref_inchikey_to_chembl(
            list(pubchem_inchikeys.keys()), chembl_db_path
        )
        stats["step2b_crossref"] = len(crossref)

        remaining = [n for n in remaining if n not in step2]

        # Step 3: Fuzzy match (optional)
        step3: dict[str, str] = {}
        if not skip_fuzzy and remaining and synonym_index:
            threshold = dr_cfg.get("fuzzy_threshold", 0.90)
            step3 = resolve_step3_fuzzy(remaining, synonym_index, threshold)
        stats["step3_fuzzy"] = len(step3)

        # Combine all resolutions → chembl_ids
        all_chembl_ids: dict[str, str] = {}  # name -> chembl_id
        for name, chembl_id in step1.items():
            all_chembl_ids[name] = chembl_id
        for name, chembl_id in step3.items():
            all_chembl_ids.setdefault(name, chembl_id)
        for name, override in step4.items():
            if override.get("chembl_id"):
                all_chembl_ids[name] = override["chembl_id"]

        # Add PubChem→ChEMBL cross-referenced entries
        for inchikey, chembl_id in crossref.items():
            for name in pubchem_inchikeys.get(inchikey, []):
                all_chembl_ids.setdefault(name, chembl_id)

        # Fetch compound details from ChEMBL
        unique_chembl_ids = list(set(all_chembl_ids.values()))
        details = fetch_chembl_details(unique_chembl_ids, chembl_db_path)

        # Build resolution dict: intervention_id -> data
        resolutions: dict[int, dict] = {}
        chembl_to_interv: dict[str, list[int]] = {}

        for name, chembl_id in all_chembl_ids.items():
            for interv_id in id_map.get(name, []):
                d = details.get(chembl_id, {})
                resolutions[interv_id] = {
                    "chembl_id": chembl_id,
                    "canonical_smiles": d.get("canonical_smiles"),
                    "inchikey": d.get("inchikey"),
                    "inchikey_connectivity": d.get("inchikey_connectivity"),
                    "molecular_type": d.get("molecular_type", "unknown"),
                }
                chembl_to_interv.setdefault(chembl_id, []).append(interv_id)

        # Add PubChem-only data (no ChEMBL match)
        for name, pdata in step2.items():
            for interv_id in id_map.get(name, []):
                if interv_id not in resolutions:
                    resolutions[interv_id] = {}
                resolutions[interv_id]["pubchem_cid"] = pdata.get("cid")
                # Store PubChem SMILES/InChIKey if no ChEMBL data present
                if not resolutions[interv_id].get("canonical_smiles") and pdata.get("smiles"):
                    resolutions[interv_id]["canonical_smiles"] = pdata["smiles"]
                if not resolutions[interv_id].get("inchikey") and pdata.get("inchikey"):
                    ik = pdata["inchikey"]
                    resolutions[interv_id]["inchikey"] = ik
                    resolutions[interv_id]["inchikey_connectivity"] = ik.split("-")[0] if ik else None

        # Add override data (skip entries with no useful values)
        for name, override in step4.items():
            has_data = any(v is not None for v in override.values())
            if not has_data:
                continue  # Skip entry (placebo, generic class, etc.)
            for interv_id in id_map.get(name, []):
                if interv_id not in resolutions:
                    resolutions[interv_id] = {}
                for k, v in override.items():
                    if v is not None:
                        resolutions[interv_id][k] = v

        # Update database
        n_updated = update_interventions(conn, resolutions)
        stats["interventions_updated"] = n_updated

        # Target mapping
        targets = fetch_chembl_targets(unique_chembl_ids, chembl_db_path)
        n_targets = insert_intervention_targets(conn, targets, chembl_to_interv)
        stats["targets_inserted"] = n_targets

        # Final counts (use intervention_ids for consistent units)
        total_resolved = len(resolutions)
        total_drug_ids = sum(
            len(ids) for name, ids in id_map.items()
            if not is_non_drug_name(name)
        )
        stats["total_resolved"] = total_resolved
        stats["total_drug_interventions"] = total_drug_ids
        stats["coverage_pct"] = (
            round(100 * total_resolved / total_drug_ids, 1) if total_drug_ids > 0 else 0
        )

    finally:
        conn.close()

    logger.info("=== Drug Resolution Complete ===")
    for k, v in stats.items():
        logger.info("  %s: %s", k, v)

    return stats
