"""ETL pipeline for loading DAVIS kinase binding dataset into NegBioDB.

DAVIS provides 68 kinase inhibitors × 433 kinases with pKd values.
Entries with pKd <= 5.0 (detection limit) are loaded as hard negatives.
Entries with pKd >= 7.0 are tracked but NOT loaded (active).
Entries with 5.0 < pKd < 7.0 are excluded (borderline).

Target handling policy:
- `targets.uniprot_accession` stores canonical UniProt accession only.
- Mutation context (e.g., E255K, T315I) is stored in `target_variants`.

Reference: Davis et al., Nature Biotechnology 29, 1046-1051 (2011)
"""

import json
import logging
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests

from negbiodb.db import connect, create_database, _PROJECT_ROOT
from negbiodb.download import load_config
from negbiodb.standardize import standardize_smiles

logger = logging.getLogger(__name__)


# ============================================================
# EXTRACT
# ============================================================


def load_davis_csvs(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three DAVIS CSV files.

    Returns (drugs_df, proteins_df, affinities_df).
    """
    drugs = pd.read_csv(data_dir / "drugs.csv")
    proteins = pd.read_csv(data_dir / "proteins.csv")
    affinities = pd.read_csv(data_dir / "drug_protein_affinity.csv")
    return drugs, proteins, affinities


# ============================================================
# TRANSFORM: Compounds
# ============================================================


def standardize_compound(smiles: str, cid: int) -> dict | None:
    """Standardize a single compound using RDKit.

    Returns dict with all compound fields, or None if SMILES fails to parse.
    """
    result = standardize_smiles(smiles)
    if result is None:
        return None
    result["pubchem_cid"] = int(cid)
    return result


def standardize_all_compounds(drugs_df: pd.DataFrame) -> list[dict]:
    """Standardize all DAVIS drugs. Returns list of compound dicts."""
    compounds = []
    for _, row in drugs_df.iterrows():
        comp = standardize_compound(row["Canonical_SMILES"], row["CID"])
        if comp is not None:
            comp["drug_index"] = int(row["Drug_Index"])
            compounds.append(comp)
        else:
            logger.warning("Failed to standardize Drug_Index=%d", row["Drug_Index"])
    return compounds


# ============================================================
# TRANSFORM: Targets
# ============================================================


def parse_gene_name(gene_name: str) -> tuple[str, str | None]:
    """Parse DAVIS gene name into (base_gene, mutation).

    Examples:
        'ABL1(E255K)-phosphorylated' -> ('ABL1', 'E255K')
        'ABL1(T315I)-phosphorylated' -> ('ABL1', 'T315I')
        'ABL1-phosphorylated'        -> ('ABL1', None)
        'PIK3CA'                     -> ('PIK3CA', None)
    """
    m = re.match(r"^([A-Za-z0-9_]+)\(([^)]+)\)", gene_name)
    if m:
        return m.group(1), m.group(2)

    # Strip suffixes like '-phosphorylated', '-alpha', '-nonphosphorylated'
    base = re.split(r"-(?:phosphorylated|nonphosphorylated)", gene_name)[0]
    return base, None


def _classify_accession(acc_id: str) -> str:
    """Classify a protein accession by database type.

    Returns 'refseq', 'genbank', or 'uniprot'.
    """
    if acc_id.startswith(("NP_", "XP_", "YP_", "WP_")):
        return "refseq"
    # UniProt accessions: old format [OPQ]XXXXX, new format [A-NR-Z]XXXXXXXXX
    base = acc_id.split(".")[0]
    if re.match(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$", base) or re.match(
        r"^[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]([A-Z0-9]{0,4})$", base
    ):
        return "uniprot"
    return "genbank"


def _download_refseq_mapping() -> dict[str, str]:
    """Download reviewed human RefSeq→UniProt mapping via UniProt search API.

    Returns dict mapping unversioned RefSeq IDs (e.g. 'NP_005408') to
    reviewed UniProt accessions. Downloads ~19k entries in one request.
    """
    url = (
        "https://rest.uniprot.org/uniprotkb/stream"
        "?query=(organism_id:9606)+AND+(reviewed:true)+AND+(database:refseq)"
        "&fields=accession,xref_refseq&format=tsv"
    )
    logger.info("Downloading human RefSeq->UniProt mapping from UniProt...")
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to download RefSeq mapping: %s", e)
        return {}

    refseq_to_uniprot: dict[str, str] = {}
    lines = resp.text.strip().split("\n")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        uniprot_acc = parts[0]
        for entry in parts[1].split(";"):
            entry = entry.strip()
            if not entry:
                continue
            refseq_id = entry.split()[0]
            base = refseq_id.split(".")[0]
            if base not in refseq_to_uniprot:
                refseq_to_uniprot[base] = uniprot_acc

    logger.info(
        "Downloaded %d RefSeq->UniProt mappings (%d UniProt entries)",
        len(refseq_to_uniprot), len(lines) - 1,
    )
    return refseq_to_uniprot


def _search_uniprot_by_gene(gene_symbols: list[str]) -> dict[str, str]:
    """Map gene symbols to reviewed human UniProt accessions via search API.

    Uses individual search queries (one per gene) with rate limiting.
    Returns dict mapping gene symbol to UniProt accession.
    """
    if not gene_symbols:
        return {}

    mapping: dict[str, str] = {}
    for gene in gene_symbols:
        url = (
            f"https://rest.uniprot.org/uniprotkb/search"
            f"?query=(gene_exact:{gene})+AND+(organism_id:9606)+AND+(reviewed:true)"
            f"&fields=accession&format=tsv&size=1"
        )
        try:
            r = requests.get(url, timeout=10)
            if r.ok:
                lines = r.text.strip().split("\n")
                if len(lines) > 1:
                    mapping[gene] = lines[1].strip()
        except requests.RequestException:
            pass
        time.sleep(0.2)

    logger.info("Gene symbol search: mapped %d / %d genes", len(mapping), len(gene_symbols))
    return mapping


def _call_uniprot_idmapping(
    ids: list[str], from_db: str
) -> dict[str, str]:
    """Call UniProt ID Mapping API for a single database type.

    Strips version numbers before submitting.
    Uses allow_redirects=False for reliable polling.

    Returns dict mapping original input IDs to UniProt accessions.
    """
    if not ids:
        return {}

    unversioned_to_original: dict[str, str] = {}
    for acc_id in ids:
        base = acc_id.split(".")[0]
        unversioned_to_original[base] = acc_id
    unversioned_ids = list(unversioned_to_original.keys())

    logger.info("Submitting %d %s IDs to UniProt ID Mapping API...", len(unversioned_ids), from_db)
    try:
        resp = requests.post(
            "https://rest.uniprot.org/idmapping/run",
            data={"from": from_db, "to": "UniProtKB", "ids": ",".join(unversioned_ids)},
            timeout=30,
        )
        resp.raise_for_status()
        job_id = resp.json()["jobId"]
    except (requests.RequestException, KeyError) as e:
        logger.warning("UniProt API submission failed (%s): %s", from_db, e)
        return {}

    for attempt in range(30):
        time.sleep(min(3 + attempt * 2, 30))
        try:
            status = requests.get(
                f"https://rest.uniprot.org/idmapping/status/{job_id}",
                timeout=10, allow_redirects=False,
            )
            if status.status_code in (303, 302):
                break
            data = status.json()
            js = data.get("jobStatus")
            if js is None or js == "FINISHED":
                break
            if js == "ERROR":
                logger.warning("UniProt job failed (%s): %s", from_db, data)
                return {}
        except requests.RequestException:
            continue
    else:
        logger.warning("UniProt API polling timed out (%s)", from_db)
        return {}

    all_results: list[dict] = []
    next_url = f"https://rest.uniprot.org/idmapping/results/{job_id}?size=500&fields=accession"
    while next_url:
        try:
            r = requests.get(next_url, timeout=120)
            r.raise_for_status()
            rdata = r.json()
            all_results.extend(rdata.get("results", []))
            link = r.headers.get("Link", "")
            next_url = None
            if 'rel="next"' in link:
                next_url = link.split(";")[0].strip("<>")
        except requests.RequestException as e:
            logger.warning("Failed to fetch results page (%s): %s", from_db, e)
            break

    raw_mapping: dict[str, str] = {}
    for result in all_results:
        from_id = result["from"]
        to_entry = result["to"]
        acc = to_entry["primaryAccession"] if isinstance(to_entry, dict) else to_entry
        if from_id not in raw_mapping:
            raw_mapping[from_id] = acc

    mapping = {}
    for unversioned, acc in raw_mapping.items():
        original = unversioned_to_original.get(unversioned, unversioned)
        mapping[original] = acc

    logger.info("Mapped %d / %d %s IDs to UniProt", len(mapping), len(ids), from_db)
    return mapping


def map_accessions_to_uniprot(
    accession_ids: list[str],
    cache_path: Path | None = None,
    gene_symbol_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map mixed-type protein accessions to UniProt.

    Handles RefSeq (NP_*/XP_*), GenBank (AAA*/BAA*/CAA*),
    and already-UniProt (P0C*/Q6X*) accessions.
    Uses UniProt ID Mapping API with caching.

    If gene_symbol_map is provided (accession -> gene_symbol), unmapped
    accessions are retried via Gene_Name mapping as a fallback.
    """
    # Check cache first
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if set(accession_ids).issubset(set(cached.keys())):
            logger.info("Using cached accession->UniProt mapping (%d entries)", len(cached))
            return cached

    # Classify accessions
    refseq_ids = []
    genbank_ids = []
    mapping = {}

    for acc in accession_ids:
        acc_type = _classify_accession(acc)
        if acc_type == "refseq":
            refseq_ids.append(acc)
        elif acc_type == "genbank":
            genbank_ids.append(acc)
        else:  # uniprot
            base = acc.split(".")[0]
            mapping[acc] = base
            logger.info("Already UniProt: %s -> %s", acc, base)

    logger.info(
        "Accession types: %d RefSeq, %d GenBank, %d UniProt",
        len(refseq_ids), len(genbank_ids), len(mapping),
    )

    # RefSeq: download bulk mapping from UniProt (one fast request)
    if refseq_ids:
        refseq_table = _download_refseq_mapping()
        for acc in refseq_ids:
            base = acc.split(".")[0]
            if base in refseq_table:
                mapping[acc] = refseq_table[base]
        logger.info(
            "RefSeq mapped: %d / %d via bulk download",
            sum(1 for a in refseq_ids if a in mapping), len(refseq_ids),
        )

    # GenBank: use ID Mapping API (small batch, reliable)
    if genbank_ids:
        genbank_map = _call_uniprot_idmapping(genbank_ids, "EMBL-GenBank-DDBJ_CDS")
        mapping.update(genbank_map)

    # Fallback: gene symbol search for unmapped accessions
    unmapped = set(accession_ids) - set(mapping.keys())
    if unmapped and gene_symbol_map:
        gene_to_accs: dict[str, list[str]] = {}
        for acc in unmapped:
            gene = gene_symbol_map.get(acc)
            if gene:
                gene_to_accs.setdefault(gene, []).append(acc)

        if gene_to_accs:
            gene_list = list(gene_to_accs.keys())
            logger.info(
                "Gene symbol fallback for %d unmapped accessions (%d unique genes)...",
                len(unmapped), len(gene_list),
            )
            gene_map = _search_uniprot_by_gene(gene_list)
            for gene, uniprot_acc in gene_map.items():
                for acc in gene_to_accs.get(gene, []):
                    if acc not in mapping:
                        mapping[acc] = uniprot_acc

    # Log remaining unmapped
    still_unmapped = set(accession_ids) - set(mapping.keys())
    if still_unmapped:
        logger.warning(
            "Failed to map %d accessions after all strategies",
            len(still_unmapped),
        )

    logger.info("Total mapped: %d / %d accessions", len(mapping), len(accession_ids))

    # Cache results
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(mapping, f, indent=2)

    return mapping


def standardize_all_targets(
    proteins_df: pd.DataFrame,
    refseq_map: dict[str, str],
) -> list[dict]:
    """Standardize all DAVIS protein targets.

    Targets whose RefSeq ID cannot be mapped to UniProt are skipped.
    Canonical UniProt accessions are stored in `targets`; mutation labels
    are retained for later insertion into `target_variants`.
    """
    targets = []
    for _, row in proteins_df.iterrows():
        refseq = row["Accession_Number"]
        gene_name = row["Gene_Name"]
        base_gene, mutation = parse_gene_name(gene_name)

        base_uniprot = refseq_map.get(refseq)
        if base_uniprot is None:
            logger.warning(
                "No UniProt mapping for RefSeq=%s (Gene=%s), skipping",
                refseq, gene_name,
            )
            continue

        seq = row["Sequence"]
        targets.append({
            "protein_index": int(row["Protein_Index"]),
            "uniprot_accession": base_uniprot,
            "gene_symbol": base_gene,
            "amino_acid_sequence": seq,
            "sequence_length": len(seq),
            "target_family": "kinase",
            "variant_label": mutation,
            "raw_gene_name": gene_name,
        })

    return targets


# ============================================================
# TRANSFORM: Affinities
# ============================================================


def classify_affinities(
    affinities_df: pd.DataFrame,
    inactive_threshold: float = 5.0,
    active_threshold: float = 7.0,
) -> pd.DataFrame:
    """Classify DAVIS affinities into inactive/borderline/active.

    pKd <= inactive_threshold -> 'inactive'
    pKd >= active_threshold   -> 'active'
    otherwise                 -> 'borderline'
    """
    import numpy as np

    df = affinities_df.copy()
    conditions = [
        df["Affinity"] <= inactive_threshold,
        df["Affinity"] >= active_threshold,
    ]
    choices = ["inactive", "active"]
    df["classification"] = np.select(conditions, choices, default="borderline")
    return df


# ============================================================
# LOAD
# ============================================================


def insert_compounds(
    conn: sqlite3.Connection,
    compounds: list[dict],
) -> dict[int, int]:
    """Insert standardized compounds into the database.

    Returns dict mapping DAVIS Drug_Index -> database compound_id.
    """
    drug_to_cid = {}
    for comp in compounds:
        conn.execute(
            """INSERT OR IGNORE INTO compounds
            (canonical_smiles, inchikey, inchikey_connectivity, inchi,
             pubchem_cid, molecular_weight, logp, hbd, hba, tpsa,
             rotatable_bonds, num_heavy_atoms, qed, pains_alert,
             lipinski_violations)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                comp["canonical_smiles"],
                comp["inchikey"],
                comp["inchikey_connectivity"],
                comp["inchi"],
                comp["pubchem_cid"],
                comp["molecular_weight"],
                comp["logp"],
                comp["hbd"],
                comp["hba"],
                comp["tpsa"],
                comp["rotatable_bonds"],
                comp["num_heavy_atoms"],
                comp["qed"],
                comp["pains_alert"],
                comp["lipinski_violations"],
            ),
        )
        row = conn.execute(
            "SELECT compound_id FROM compounds WHERE inchikey = ?",
            (comp["inchikey"],),
        ).fetchone()
        drug_to_cid[comp["drug_index"]] = row[0]

    return drug_to_cid


def insert_targets(
    conn: sqlite3.Connection,
    targets: list[dict],
) -> dict[int, int]:
    """Insert standardized targets into the database.

    Returns dict mapping DAVIS Protein_Index -> database target_id.
    """
    prot_to_tid = {}
    for tgt in targets:
        conn.execute(
            """INSERT OR IGNORE INTO targets
            (uniprot_accession, gene_symbol, amino_acid_sequence,
             sequence_length, target_family)
            VALUES (?, ?, ?, ?, ?)""",
            (
                tgt["uniprot_accession"],
                tgt["gene_symbol"],
                tgt["amino_acid_sequence"],
                tgt["sequence_length"],
                tgt["target_family"],
            ),
        )
        row = conn.execute(
            "SELECT target_id FROM targets WHERE uniprot_accession = ?",
            (tgt["uniprot_accession"],),
        ).fetchone()
        prot_to_tid[tgt["protein_index"]] = row[0]

    return prot_to_tid


def insert_target_variants(
    conn: sqlite3.Connection,
    targets: list[dict],
    target_map: dict[int, int],
) -> tuple[dict[int, int], int]:
    """Insert mutation variants into target_variants and map protein_index->variant_id.

    Returns:
        (protein_to_variant_id, unique_variant_rows)
    """
    prot_to_variant: dict[int, int] = {}

    for tgt in targets:
        variant_label = tgt.get("variant_label")
        if not variant_label:
            continue

        protein_index = tgt["protein_index"]
        target_id = target_map.get(protein_index)
        if target_id is None:
            continue

        source_record_id = f"DAVIS:PROTEIN:{protein_index}"
        conn.execute(
            """INSERT OR IGNORE INTO target_variants
            (target_id, variant_label, raw_gene_name, source_db, source_record_id)
            VALUES (?, ?, ?, 'davis', ?)""",
            (target_id, variant_label, tgt.get("raw_gene_name"), source_record_id),
        )
        row = conn.execute(
            """SELECT variant_id FROM target_variants
            WHERE target_id = ? AND variant_label = ?
              AND source_db = 'davis' AND source_record_id = ?""",
            (target_id, variant_label, source_record_id),
        ).fetchone()
        if row is not None:
            prot_to_variant[protein_index] = row[0]

    unique_variants = conn.execute(
        "SELECT COUNT(*) FROM target_variants WHERE source_db = 'davis'"
    ).fetchone()[0]
    return prot_to_variant, unique_variants


def insert_negative_results(
    conn: sqlite3.Connection,
    inactive_df: pd.DataFrame,
    drug_map: dict[int, int],
    target_map: dict[int, int],
    variant_map: dict[int, int] | None = None,
) -> tuple[int, int]:
    """Insert inactive DAVIS results into negative_results table.

    Returns (inserted_count, skipped_count).
    """
    params = []
    skipped = 0
    for _, row in inactive_df.iterrows():
        drug_idx = int(row["Drug_Index"])
        prot_idx = int(row["Protein_Index"])
        pkd = float(row["Affinity"])

        compound_id = drug_map.get(drug_idx)
        target_id = target_map.get(prot_idx)

        if compound_id is None or target_id is None:
            skipped += 1
            continue

        source_record_id = f"DAVIS:{drug_idx}_{prot_idx}"
        variant_id = None
        if variant_map is not None:
            variant_id = variant_map.get(prot_idx)

        params.append((
            compound_id,
            target_id,
            variant_id,
            pkd,
            source_record_id,
        ))

    conn.executemany(
        """INSERT OR IGNORE INTO negative_results
        (compound_id, target_id, variant_id, assay_id,
         result_type, confidence_tier,
         activity_type, activity_value, activity_unit, activity_relation,
         pchembl_value,
         inactivity_threshold, inactivity_threshold_unit,
         source_db, source_record_id, extraction_method,
         curator_validated, publication_year, species_tested)
        VALUES (?, ?, ?, NULL,
                'hard_negative', 'bronze',
                'Kd', 10000.0, 'nM', '>=',
                ?,
                10000.0, 'nM',
                'davis', ?, 'database_direct',
                1, 2011, 'Homo sapiens')""",
        params,
    )

    return len(params), skipped


def refresh_pairs(conn: sqlite3.Connection, source_db: str = "davis") -> int:
    """Refresh compound_target_pairs aggregation for a given source."""
    # Use CASE to map confidence tier rank back to text in the same query
    conn.execute(
        """INSERT OR REPLACE INTO compound_target_pairs
        (compound_id, target_id, num_assays, num_sources,
         best_confidence, best_result_type, earliest_year,
         median_pchembl, min_activity_value, max_activity_value)
        SELECT
            compound_id,
            target_id,
            COUNT(DISTINCT COALESCE(assay_id, -1)),
            COUNT(DISTINCT source_db),
            CASE MIN(CASE confidence_tier
                WHEN 'gold' THEN 1 WHEN 'silver' THEN 2
                WHEN 'bronze' THEN 3 WHEN 'copper' THEN 4 END)
                WHEN 1 THEN 'gold' WHEN 2 THEN 'silver'
                WHEN 3 THEN 'bronze' WHEN 4 THEN 'copper' END,
            MIN(result_type),
            MIN(publication_year),
            AVG(pchembl_value),
            MIN(activity_value),
            MAX(activity_value)
        FROM negative_results
        WHERE source_db = ?
        GROUP BY compound_id, target_id""",
        (source_db,),
    )

    count = conn.execute(
        """SELECT COUNT(*) FROM compound_target_pairs
        WHERE compound_id IN (
            SELECT DISTINCT compound_id FROM negative_results WHERE source_db = ?
        )""",
        (source_db,),
    ).fetchone()[0]
    return count


# ============================================================
# ORCHESTRATOR
# ============================================================


def run_davis_etl(
    db_path: Path,
    data_dir: Path | None = None,
    skip_api: bool = False,
) -> dict:
    """Run the full DAVIS ETL pipeline.

    Returns dict with ETL statistics.
    """
    cfg = load_config()

    if data_dir is None:
        data_dir = _PROJECT_ROOT / cfg["downloads"]["davis"]["dest_dir"]

    inactive_threshold = cfg.get("davis_inactive_pkd_threshold", 5.0)
    active_threshold = cfg.get("davis_active_pkd_threshold", 7.0)

    cache_path = data_dir / "refseq_to_uniprot.json"

    # === EXTRACT ===
    logger.info("Loading DAVIS CSV files from %s", data_dir)
    drugs_df, proteins_df, affinities_df = load_davis_csvs(data_dir)
    logger.info(
        "Loaded: %d drugs, %d proteins, %d affinities",
        len(drugs_df), len(proteins_df), len(affinities_df),
    )

    # === TRANSFORM: Compounds ===
    logger.info("Standardizing compounds with RDKit...")
    compounds = standardize_all_compounds(drugs_df)
    logger.info("Standardized %d / %d compounds", len(compounds), len(drugs_df))

    # === TRANSFORM: Targets ===
    unique_accessions = proteins_df["Accession_Number"].unique().tolist()

    # Build gene symbol map for fallback mapping
    gene_symbol_map = {}
    for _, row in proteins_df.drop_duplicates("Accession_Number").iterrows():
        base_gene, _ = parse_gene_name(row["Gene_Name"])
        gene_symbol_map[row["Accession_Number"]] = base_gene

    if skip_api:
        refseq_map = {}
        if cache_path.exists():
            with open(cache_path) as f:
                refseq_map = json.load(f)
    else:
        refseq_map = map_accessions_to_uniprot(
            unique_accessions, cache_path=cache_path,
            gene_symbol_map=gene_symbol_map,
        )

    targets = standardize_all_targets(proteins_df, refseq_map)
    logger.info(
        "Standardized %d / %d targets (%d unmapped)",
        len(targets), len(proteins_df), len(proteins_df) - len(targets),
    )

    # === TRANSFORM: Affinities ===
    classified = classify_affinities(affinities_df, inactive_threshold, active_threshold)
    n_inactive = (classified["classification"] == "inactive").sum()
    n_active = (classified["classification"] == "active").sum()
    n_borderline = (classified["classification"] == "borderline").sum()
    logger.info(
        "Classification: %d inactive, %d active, %d borderline",
        n_inactive, n_active, n_borderline,
    )

    inactive_df = classified[classified["classification"] == "inactive"]

    # === LOAD ===
    # Ensure DB exists with migrations
    create_database(db_path)

    with connect(db_path) as conn:
        logger.info("Inserting compounds...")
        drug_map = insert_compounds(conn, compounds)

        logger.info("Inserting targets...")
        target_map = insert_targets(conn, targets)
        prot_to_variant, n_variants = insert_target_variants(conn, targets, target_map)

        logger.info("Inserting negative results...")
        total_params, skipped = insert_negative_results(
            conn, inactive_df, drug_map, target_map, variant_map=prot_to_variant,
        )

        logger.info("Refreshing compound-target pairs...")
        n_pairs = refresh_pairs(conn, source_db="davis")

        conn.commit()

    stats = {
        "compounds_inserted": len(compounds),
        "targets_inserted": len(targets),
        "targets_unmapped": len(proteins_df) - len(targets),
        "variants_tracked": n_variants,
        "results_loaded": total_params,
        "results_skipped_unmapped": skipped,
        "results_skipped_active": int(n_active),
        "results_skipped_borderline": int(n_borderline),
        "pairs_created": n_pairs,
    }

    logger.info("DAVIS ETL complete: %s", stats)
    return stats
