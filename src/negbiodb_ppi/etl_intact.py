"""IntAct negative PPI ETL — experimentally reported non-interactions.

Parses IntAct's pre-filtered negative interaction file (PSI-MI TAB 2.7).
Assigns confidence tiers based on detection method:
  - Gold: co-immunoprecipitation, pull down, x-ray, cross-linking, etc.
  - Silver: two hybrid or other indirect methods

License: CC BY 4.0
"""

import re
from pathlib import Path

from negbiodb_ppi.protein_mapper import canonical_pair, get_or_insert_protein, validate_uniprot


def _parse_uniprot_id(col: str) -> str | None:
    """Parse UniProt accession from MITAB column 1 or 2.

    Format: "uniprotkb:P12346" or "uniprotkb:P12346-2" (isoform)
    Multi-value: separated by "|"
    """
    for part in col.split("|"):
        part = part.strip()
        if part.startswith("uniprotkb:"):
            raw = part.split(":")[1]
            return validate_uniprot(raw)
    return None


def _parse_taxon_id(col: str) -> int | None:
    """Parse taxonomy ID from MITAB column 10 or 11.

    Format: "taxid:9606(Homo sapiens)" or "taxid:9606(human)"
    """
    m = re.match(r"taxid:(\d+)", col.strip())
    return int(m.group(1)) if m else None


def _parse_mi_id(col: str) -> str | None:
    """Parse MI ontology ID from MITAB detection method column.

    Format: 'psi-mi:"MI:0018"(two hybrid)'
    """
    m = re.search(r'"(MI:\d+)"', col)
    return m.group(1) if m else None


def _parse_mi_term(col: str) -> str | None:
    """Parse MI ontology term name from MITAB column.

    Format: 'psi-mi:"MI:0018"(two hybrid)' → 'two hybrid'
    Returns the FIRST parenthesized term (consistent with _parse_mi_id).
    """
    m = re.search(r"\(([^)]+)\)", col)
    return m.group(1) if m else None


def _parse_pubmed(col: str) -> int | None:
    """Parse PubMed ID from MITAB publication column.

    Format: "pubmed:12345678" or "pubmed:12345678|pubmed:99999"
    """
    for part in col.split("|"):
        part = part.strip()
        if part.startswith("pubmed:"):
            try:
                return int(part.split(":")[1])
            except ValueError:
                continue
    return None


def _parse_miscore(col: str) -> float | None:
    """Parse intact-miscore from MITAB confidence column.

    Format: "intact-miscore:0.56"
    """
    for part in col.split("|"):
        part = part.strip()
        if part.startswith("intact-miscore:"):
            try:
                return float(part.split(":")[1])
            except ValueError:
                return None
    return None


# Detection methods considered Gold-tier (direct physical evidence)
_GOLD_METHODS = frozenset({
    "MI:0004",  # affinity chromatography technology
    "MI:0006",  # anti bait coimmunoprecipitation
    "MI:0019",  # coimmunoprecipitation
    "MI:0030",  # cross-linking study
    "MI:0096",  # pull down
    "MI:0114",  # x-ray crystallography
    "MI:0071",  # molecular sieving
    "MI:0676",  # tandem affinity purification
})


def classify_tier(detection_method_id: str | None) -> str:
    """Map detection method MI ID to confidence tier."""
    if detection_method_id and detection_method_id in _GOLD_METHODS:
        return "gold"
    return "silver"


def parse_mitab_line(line: str) -> dict | None:
    """Parse one PSI-MI TAB 2.7 line into structured dict.

    Returns None if:
      - Line has fewer than 36 columns
      - Column 36 (negative flag) is not "true"
      - Either interactor is not a UniProt protein
    """
    cols = line.rstrip("\n").split("\t")
    if len(cols) < 36:
        return None

    # Column 36 (0-indexed: 35) is the negative flag
    neg_flag = cols[35].strip().lower()
    if neg_flag != "true":
        return None

    uniprot_a = _parse_uniprot_id(cols[0])
    uniprot_b = _parse_uniprot_id(cols[1])
    if not uniprot_a or not uniprot_b:
        return None

    return {
        "uniprot_a": uniprot_a,
        "uniprot_b": uniprot_b,
        "detection_method": _parse_mi_term(cols[6]),
        "detection_method_id": _parse_mi_id(cols[6]),
        "taxon_a": _parse_taxon_id(cols[9]),
        "taxon_b": _parse_taxon_id(cols[10]),
        "interaction_type": _parse_mi_term(cols[11]) if len(cols) > 11 else None,
        "pubmed_id": _parse_pubmed(cols[8]),
        "mi_score": _parse_miscore(cols[14]) if len(cols) > 14 else None,
        "interaction_id": cols[13].strip() if len(cols) > 13 else None,
    }


def run_intact_etl(
    db_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    filename: str = "intact_negative.txt",
    human_only: bool = True,
) -> dict:
    """Orchestrator: load IntAct negatives into PPI database.

    Args:
        db_path: Path to PPI database.
        data_dir: Directory containing IntAct data files.
        filename: Name of the negative interactions file.
        human_only: If True, filter to human-human interactions only.

    Returns:
        Stats dict.
    """
    from negbiodb_ppi.ppi_db import DEFAULT_PPI_DB_PATH, get_connection, run_ppi_migrations

    if db_path is None:
        db_path = DEFAULT_PPI_DB_PATH
    if data_dir is None:
        data_dir = Path(db_path).parent.parent / "data" / "ppi" / "intact"

    db_path = Path(db_path)
    data_dir = Path(data_dir)
    run_ppi_migrations(db_path)

    file_path = data_dir / filename
    stats = {
        "lines_total": 0,
        "lines_parsed": 0,
        "lines_skipped_comment": 0,
        "lines_skipped_short": 0,
        "lines_skipped_non_negative": 0,
        "lines_skipped_no_uniprot": 0,
        "lines_skipped_non_human": 0,
        "lines_skipped_self_interaction": 0,
        "pairs_gold": 0,
        "pairs_silver": 0,
        "pairs_inserted": 0,
    }

    conn = get_connection(db_path)
    try:
        rows_processed = 0
        with open(file_path) as f:
            for line in f:
                stats["lines_total"] += 1
                if line.startswith("#"):
                    stats["lines_skipped_comment"] += 1
                    continue

                cols = line.rstrip("\n").split("\t")
                if len(cols) < 36:
                    stats["lines_skipped_short"] += 1
                    continue

                neg_flag = cols[35].strip().lower()
                if neg_flag != "true":
                    stats["lines_skipped_non_negative"] += 1
                    continue

                parsed = parse_mitab_line(line)
                if parsed is None:
                    stats["lines_skipped_no_uniprot"] += 1
                    continue

                # Filter human-human
                if human_only:
                    if parsed["taxon_a"] != 9606 or parsed["taxon_b"] != 9606:
                        stats["lines_skipped_non_human"] += 1
                        continue

                acc_a, acc_b = canonical_pair(parsed["uniprot_a"], parsed["uniprot_b"])
                if acc_a == acc_b:
                    stats["lines_skipped_self_interaction"] += 1
                    continue

                stats["lines_parsed"] += 1
                tier = classify_tier(parsed["detection_method_id"])
                if tier == "gold":
                    stats["pairs_gold"] += 1
                else:
                    stats["pairs_silver"] += 1

                # Insert or get experiment
                raw_id = parsed["interaction_id"]
                exp_id_str = (
                    raw_id
                    if raw_id and raw_id != "-"
                    else f"intact-{acc_a}-{acc_b}"
                )
                conn.execute(
                    "INSERT OR IGNORE INTO ppi_experiments "
                    "(source_db, source_experiment_id, experiment_type, "
                    " detection_method, detection_method_id, pubmed_id) "
                    "VALUES ('intact', ?, 'negative_interaction', ?, ?, ?)",
                    (
                        exp_id_str,
                        parsed["detection_method"],
                        parsed["detection_method_id"],
                        parsed["pubmed_id"],
                    ),
                )
                exp_row = conn.execute(
                    "SELECT experiment_id FROM ppi_experiments "
                    "WHERE source_db = 'intact' AND source_experiment_id = ?",
                    (exp_id_str,),
                ).fetchone()
                experiment_id = exp_row[0]

                # Insert proteins
                pid_a = get_or_insert_protein(conn, acc_a)
                pid_b = get_or_insert_protein(conn, acc_b)
                if pid_a > pid_b:
                    pid_a, pid_b = pid_b, pid_a

                # Insert negative result
                conn.execute(
                    "INSERT OR IGNORE INTO ppi_negative_results "
                    "(protein1_id, protein2_id, experiment_id, evidence_type, "
                    " confidence_tier, interaction_score, source_db, "
                    " source_record_id, extraction_method, publication_year) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'intact', ?, 'database_direct', NULL)",
                    (
                        pid_a,
                        pid_b,
                        experiment_id,
                        "experimental_non_interaction",
                        tier,
                        parsed["mi_score"],
                        exp_id_str,
                    ),
                )
                rows_processed += 1

                # Periodic commit to avoid large uncommitted transactions
                if rows_processed % 5000 == 0:
                    conn.commit()

        conn.commit()

        # Query actual inserted count (INSERT OR IGNORE may skip duplicates)
        inserted = conn.execute(
            "SELECT COUNT(*) FROM ppi_negative_results WHERE source_db = 'intact'"
        ).fetchone()[0]
        stats["pairs_inserted"] = inserted

        # Idempotent dataset_versions
        conn.execute(
            "DELETE FROM dataset_versions "
            "WHERE name = 'intact_negative' AND version = 'current'"
        )
        conn.execute(
            "INSERT INTO dataset_versions (name, version, source_url, row_count, notes) "
            "VALUES ('intact_negative', 'current', "
            "'https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/intact_negative.txt', "
            "?, 'IntAct curated negative interactions')",
            (inserted,),
        )
        conn.commit()

    finally:
        conn.close()

    return stats
