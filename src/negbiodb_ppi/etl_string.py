"""STRING v12.0 negative PPI ETL — zero-score pairs between well-studied proteins.

Derives Bronze-tier negatives from STRING: protein pairs where BOTH proteins
have degree >= min_degree but no interaction evidence across all STRING channels.

Since the STRING links file only contains pairs with score > 0, we must:
  1. Build the set of well-studied proteins (degree >= threshold)
  2. Restrict to a protein universe (e.g., proteins in IntAct/hu.MAP)
  3. Compute Cartesian product minus STRING-linked pairs
  4. Cap at max_pairs to prevent DB bloat

License: CC BY 4.0
"""

import gzip
import itertools
import random
from pathlib import Path

from negbiodb_ppi.protein_mapper import canonical_pair, get_or_insert_protein, validate_uniprot


def load_string_mapping(mapping_path: str | Path) -> dict[str, str]:
    """Load ENSP → UniProt mapping from STRING mapping file.

    File format: 5 columns, tab-separated, NO header
      Col 1: Species taxon ID (e.g., '9606')
      Col 2: 'P31946|1433B_HUMAN' → split on '|' → take [0]
      Col 3: '9606.ENSP00000361930'
      Col 4: Identity %
      Col 5: Bit score

    Returns:
        Dict mapping STRING ID (e.g., '9606.ENSP00000361930') to UniProt accession.
    """
    mapping = {}
    path = Path(mapping_path)

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            # Col 2: 'P31946|1433B_HUMAN'
            uniprot_raw = parts[1].split("|")[0].strip()
            acc = validate_uniprot(uniprot_raw)
            if not acc:
                continue
            # Col 3: STRING ID
            string_id = parts[2].strip()
            if string_id:
                mapping[string_id] = acc

    return mapping


def compute_protein_degrees(
    links_path: str | Path,
    ensp_to_uniprot: dict[str, str] | None = None,
) -> dict[str, int]:
    """Compute degree for each protein from STRING links file.

    File format: space-separated, header: 'protein1 protein2 combined_score'
    IDs: '9606.ENSP00000361930'

    Args:
        links_path: Path to STRING links file (may be .gz).
        ensp_to_uniprot: If provided, returns degrees keyed by UniProt accession.

    Returns:
        Dict mapping protein ID (ENSP or UniProt) to degree.
    """
    degrees: dict[str, int] = {}
    path = Path(links_path)

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("protein"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            p1, p2 = parts[0], parts[1]

            if ensp_to_uniprot is not None:
                p1 = ensp_to_uniprot.get(p1, p1)
                p2 = ensp_to_uniprot.get(p2, p2)
                if p1 == p2:
                    continue  # skip isoform self-links

            degrees[p1] = degrees.get(p1, 0) + 1
            degrees[p2] = degrees.get(p2, 0) + 1

    return degrees


def load_linked_pairs(
    links_path: str | Path,
    ensp_to_uniprot: dict[str, str],
) -> set[tuple[str, str]]:
    """Load all STRING-linked UniProt pairs (score > 0).

    Returns:
        Set of canonical (uniprot_a, uniprot_b) pairs.
    """
    pairs = set()
    path = Path(links_path)

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("protein"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue

            acc_a = ensp_to_uniprot.get(parts[0])
            acc_b = ensp_to_uniprot.get(parts[1])
            if acc_a and acc_b and acc_a != acc_b:
                pairs.add(canonical_pair(acc_a, acc_b))

    return pairs


def extract_zero_score_pairs(
    linked_pairs: set[tuple[str, str]],
    protein_degrees: dict[str, int],
    min_degree: int = 5,
    max_pairs: int = 500_000,
    protein_universe: set[str] | None = None,
    random_seed: int = 42,
) -> list[tuple[str, str]]:
    """Extract protein pairs with no STRING evidence.

    Strategy:
      1. Collect all proteins with degree >= min_degree
      2. If protein_universe provided, intersect with it
      3. Compute Cartesian product of qualifying proteins
      4. Subtract all pairs present in STRING links
      5. Cap at max_pairs (random sample if exceeding)

    Args:
        linked_pairs: Set of canonical pairs from STRING links.
        protein_degrees: Dict mapping UniProt to degree.
        min_degree: Minimum STRING degree to consider "well-studied".
        max_pairs: Maximum number of negative pairs.
        protein_universe: Optional set of proteins to restrict to.
        random_seed: Seed for reproducible sampling.

    Returns:
        Sorted list of (uniprot_a, uniprot_b) canonical pairs.
    """
    # Well-studied proteins
    candidates = {p for p, d in protein_degrees.items() if d >= min_degree}
    # Only keep valid UniProt accessions
    candidates = {p for p in candidates if validate_uniprot(p)}

    if protein_universe is not None:
        candidates = candidates & protein_universe

    sorted_candidates = sorted(candidates)

    # Reservoir sampling over lazy Cartesian product to avoid OOM.
    # For ~15K proteins, set(combinations) would need ~11 GB.
    rng = random.Random(random_seed)
    result = []
    count = 0
    for pair in itertools.combinations(sorted_candidates, 2):
        if pair in linked_pairs:
            continue
        count += 1
        if len(result) < max_pairs:
            result.append(pair)
        else:
            j = rng.randrange(count)
            if j < max_pairs:
                result[j] = pair

    return sorted(result)


def run_string_etl(
    db_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    min_degree: int = 5,
    max_pairs: int = 500_000,
    protein_universe: set[str] | None = None,
) -> dict:
    """Orchestrator: load STRING zero-score negatives.

    Args:
        db_path: Path to PPI database.
        data_dir: Directory containing STRING data files.
        min_degree: Minimum STRING degree for "well-studied".
        max_pairs: Maximum number of negative pairs.
        protein_universe: Optional set of proteins to restrict to.

    Returns:
        Stats dict.
    """
    from negbiodb_ppi.ppi_db import DEFAULT_PPI_DB_PATH, get_connection, run_ppi_migrations

    if db_path is None:
        db_path = DEFAULT_PPI_DB_PATH
    if data_dir is None:
        data_dir = Path(db_path).parent.parent / "data" / "ppi" / "string"

    db_path = Path(db_path)
    data_dir = Path(data_dir)
    run_ppi_migrations(db_path)

    # Find files
    mapping_files = list(data_dir.glob("*.uniprot_2_string*"))
    links_files = list(data_dir.glob("*.protein.links*"))
    if not mapping_files:
        raise FileNotFoundError(f"No STRING mapping file found in {data_dir}")
    if not links_files:
        raise FileNotFoundError(f"No STRING links file found in {data_dir}")

    mapping_path = mapping_files[0]
    links_path = links_files[0]

    # Load mapping
    ensp_to_uniprot = load_string_mapping(mapping_path)

    # Compute degrees in UniProt space
    degrees = compute_protein_degrees(links_path, ensp_to_uniprot)

    # Load linked pairs
    linked = load_linked_pairs(links_path, ensp_to_uniprot)

    # Extract zero-score pairs
    negatives = extract_zero_score_pairs(
        linked, degrees, min_degree, max_pairs, protein_universe
    )

    # Insert into database
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO ppi_experiments "
            "(source_db, source_experiment_id, experiment_type, description) "
            "VALUES ('string', 'string-v12.0-zero-score', 'computational', "
            "'Zero-score pairs between well-studied proteins in STRING v12.0')",
        )
        conn.commit()

        exp_row = conn.execute(
            "SELECT experiment_id FROM ppi_experiments "
            "WHERE source_db = 'string' AND source_experiment_id = 'string-v12.0-zero-score'"
        ).fetchone()
        experiment_id = exp_row[0]

        batch = []
        for acc_a, acc_b in negatives:
            pid_a = get_or_insert_protein(conn, acc_a)
            pid_b = get_or_insert_protein(conn, acc_b)
            if pid_a > pid_b:
                pid_a, pid_b = pid_b, pid_a

            batch.append((
                pid_a, pid_b, experiment_id,
                "low_score_negative", "bronze",
                "string", f"string-{acc_a}-{acc_b}",
                "score_threshold",
            ))

            if len(batch) >= 10000:
                conn.executemany(
                    "INSERT OR IGNORE INTO ppi_negative_results "
                    "(protein1_id, protein2_id, experiment_id, "
                    " evidence_type, confidence_tier, source_db, "
                    " source_record_id, extraction_method) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                batch = []
                conn.commit()

        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO ppi_negative_results "
                "(protein1_id, protein2_id, experiment_id, "
                " evidence_type, confidence_tier, source_db, "
                " source_record_id, extraction_method) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            conn.commit()

        # Query actual inserted count (INSERT OR IGNORE may skip duplicates)
        inserted = conn.execute(
            "SELECT COUNT(*) FROM ppi_negative_results WHERE source_db = 'string'"
        ).fetchone()[0]

        # Idempotent dataset_versions: delete old row before inserting
        conn.execute(
            "DELETE FROM dataset_versions WHERE name = 'string' AND version = 'v12.0'"
        )
        conn.execute(
            "INSERT INTO dataset_versions (name, version, source_url, row_count, notes) "
            "VALUES ('string', 'v12.0', "
            "'https://stringdb-downloads.org/', ?, "
            "'Zero-score pairs between well-studied human proteins')",
            (inserted,),
        )
        conn.commit()

    finally:
        conn.close()

    well_studied = {p for p, d in degrees.items() if d >= min_degree and validate_uniprot(p)}
    if protein_universe is not None:
        well_studied = well_studied & protein_universe

    return {
        "mapping_entries": len(ensp_to_uniprot),
        "linked_pairs": len(linked),
        "proteins_with_degree": len(degrees),
        "well_studied_proteins": len(well_studied),
        "negative_pairs_derived": len(negatives),
        "negative_pairs_inserted": inserted,
    }
