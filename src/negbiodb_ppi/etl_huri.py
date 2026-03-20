"""HuRI Y2H-derived negative PPI ETL — systematic screen non-interactions.

Derives high-confidence PPI negatives from the HuRI systematic Y2H screen
(Luck et al., Nature 2020). Strategy:
  1. Load HuRI positive PPIs (HI-union.tsv, Ensembl gene IDs → UniProt)
  2. Load ORFeome v9.1 protein list (the Y2H search space)
  3. Identify Y2H-viable proteins (appear in >= 1 positive PPI)
  4. Compute Cartesian product of viable proteins minus positives
  5. All resulting pairs → Gold tier

License: CC BY 4.0 (interactome-atlas.org)
"""

import itertools
import random
from pathlib import Path

from negbiodb_ppi.protein_mapper import (
    canonical_pair,
    get_or_insert_protein,
    load_ensg_mapping,
    validate_uniprot,
)


def load_huri_positives(
    ppi_path: str | Path,
    ensg_mapping: dict[str, str] | None = None,
) -> set[tuple[str, str]]:
    """Load HuRI positive PPIs as canonical UniProt pairs.

    File format: tab-separated, 2 columns, NO header.
    IDs may be Ensembl gene IDs (ENSG*) or UniProt accessions.
    When ENSG IDs are found and ensg_mapping is provided, IDs are
    translated to UniProt before validation.

    Args:
        ppi_path: Path to HuRI PPI file.
        ensg_mapping: Optional ENSG→UniProt mapping dict.

    Returns:
        Set of (uniprot_a, uniprot_b) with a < b.
    """
    pairs = set()
    unmapped = 0
    path = Path(ppi_path)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            id_a, id_b = parts[0].strip(), parts[1].strip()

            # Try direct UniProt validation first, then ENSG fallback
            acc_a = validate_uniprot(id_a)
            if not acc_a and ensg_mapping:
                acc_a = ensg_mapping.get(id_a)

            acc_b = validate_uniprot(id_b)
            if not acc_b and ensg_mapping:
                acc_b = ensg_mapping.get(id_b)

            if acc_a and acc_b and acc_a != acc_b:
                pairs.add(canonical_pair(acc_a, acc_b))
            else:
                unmapped += 1

    if unmapped > 0:
        print(f"  HuRI: {unmapped} pairs skipped (unmapped IDs)")
    return pairs


def load_orfeome_proteins(orfeome_path: str | Path) -> set[str]:
    """Load ORFeome v9.1 gene list → UniProt accessions.

    Accepts a file with one protein identifier per line (or tab-separated
    with UniProt in the first column). Lines starting with '#' are skipped.

    Returns:
        Set of validated UniProt accessions.
    """
    proteins = set()
    path = Path(orfeome_path)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Take first column (tab or comma separated)
            token = line.split("\t")[0].split(",")[0].strip()
            acc = validate_uniprot(token)
            if acc:
                proteins.add(acc)
    return proteins


def get_y2h_viable_proteins(
    orfeome_proteins: set[str],
    positive_pairs: set[tuple[str, str]],
) -> set[str]:
    """Filter ORFeome proteins to those appearing in at least one positive PPI.

    A protein that appears as positive with ANY partner is confirmed
    Y2H-viable (expressed, folded, and functional in the assay).
    Proteins with zero positives may have failed for technical reasons.

    Args:
        orfeome_proteins: Set of UniProt accessions from ORFeome.
        positive_pairs: Set of canonical positive PPI pairs.

    Returns:
        Set of UniProt accessions that are Y2H-viable.
    """
    viable = set()
    for a, b in positive_pairs:
        if a in orfeome_proteins:
            viable.add(a)
        if b in orfeome_proteins:
            viable.add(b)
    return viable


def derive_huri_negatives(
    viable_proteins: set[str],
    positive_pairs: set[tuple[str, str]],
    max_pairs: int | None = None,
    random_seed: int = 42,
) -> list[tuple[str, str]]:
    """Derive high-confidence Y2H negatives from HuRI screen.

    Uses reservoir sampling to avoid materializing the full Cartesian product
    when max_pairs is set (for large protein sets, C(N,2) can be millions).

    Algorithm:
      1. Lazily iterate canonical pairs from viable_proteins
      2. Skip positive_pairs
      3. Collect up to max_pairs via reservoir sampling

    Args:
        viable_proteins: Set of Y2H-viable UniProt accessions.
        positive_pairs: Set of canonical positive PPI pairs to exclude.
        max_pairs: Maximum number of negative pairs (None = keep all).
        random_seed: Seed for reproducible sampling.

    Returns:
        Sorted list of (uniprot_a, uniprot_b) canonical negative pairs.
    """
    sorted_proteins = sorted(viable_proteins)
    rng = random.Random(random_seed)
    result = []
    count = 0

    for pair in itertools.combinations(sorted_proteins, 2):
        if pair in positive_pairs:
            continue
        count += 1
        if max_pairs is None or len(result) < max_pairs:
            result.append(pair)
        else:
            j = rng.randrange(count)
            if j < max_pairs:
                result[j] = pair

    return sorted(result)


def run_huri_etl(
    db_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    max_pairs: int | None = None,
    ppi_file: str = "HI-union.tsv",
    orfeome_file: str | None = None,
    mapping_file: str | None = None,
) -> dict:
    """Orchestrator: derive and load HuRI Y2H negatives.

    Args:
        db_path: Path to PPI database (default from ppi_db).
        data_dir: Directory containing HuRI data files.
        max_pairs: Maximum number of negative pairs to insert.
        ppi_file: Name of HuRI positive PPI file.
        orfeome_file: Name of ORFeome gene list file.
            If None, viable proteins are derived from positive PPIs only.
        mapping_file: Name of ENSG→UniProt mapping TSV file.
            If None, auto-detects 'ensg_to_uniprot.tsv' in data_dir.

    Returns:
        Stats dict with counts.
    """
    from negbiodb_ppi.ppi_db import DEFAULT_PPI_DB_PATH, get_connection, run_ppi_migrations

    if db_path is None:
        db_path = DEFAULT_PPI_DB_PATH
    if data_dir is None:
        data_dir = Path(db_path).parent.parent / "data" / "ppi" / "huri"

    db_path = Path(db_path)
    data_dir = Path(data_dir)

    # Ensure DB exists
    run_ppi_migrations(db_path)

    # Load optional ENSG→UniProt mapping
    ensg_mapping = None
    if mapping_file:
        mapping_path = data_dir / mapping_file
    else:
        mapping_path = data_dir / "ensg_to_uniprot.tsv"
    if mapping_path.exists():
        ensg_mapping = load_ensg_mapping(mapping_path)
        print(f"  Loaded {len(ensg_mapping)} ENSG→UniProt mappings")

    # Load positive PPIs
    ppi_path = data_dir / ppi_file
    positive_pairs = load_huri_positives(ppi_path, ensg_mapping=ensg_mapping)

    # Load ORFeome or derive protein universe from positives
    if orfeome_file:
        orfeome_proteins = load_orfeome_proteins(data_dir / orfeome_file)
    else:
        # Fallback: use all proteins appearing in positives
        orfeome_proteins = set()
        for a, b in positive_pairs:
            orfeome_proteins.add(a)
            orfeome_proteins.add(b)

    # Get Y2H-viable proteins
    viable = get_y2h_viable_proteins(orfeome_proteins, positive_pairs)

    # Derive negatives
    negatives = derive_huri_negatives(viable, positive_pairs, max_pairs)

    # Insert into database
    conn = get_connection(db_path)
    try:
        # Create experiment record
        conn.execute(
            "INSERT OR IGNORE INTO ppi_experiments "
            "(source_db, source_experiment_id, experiment_type, "
            " detection_method, pubmed_id, description) "
            "VALUES ('huri', 'HI-union-negatives', 'systematic_y2h', "
            "'two hybrid', 32296183, "
            "'Y2H-derived negatives from HuRI (Luck et al. 2020)')",
        )
        conn.commit()

        exp_row = conn.execute(
            "SELECT experiment_id FROM ppi_experiments "
            "WHERE source_db = 'huri' AND source_experiment_id = 'HI-union-negatives'"
        ).fetchone()
        experiment_id = exp_row[0]

        # Pre-cache all protein IDs to avoid per-pair SELECT queries
        pid_cache: dict[str, int] = {}
        unique_accs = set()
        for a, b in negatives:
            unique_accs.add(a)
            unique_accs.add(b)
        for acc in sorted(unique_accs):
            pid_cache[acc] = get_or_insert_protein(conn, acc)
        conn.commit()

        # Batch insert negatives
        batch = []
        for acc_a, acc_b in negatives:
            pid_a = pid_cache[acc_a]
            pid_b = pid_cache[acc_b]
            # Ensure canonical ordering by protein_id
            if pid_a > pid_b:
                pid_a, pid_b = pid_b, pid_a
            batch.append((
                pid_a, pid_b, experiment_id,
                "experimental_non_interaction", "gold",
                "huri", f"huri-neg-{acc_a}-{acc_b}",
                "database_direct", 2020,
            ))
            if len(batch) >= 10000:
                conn.executemany(
                    "INSERT OR IGNORE INTO ppi_negative_results "
                    "(protein1_id, protein2_id, experiment_id, evidence_type, "
                    " confidence_tier, source_db, source_record_id, "
                    " extraction_method, publication_year) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                batch = []
                conn.commit()

        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO ppi_negative_results "
                "(protein1_id, protein2_id, experiment_id, evidence_type, "
                " confidence_tier, source_db, source_record_id, "
                " extraction_method, publication_year) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            conn.commit()

        # Query actual inserted count (INSERT OR IGNORE may skip duplicates)
        inserted = conn.execute(
            "SELECT COUNT(*) FROM ppi_negative_results WHERE source_db = 'huri'"
        ).fetchone()[0]

        # Idempotent dataset_versions
        conn.execute(
            "DELETE FROM dataset_versions WHERE name = 'huri' AND version = '2020'"
        )
        conn.execute(
            "INSERT INTO dataset_versions (name, version, source_url, row_count, notes) "
            "VALUES ('huri', '2020', 'https://www.interactome-atlas.org/download', ?, "
            "'Y2H-derived negatives: viable proteins only')",
            (inserted,),
        )
        conn.commit()

    finally:
        conn.close()

    return {
        "orfeome_total": len(orfeome_proteins),
        "viable_proteins": len(viable),
        "positive_pairs": len(positive_pairs),
        "candidate_pairs": len(viable) * (len(viable) - 1) // 2,
        "negative_pairs_derived": len(negatives),
        "negative_pairs_inserted": inserted,
    }
