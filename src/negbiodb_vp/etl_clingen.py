"""ClinGen gene-disease validity ETL for NegBioDB VP domain.

Updates genes table with ClinGen validity classifications and
mode of inheritance (MOI) from the ClinGen gene-disease validity CSV.

Source: https://search.clinicalgenome.org/kb/gene-validity/download
"""

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Map ClinGen classification labels to our schema values
_VALIDITY_MAP = {
    "Definitive": "Definitive",
    "Strong": "Strong",
    "Moderate": "Moderate",
    "Limited": "Limited",
    "Disputed": "Disputed",
    "Refuted": "Refuted",
    "No Known Disease Relationship": "No Known Disease Relationship",
    "No Reported Evidence": "No Known Disease Relationship",
}

# Map ClinGen MOI to our schema
_MOI_MAP = {
    "Autosomal dominant": "AD",
    "Autosomal recessive": "AR",
    "X-linked": "XL",
    "X-linked dominant": "XLD",
    "X-linked recessive": "XLR",
    "Mitochondrial": "MT",
    "Semidominant": "AD",
    "Autosomal dominant inheritance": "AD",
    "Autosomal recessive inheritance": "AR",
}


def load_clingen_validity(conn, clingen_csv: Path) -> dict:
    """Load ClinGen gene-disease validity data.

    Updates genes.clingen_validity and genes.gene_moi.
    Uses the highest validity classification per gene if multiple exist.

    Returns stats dict.
    """
    stats = {"genes_updated": 0, "genes_not_found": 0, "rows_parsed": 0}

    # Validity priority (lower = stronger)
    validity_priority = {
        "Definitive": 1,
        "Strong": 2,
        "Moderate": 3,
        "Limited": 4,
        "Disputed": 5,
        "Refuted": 6,
        "No Known Disease Relationship": 7,
    }

    # Build gene lookup
    gene_lookup = {}
    for row in conn.execute("SELECT gene_id, gene_symbol FROM genes"):
        gene_lookup[row[1].upper()] = row[0]

    # Track best validity per gene
    best_validity: dict[int, tuple[str, str | None]] = {}  # gene_id → (validity, moi)

    with open(clingen_csv) as f:
        # Skip metadata/separator rows until we find the actual header
        for line in f:
            stripped = line.strip().strip('"')
            if stripped.startswith("GENE SYMBOL") or stripped.startswith("Gene Symbol"):
                # Parse this line as header
                header = next(csv.reader([line]))
                break
        else:
            logger.warning("ClinGen CSV: no header row found")
            return stats

        reader = csv.DictReader(f, fieldnames=header)
        for row in reader:
            # Skip separator rows
            first_val = next(iter(row.values()), "")
            if first_val and first_val.startswith("+"):
                continue
            stats["rows_parsed"] += 1

            symbol = row.get("GENE SYMBOL", row.get("Gene Symbol", "")).strip().upper()
            if not symbol:
                continue

            gene_id = gene_lookup.get(symbol)
            if gene_id is None:
                stats["genes_not_found"] += 1
                continue

            raw_validity = row.get(
                "CLASSIFICATION", row.get("Classification", "")
            ).strip()
            validity = _VALIDITY_MAP.get(raw_validity)

            raw_moi = row.get("MOI", row.get("Mode of Inheritance", "")).strip()
            moi = _MOI_MAP.get(raw_moi, "Other" if raw_moi else "Unknown")

            if validity:
                current = best_validity.get(gene_id)
                if current is None or validity_priority.get(
                    validity, 99
                ) < validity_priority.get(current[0], 99):
                    best_validity[gene_id] = (validity, moi)

    # Apply updates
    for gene_id, (validity, moi) in best_validity.items():
        conn.execute(
            """UPDATE genes SET
                clingen_validity = ?,
                gene_moi = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            WHERE gene_id = ?""",
            (validity, moi, gene_id),
        )
        stats["genes_updated"] += 1

    conn.commit()
    logger.info(
        "ClinGen: %d genes updated, %d not found",
        stats["genes_updated"],
        stats["genes_not_found"],
    )
    return stats
