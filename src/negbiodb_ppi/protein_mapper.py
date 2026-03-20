"""UniProt accession validation and canonical PPI pair ordering."""

import re
from pathlib import Path

# UniProt accession regex (SwissProt + TrEMBL)
# SwissProt: [OPQ][0-9][A-Z0-9]{3}[0-9]  (e.g., P12345, Q9UHC1)
# TrEMBL:   [A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}  (e.g., A0A0K9P0T2)
_UNIPROT_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
    r"|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)


def validate_uniprot(accession: str) -> str | None:
    """Validate and normalize UniProt accession. Strips isoform suffix.

    Returns the accession if valid, None otherwise.

    Examples:
        >>> validate_uniprot("P12345")
        'P12345'
        >>> validate_uniprot("P12345-2")  # isoform stripped
        'P12345'
        >>> validate_uniprot("12345")  # invalid
    """
    if not accession or not isinstance(accession, str):
        return None
    acc = accession.strip().split("-")[0]  # P12345-2 → P12345
    if not acc:
        return None
    return acc if _UNIPROT_RE.match(acc) else None


def load_ensg_mapping(mapping_path: str | Path) -> dict[str, str]:
    """Load Ensembl gene ID → UniProt accession mapping.

    Reads a filtered TSV file with two columns: ENSG_ID and UniProt_AC.
    No header expected. Lines starting with '#' are skipped.

    For one-to-many mappings, the first occurrence is kept (typically
    SwissProt before TrEMBL).

    Args:
        mapping_path: Path to filtered ENSG→UniProt TSV file.

    Returns:
        Dict mapping ENSG IDs to UniProt accessions.
    """
    mapping: dict[str, str] = {}
    with open(mapping_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            ensg_id = parts[0].strip()
            acc = validate_uniprot(parts[1].strip())
            # Keep first occurrence (prefer SwissProt)
            if acc and ensg_id not in mapping:
                mapping[ensg_id] = acc
    return mapping


def canonical_pair(acc1: str, acc2: str) -> tuple[str, str]:
    """Return (smaller, larger) alphabetically. Enforces protein1 < protein2.

    Args:
        acc1: First UniProt accession.
        acc2: Second UniProt accession.

    Returns:
        Tuple with accessions in canonical (sorted) order.
    """
    return (acc1, acc2) if acc1 < acc2 else (acc2, acc1)


def get_or_insert_protein(
    conn,
    accession: str,
    gene_symbol: str | None = None,
    sequence: str | None = None,
    organism: str = "Homo sapiens",
) -> int:
    """Insert protein if not exists, return protein_id.

    Args:
        conn: SQLite connection to PPI database.
        accession: UniProt accession (must be pre-validated).
        gene_symbol: HGNC gene symbol (optional).
        sequence: Amino acid sequence (optional).
        organism: Organism name (default: Homo sapiens).

    Returns:
        protein_id from the proteins table.
    """
    row = conn.execute(
        "SELECT protein_id FROM proteins WHERE uniprot_accession = ?",
        (accession,),
    ).fetchone()
    if row:
        return row[0]
    cur = conn.execute(
        """INSERT INTO proteins (uniprot_accession, gene_symbol,
           amino_acid_sequence, sequence_length, organism)
        VALUES (?, ?, ?, ?, ?)""",
        (
            accession,
            gene_symbol,
            sequence,
            len(sequence) if sequence else None,
            organism,
        ),
    )
    return cur.lastrowid
