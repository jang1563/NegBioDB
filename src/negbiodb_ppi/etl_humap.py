"""hu.MAP 3.0 negative PPI ETL — ML-derived non-interacting pairs.

Loads negative pairs from hu.MAP 3.0 ComplexPortal-derived pair lists.
File format: tab-separated, 2 columns (UniProt accessions), no header.

All pairs → Silver tier, evidence_type = 'ml_predicted_negative'.

License: CC0 (Public Domain)
"""

from pathlib import Path

from negbiodb_ppi.protein_mapper import canonical_pair, get_or_insert_protein, validate_uniprot


def parse_humap_pair_line(line: str) -> tuple[str, str] | None:
    """Parse tab-separated pair line.

    Format: 'Q9UHC1\tP12345'
    Returns canonical (a, b) pair or None if invalid.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("\t")
    if len(parts) != 2:
        return None
    acc_a = validate_uniprot(parts[0])
    acc_b = validate_uniprot(parts[1])
    if not acc_a or not acc_b or acc_a == acc_b:
        return None
    return canonical_pair(acc_a, acc_b)


def run_humap_etl(
    db_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    neg_files: list[str] | None = None,
) -> dict:
    """Orchestrator: load hu.MAP 3.0 negative pairs.

    Args:
        db_path: Path to PPI database.
        data_dir: Directory containing hu.MAP data files.
        neg_files: List of negative pair filenames.

    Returns:
        Stats dict.
    """
    from negbiodb_ppi.ppi_db import DEFAULT_PPI_DB_PATH, get_connection, run_ppi_migrations

    if db_path is None:
        db_path = DEFAULT_PPI_DB_PATH
    if data_dir is None:
        data_dir = Path(db_path).parent.parent / "data" / "ppi" / "humap"
    if neg_files is None:
        neg_files = [
            "ComplexPortal_reduced_20230309.neg_train_ppis.txt",
            "ComplexPortal_reduced_20230309.neg_test_ppis.txt",
        ]

    db_path = Path(db_path)
    data_dir = Path(data_dir)
    run_ppi_migrations(db_path)

    stats = {
        "lines_total": 0,
        "lines_parsed": 0,
        "lines_skipped": 0,
        "pairs_inserted": 0,
    }

    conn = get_connection(db_path)
    try:
        # Create experiment record
        conn.execute(
            "INSERT OR IGNORE INTO ppi_experiments "
            "(source_db, source_experiment_id, experiment_type, description) "
            "VALUES ('humap', 'humap3-negatives', 'ml_classifier', "
            "'hu.MAP 3.0 ML-derived negative pairs from ComplexPortal training')",
        )
        conn.commit()

        exp_row = conn.execute(
            "SELECT experiment_id FROM ppi_experiments "
            "WHERE source_db = 'humap' AND source_experiment_id = 'humap3-negatives'"
        ).fetchone()
        experiment_id = exp_row[0]

        batch = []
        for neg_file in neg_files:
            file_path = data_dir / neg_file
            with open(file_path) as f:
                for line in f:
                    stats["lines_total"] += 1
                    result = parse_humap_pair_line(line)
                    if result is None:
                        stats["lines_skipped"] += 1
                        continue

                    stats["lines_parsed"] += 1
                    acc_a, acc_b = result
                    pid_a = get_or_insert_protein(conn, acc_a)
                    pid_b = get_or_insert_protein(conn, acc_b)
                    if pid_a > pid_b:
                        pid_a, pid_b = pid_b, pid_a

                    batch.append((
                        pid_a, pid_b, experiment_id,
                        "ml_predicted_negative", "silver",
                        "humap", f"humap-{acc_a}-{acc_b}",
                        "ml_classifier",
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
        stats["pairs_inserted"] = conn.execute(
            "SELECT COUNT(*) FROM ppi_negative_results WHERE source_db = 'humap'"
        ).fetchone()[0]

        # Idempotent dataset_versions
        conn.execute(
            "DELETE FROM dataset_versions WHERE name = 'humap' AND version = '3.0'"
        )
        conn.execute(
            "INSERT INTO dataset_versions (name, version, source_url, row_count, notes) "
            "VALUES ('humap', '3.0', "
            "'https://humap3.proteincomplexes.org/static/downloads/humap3/', ?, "
            "'hu.MAP 3.0 ComplexPortal negative pairs')",
            (stats["pairs_inserted"],),
        )
        conn.commit()

    finally:
        conn.close()

    return stats
