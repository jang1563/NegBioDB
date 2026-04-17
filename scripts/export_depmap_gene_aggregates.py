#!/usr/bin/env python3
"""Export per-gene DepMap Chronos aggregates from negbiodb_depmap.db.

Run on HPC (where the 16 GB negbiodb_depmap.db lives). Produces a small parquet
(~2-5 MB for ~18K genes) that can be rsync'd locally for 2B analysis.

Computes per-gene:
  - chronos_median: median Chronos across all cell lines (primary — robust)
  - chronos_mean:   mean Chronos (secondary)
  - chronos_min:    most-lethal Chronos (tertiary)
  - n_cell_lines:   number of cell lines with measured Chronos for this gene

Median is computed via a pure-SQL ORDER BY + GROUP_CONCAT trick (no full raw pull).
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Export DepMap per-gene Chronos aggregates")
    ap.add_argument("--db", default="/athena/masonlab/scratch/users/jak4013/negbiodb/data/negbiodb_depmap.db")
    ap.add_argument("--out", default="exports/ge_gene_aggregates.parquet")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _log(f"Opening DB: {args.db}")
    conn = sqlite3.connect(args.db, timeout=600)
    # Speed up large reads
    conn.execute("PRAGMA cache_size = -2000000")  # 2 GB page cache
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 8000000000")  # 8 GB mmap

    _log("Discovering schema")
    ge_cols = [r[1] for r in conn.execute("PRAGMA table_info('ge_negative_results')")]
    _log(f"  ge_negative_results cols: {ge_cols}")
    genes_cols = [r[1] for r in conn.execute("PRAGMA table_info('genes')")]
    _log(f"  genes cols: {genes_cols}")

    chronos_col = None
    for candidate in ("gene_effect_score", "chronos_effect", "effect",
                      "chronos", "gene_effect", "dependency_score"):
        if candidate in ge_cols:
            chronos_col = candidate
            break
    if chronos_col is None:
        raise SystemExit(f"No Chronos column found in ge_negative_results: {ge_cols}")
    _log(f"Using Chronos column: {chronos_col}")

    # ── Pass 1: mean / min / count via streaming aggregation ─────────────────
    _log("Pass 1: GROUP BY gene_id for mean/min/count (streaming)")
    t0 = time.time()
    agg_df = pd.read_sql_query(f"""
        SELECT gene_id,
               AVG({chronos_col}) AS chronos_mean,
               MIN({chronos_col}) AS chronos_min,
               COUNT(*) AS n_cell_lines
        FROM ge_negative_results
        WHERE {chronos_col} IS NOT NULL
        GROUP BY gene_id
    """, conn)
    _log(f"  Pass 1 done in {time.time()-t0:.1f}s: {len(agg_df):,} genes")

    # ── Pass 2: median per gene (streamed via cursor) ────────────────────────
    # For each gene, pull ordered Chronos values and compute median in-line.
    # This avoids pulling all 28M rows at once and avoids SQLite MEDIAN unavailability.
    _log("Pass 2: median per gene (cursor-based streaming)")
    t0 = time.time()
    gene_ids = agg_df["gene_id"].tolist()

    # Single streaming cursor ordered by (gene_id, chronos) — compute median per gene as we iterate
    cursor = conn.execute(f"""
        SELECT gene_id, {chronos_col} AS chronos
        FROM ge_negative_results
        WHERE {chronos_col} IS NOT NULL
        ORDER BY gene_id, {chronos_col}
    """)

    medians: dict[int, float] = {}
    current_gene: int | None = None
    values: list[float] = []
    n_processed = 0
    for gid, chronos in cursor:
        if gid != current_gene:
            if current_gene is not None and values:
                m = len(values)
                if m % 2 == 0:
                    med = 0.5 * (values[m // 2 - 1] + values[m // 2])
                else:
                    med = values[m // 2]
                medians[current_gene] = med
                n_processed += 1
                if n_processed % 2000 == 0:
                    _log(f"  medians computed: {n_processed:,} genes ({time.time()-t0:.1f}s)")
            current_gene = gid
            values = []
        values.append(float(chronos))
    # Flush last gene
    if current_gene is not None and values:
        m = len(values)
        if m % 2 == 0:
            med = 0.5 * (values[m // 2 - 1] + values[m // 2])
        else:
            med = values[m // 2]
        medians[current_gene] = med
        n_processed += 1

    _log(f"  Pass 2 done in {time.time()-t0:.1f}s: {n_processed:,} gene medians computed")

    # Attach medians
    agg_df["chronos_median"] = agg_df["gene_id"].map(medians)

    # Attach gene metadata
    _log("Joining with genes table")
    genes_df = pd.read_sql_query(
        "SELECT gene_id, gene_symbol, entrez_id FROM genes", conn
    )
    final = agg_df.merge(genes_df, on="gene_id", how="left")
    final = final[["gene_id", "gene_symbol", "entrez_id",
                   "chronos_median", "chronos_mean", "chronos_min",
                   "n_cell_lines"]]
    final = final.dropna(subset=["chronos_median"])

    _log(f"Final rows: {len(final):,}")
    final.to_parquet(out_path, index=False)
    _log(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    _log("Sample:")
    print(final.head(5).to_string(index=False), flush=True)
    conn.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
