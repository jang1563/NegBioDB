"""CLI to materialize DuckDB marts from NegBioGraph."""

from __future__ import annotations

import argparse
from pathlib import Path

from negbiodb_graph.db import DEFAULT_GRAPH_DB_PATH, DEFAULT_GRAPH_DUCKDB_PATH
from negbiodb_graph.duckdb_marts import materialize_duckdb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize DuckDB marts from NegBioGraph")
    parser.add_argument("--graph-db", type=Path, default=DEFAULT_GRAPH_DB_PATH, help="SQLite graph DB path")
    parser.add_argument("--duckdb", type=Path, default=DEFAULT_GRAPH_DUCKDB_PATH, help="Output DuckDB path")
    args = parser.parse_args(argv)
    print(materialize_duckdb(args.graph_db, args.duckdb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
