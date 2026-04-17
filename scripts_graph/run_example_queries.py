"""CLI to run the fixed example research queries for NegBioGraph."""

from __future__ import annotations

import argparse
from pathlib import Path

from negbiodb_graph.db import DEFAULT_GRAPH_DB_PATH
from negbiodb_graph.queries import run_example_queries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run example queries against NegBioGraph")
    parser.add_argument("--graph-db", type=Path, default=DEFAULT_GRAPH_DB_PATH, help="SQLite graph DB path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "negbiodb_graph_example_queries.json",
        help="JSON output path",
    )
    args = parser.parse_args(argv)
    print(run_example_queries(args.graph_db, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
