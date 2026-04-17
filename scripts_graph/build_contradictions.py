"""CLI to populate NegBioGraph contradiction tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from negbiodb_graph.contradictions import build_contradictions
from negbiodb_graph.db import DEFAULT_GRAPH_DB_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build contradiction groups for NegBioGraph")
    parser.add_argument("--graph-db", type=Path, default=DEFAULT_GRAPH_DB_PATH, help="SQLite graph DB path")
    args = parser.parse_args(argv)
    print(build_contradictions(args.graph_db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
