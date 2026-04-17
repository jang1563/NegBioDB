"""CLI to build the derived NegBioGraph SQLite database."""

from __future__ import annotations

import argparse
from pathlib import Path

from negbiodb_graph.builder import build_graph
from negbiodb_graph.db import DEFAULT_GRAPH_DB_PATH
from negbiodb_graph.registry import GRAPH_DOMAIN_ORDER


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build NegBioGraph from domain databases")
    parser.add_argument("--graph-db", type=Path, default=DEFAULT_GRAPH_DB_PATH, help="Output SQLite graph DB path")
    parser.add_argument("--manifest", type=Path, default=Path("reference-manifest.json"), help="Reference manifest JSON path")
    parser.add_argument("--strict", action="store_true", help="Fail on missing domain DBs or required feeds")
    parser.add_argument("--build-tag", type=str, default=None, help="Optional build tag")
    for code in GRAPH_DOMAIN_ORDER:
        parser.add_argument(f"--{code}-db", type=Path, default=None, help=f"Override {code.upper()} domain DB path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    domain_paths = {
        code: getattr(args, f"{code}_db")
        for code in GRAPH_DOMAIN_ORDER
        if getattr(args, f"{code}_db") is not None
    }
    manifest_path = args.manifest if args.manifest.exists() else None
    result = build_graph(
        args.graph_db,
        domain_paths=domain_paths or None,
        manifest_path=manifest_path,
        strict=args.strict,
        build_tag=args.build_tag,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
