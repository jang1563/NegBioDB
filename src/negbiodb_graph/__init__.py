"""NegBioGraph package: derived cross-domain evidence graph for NegBioDB."""

from negbiodb_graph.builder import build_graph
from negbiodb_graph.contradictions import build_contradictions
from negbiodb_graph.db import (
    DEFAULT_GRAPH_DB_PATH,
    DEFAULT_GRAPH_DUCKDB_PATH,
    create_graph_database,
    get_connection,
)
from negbiodb_graph.duckdb_marts import materialize_duckdb
from negbiodb_graph.queries import run_example_queries

__all__ = [
    "DEFAULT_GRAPH_DB_PATH",
    "DEFAULT_GRAPH_DUCKDB_PATH",
    "build_graph",
    "build_contradictions",
    "create_graph_database",
    "get_connection",
    "materialize_duckdb",
    "run_example_queries",
]
