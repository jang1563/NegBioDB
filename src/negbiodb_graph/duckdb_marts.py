"""Materialize DuckDB marts from the graph SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from negbiodb_graph.db import DEFAULT_GRAPH_DB_PATH, DEFAULT_GRAPH_DUCKDB_PATH


def _flatten_claims(conn: sqlite3.Connection) -> pd.DataFrame:
    claims = pd.read_sql_query(
        """
        SELECT
            c.claim_id, c.domain_code, c.claim_family, c.claim_label,
            c.claim_level, c.anchor_key, c.base_anchor_key, c.context_hash, c.context_json,
            COUNT(e.evidence_id) AS n_evidence,
            MIN(e.publication_year) AS min_year,
            MAX(e.publication_year) AS max_year
        FROM graph_claims c
        LEFT JOIN graph_evidence e ON c.claim_id = e.claim_id
        GROUP BY
            c.claim_id, c.domain_code, c.claim_family, c.claim_label,
            c.claim_level, c.anchor_key, c.base_anchor_key, c.context_hash, c.context_json
        """,
        conn,
    )
    entities = pd.read_sql_query(
        """
        SELECT ce.claim_id, ce.role, ce.ordinal, e.entity_id, e.entity_type, e.display_name, e.canonical_key
        FROM graph_claim_entities ce
        JOIN graph_entities e ON ce.entity_id = e.entity_id
        """,
        conn,
    )
    if claims.empty:
        return claims
    role_frame = (
        entities.sort_values(["claim_id", "role", "ordinal"])
        .groupby(["claim_id", "role"])
        .first()
        .reset_index()
    )
    for role in ["subject", "object", "context_cell_line", "mediator_gene", "subject_chemical"]:
        subset = role_frame[role_frame["role"] == role][["claim_id", "entity_id", "display_name", "canonical_key"]]
        subset = subset.rename(
            columns={
                "entity_id": f"{role}_entity_id",
                "display_name": f"{role}_display_name",
                "canonical_key": f"{role}_canonical_key",
            }
        )
        claims = claims.merge(subset, on="claim_id", how="left")
    return claims


def _entity_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    entities = pd.read_sql_query(
        """
        SELECT
            e.entity_id, e.entity_type, e.canonical_key, e.display_name, e.primary_domain,
            COUNT(DISTINCT a.alias_id) AS n_aliases,
            COUNT(DISTINCT ce.claim_id) AS n_claims,
            COUNT(DISTINCT c.domain_code) AS n_domains
        FROM graph_entities e
        LEFT JOIN graph_entity_aliases a ON e.entity_id = a.entity_id
        LEFT JOIN graph_claim_entities ce ON e.entity_id = ce.entity_id
        LEFT JOIN graph_claims c ON ce.claim_id = c.claim_id
        GROUP BY e.entity_id, e.entity_type, e.canonical_key, e.display_name, e.primary_domain
        """,
        conn,
    )
    return entities


def _contradiction_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            g.group_id, g.group_key, g.contradiction_type, g.claim_family,
            g.anchor_key, g.severity_tier, g.discovery_score, g.explanation,
            COUNT(m.claim_id) AS n_claims
        FROM graph_contradiction_groups g
        LEFT JOIN graph_contradiction_members m ON g.group_id = m.group_id
        GROUP BY
            g.group_id, g.group_key, g.contradiction_type, g.claim_family,
            g.anchor_key, g.severity_tier, g.discovery_score, g.explanation
        """,
        conn,
    )


def _cross_domain_paths(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            c1.claim_id AS left_claim_id,
            c1.domain_code AS left_domain,
            c2.claim_id AS right_claim_id,
            c2.domain_code AS right_domain,
            e.entity_id AS shared_entity_id,
            e.entity_type AS shared_entity_type,
            e.display_name AS shared_entity_name,
            e.canonical_key AS shared_entity_key
        FROM graph_claim_entities ce1
        JOIN graph_claim_entities ce2
          ON ce1.entity_id = ce2.entity_id
         AND ce1.claim_id < ce2.claim_id
        JOIN graph_claims c1 ON ce1.claim_id = c1.claim_id
        JOIN graph_claims c2 ON ce2.claim_id = c2.claim_id
        JOIN graph_entities e ON ce1.entity_id = e.entity_id
        WHERE c1.domain_code != c2.domain_code
        """,
        conn,
    )


def materialize_duckdb(
    graph_db_path: str | Path = DEFAULT_GRAPH_DB_PATH,
    duckdb_path: str | Path = DEFAULT_GRAPH_DUCKDB_PATH,
) -> dict[str, Any]:
    """Build DuckDB marts from the graph SQLite database."""
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "duckdb is required to materialize NegBioGraph marts"
        ) from exc

    graph_db_path = Path(graph_db_path)
    duckdb_path = Path(duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    if duckdb_path.exists():
        duckdb_path.unlink()

    sqlite_conn = sqlite3.connect(str(graph_db_path))
    try:
        mart_claims = _flatten_claims(sqlite_conn)
        mart_entity_summary = _entity_summary(sqlite_conn)
        mart_contradictions = _contradiction_summary(sqlite_conn)
        mart_cross_domain_paths = _cross_domain_paths(sqlite_conn)
    finally:
        sqlite_conn.close()

    duck = duckdb.connect(str(duckdb_path))
    try:
        duck.register("mart_claims_df", mart_claims)
        duck.register("mart_entity_summary_df", mart_entity_summary)
        duck.register("mart_contradictions_df", mart_contradictions)
        duck.register("mart_cross_domain_paths_df", mart_cross_domain_paths)
        duck.execute("CREATE TABLE mart_claims AS SELECT * FROM mart_claims_df")
        duck.execute("CREATE TABLE mart_entity_summary AS SELECT * FROM mart_entity_summary_df")
        duck.execute("CREATE TABLE mart_contradictions AS SELECT * FROM mart_contradictions_df")
        duck.execute("CREATE TABLE mart_cross_domain_paths AS SELECT * FROM mart_cross_domain_paths_df")
        duck.commit()
    finally:
        duck.close()

    return {
        "graph_db_path": str(graph_db_path),
        "duckdb_path": str(duckdb_path),
        "mart_claims_rows": int(len(mart_claims)),
        "mart_entity_summary_rows": int(len(mart_entity_summary)),
        "mart_contradictions_rows": int(len(mart_contradictions)),
        "mart_cross_domain_paths_rows": int(len(mart_cross_domain_paths)),
    }
