"""Example cross-domain research queries for NegBioGraph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from negbiodb_graph.db import DEFAULT_GRAPH_DB_PATH, get_connection


def _rows(conn, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return pd.read_sql_query(query, conn, params=params).to_dict(orient="records")


def run_example_queries(
    graph_db_path: str | Path = DEFAULT_GRAPH_DB_PATH,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a fixed suite of example research queries against the graph DB."""
    conn = get_connection(graph_db_path)
    try:
        result = {
            "domain_summary": _rows(
                conn,
                """
                SELECT domain_code, COUNT(*) AS n_claims
                FROM graph_claims
                GROUP BY domain_code
                ORDER BY domain_code
                """,
            ),
            "contradiction_summary": _rows(
                conn,
                """
                SELECT contradiction_type, claim_family, COUNT(*) AS n_groups
                FROM graph_contradiction_groups
                GROUP BY contradiction_type, claim_family
                ORDER BY contradiction_type, claim_family
                """,
            ),
            "dti_ct_ge_paths": _rows(
                conn,
                """
                WITH dti_ct AS (
                    SELECT
                        d.claim_id AS dti_claim_id,
                        ct.claim_id AS ct_claim_id,
                        m.entity_id AS shared_molecule_id
                    FROM graph_claims d
                    JOIN graph_claim_entities de ON d.claim_id = de.claim_id AND de.role = 'subject'
                    JOIN graph_claims ct ON ct.domain_code = 'ct'
                    JOIN graph_claim_entities cte ON ct.claim_id = cte.claim_id AND cte.role = 'subject_chemical'
                    JOIN graph_entities m ON de.entity_id = m.entity_id AND cte.entity_id = m.entity_id
                    WHERE d.domain_code = 'dti'
                ),
                ct_ge AS (
                    SELECT
                        ct.claim_id AS ct_claim_id,
                        ge.claim_id AS ge_claim_id,
                        g.entity_id AS shared_gene_id
                    FROM graph_claims ct
                    JOIN graph_claim_entities cte ON ct.claim_id = cte.claim_id AND cte.role = 'mediator_gene'
                    JOIN graph_claims ge ON ge.domain_code = 'ge'
                    JOIN graph_claim_entities gee ON ge.claim_id = gee.claim_id AND gee.role = 'subject'
                    JOIN graph_entities g ON cte.entity_id = g.entity_id AND gee.entity_id = g.entity_id
                    WHERE ct.domain_code = 'ct'
                )
                SELECT dti_claim_id, dti_ct.ct_claim_id, ge_claim_id, shared_molecule_id, shared_gene_id
                FROM dti_ct
                JOIN ct_ge ON dti_ct.ct_claim_id = ct_ge.ct_claim_id
                ORDER BY dti_claim_id, ge_claim_id
                LIMIT 50
                """,
            ),
            "dti_cp_paths": _rows(
                conn,
                """
                SELECT
                    d.claim_id AS dti_claim_id,
                    cp.claim_id AS cp_claim_id,
                    e.display_name AS shared_entity_name
                FROM graph_claims d
                JOIN graph_claim_entities de ON d.claim_id = de.claim_id AND de.role = 'subject'
                JOIN graph_claims cp ON cp.domain_code = 'cp'
                JOIN graph_claim_entities cpe ON cp.claim_id = cpe.claim_id AND cpe.role = 'subject'
                JOIN graph_entities e ON de.entity_id = cpe.entity_id AND de.entity_id = e.entity_id
                WHERE d.domain_code = 'dti'
                ORDER BY d.claim_id, cp.claim_id
                LIMIT 50
                """,
            ),
            "ppi_direct_conflicts": _rows(
                conn,
                """
                SELECT group_id, anchor_key, discovery_score
                FROM graph_contradiction_groups
                WHERE claim_family = 'interaction' AND contradiction_type = 'direct_label_conflict'
                ORDER BY discovery_score DESC, group_id
                """,
            ),
            "md_mixed_consensus": _rows(
                conn,
                """
                SELECT group_id, anchor_key, discovery_score
                FROM graph_contradiction_groups
                WHERE claim_family = 'biomarker' AND contradiction_type = 'mixed_consensus'
                ORDER BY discovery_score DESC, group_id
                """,
            ),
            "dc_context_specificity": _rows(
                conn,
                """
                SELECT group_id, anchor_key, discovery_score
                FROM graph_contradiction_groups
                WHERE claim_family = 'combination' AND contradiction_type = 'context_specificity'
                ORDER BY discovery_score DESC, group_id
                """,
            ),
            "vp_submission_conflicts": _rows(
                conn,
                """
                SELECT group_id, anchor_key, discovery_score
                FROM graph_contradiction_groups
                WHERE claim_family = 'classification' AND contradiction_type = 'submission_conflict'
                ORDER BY discovery_score DESC, group_id
                """,
            ),
        }
    finally:
        conn.close()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    return result
