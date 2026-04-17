"""Contradiction and tension generation for NegBioGraph."""

from __future__ import annotations

import json
import sqlite3
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from negbiodb_graph.db import DEFAULT_GRAPH_DB_PATH, get_connection
from negbiodb_graph.utils import stable_json_dumps


INCOMPATIBLE_LABELS: dict[str, set[frozenset[str]]] = {
    "binding": {frozenset({"active_against", "inactive_against"})},
    "trial_outcome": {frozenset({"successful_for", "failed_for"})},
    "interaction": {frozenset({"interacts_with", "non_interacting_with"})},
    "essentiality": {frozenset({"essential_in", "nonessential_in"})},
    "biomarker": {frozenset({"differential_in", "not_differential_in"})},
    "combination": {
        frozenset({"synergistic_in", "antagonistic_in"}),
        frozenset({"synergistic_in", "additive_in"}),
        frozenset({"antagonistic_in", "additive_in"}),
    },
    "phenotype": {
        frozenset({"inactive_in", "weak_phenotype_in"}),
        frozenset({"inactive_in", "strong_phenotype_in"}),
        frozenset({"inactive_in", "artifact_or_toxic_in"}),
        frozenset({"weak_phenotype_in", "strong_phenotype_in"}),
        frozenset({"weak_phenotype_in", "artifact_or_toxic_in"}),
        frozenset({"strong_phenotype_in", "artifact_or_toxic_in"}),
    },
}

SEVERITY_BY_TYPE = {
    "direct_label_conflict": "high",
    "temporal_revision": "high",
    "submission_conflict": "high",
    "context_specificity": "medium",
    "source_or_batch_disagreement": "medium",
    "mixed_consensus": "medium",
}


def _load_claim_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    claims = pd.read_sql_query(
        """
        SELECT
            c.claim_id, c.domain_code, c.claim_family, c.claim_label,
            c.anchor_key, c.base_anchor_key, c.context_hash, c.context_json,
            COUNT(e.evidence_id) AS n_evidence,
            MIN(e.publication_year) AS min_year,
            MAX(e.publication_year) AS max_year,
            GROUP_CONCAT(DISTINCT COALESCE(e.source_db, e.reference_name, e.source_domain)) AS sources,
            MAX(CASE
                WHEN json_extract(e.payload_json, '$.has_conflict') IN (1, '1', true)
                THEN 1 ELSE 0 END) AS has_conflict
        FROM graph_claims c
        LEFT JOIN graph_evidence e ON c.claim_id = e.claim_id
        GROUP BY
            c.claim_id, c.domain_code, c.claim_family, c.claim_label,
            c.anchor_key, c.base_anchor_key, c.context_hash, c.context_json
        """,
        conn,
    )
    if claims.empty:
        return claims
    claims["context_json"] = claims["context_json"].apply(lambda value: json.loads(value) if value else {})
    return claims


def _load_rollup_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    rollups = pd.read_sql_query(
        """
        SELECT rollup_id, domain_code, source_table, source_row_id, claim_family,
               anchor_key, rollup_type, summary_json
        FROM graph_claim_rollups
        """,
        conn,
    )
    if rollups.empty:
        return rollups
    rollups["summary_json"] = rollups["summary_json"].apply(lambda value: json.loads(value) if value else {})
    return rollups


def _wipe_contradictions(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM graph_contradiction_members")
    conn.execute("DELETE FROM graph_contradiction_groups")
    conn.commit()


def _insert_group(
    conn: sqlite3.Connection,
    *,
    build_id: int | None,
    group_key: str,
    contradiction_type: str,
    claim_family: str,
    anchor_key: str | None,
    severity_tier: str,
    discovery_score: float,
    explanation: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    conn.execute(
        """INSERT OR IGNORE INTO graph_contradiction_groups
           (build_id, group_key, contradiction_type, claim_family, anchor_key,
            severity_tier, discovery_score, explanation, metadata_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            build_id,
            group_key,
            contradiction_type,
            claim_family,
            anchor_key,
            severity_tier,
            float(discovery_score),
            explanation,
            stable_json_dumps(metadata) if metadata is not None else None,
        ),
    )
    row = conn.execute(
        "SELECT group_id FROM graph_contradiction_groups WHERE group_key = ?",
        (group_key,),
    ).fetchone()
    return int(row[0])


def _insert_members(conn: sqlite3.Connection, group_id: int, claim_ids: list[int], member_role: str) -> None:
    for claim_id in sorted(set(claim_ids)):
        conn.execute(
            """INSERT OR IGNORE INTO graph_contradiction_members
               (group_id, claim_id, member_role)
               VALUES (?, ?, ?)""",
            (group_id, int(claim_id), member_role),
        )


def build_contradictions(graph_db_path: str | Path = DEFAULT_GRAPH_DB_PATH) -> dict[str, Any]:
    """Populate contradiction tables from graph claims and rollups."""
    conn = get_connection(graph_db_path)
    try:
        _wipe_contradictions(conn)
        claims = _load_claim_frame(conn)
        rollups = _load_rollup_frame(conn)
        build_id_row = conn.execute("SELECT MAX(build_id) FROM graph_builds").fetchone()
        build_id = int(build_id_row[0]) if build_id_row and build_id_row[0] is not None else None

        group_count = 0
        member_count = 0

        if not claims.empty:
            for family, family_df in claims.groupby("claim_family"):
                incompatible = INCOMPATIBLE_LABELS.get(family, set())
                if not incompatible:
                    continue

                by_anchor = family_df.groupby("anchor_key")
                for anchor, anchor_df in by_anchor:
                    labels = set(anchor_df["claim_label"])
                    for pair in incompatible:
                        if not pair.issubset(labels):
                            continue
                        subset = anchor_df[anchor_df["claim_label"].isin(pair)]
                        group_key = f"direct|{family}|{anchor}|{'-'.join(sorted(pair))}"
                        group_id = _insert_group(
                            conn,
                            build_id=build_id,
                            group_key=group_key,
                            contradiction_type="direct_label_conflict",
                            claim_family=family,
                            anchor_key=anchor,
                            severity_tier=SEVERITY_BY_TYPE["direct_label_conflict"],
                            discovery_score=10.0 + float(subset["n_evidence"].fillna(0).sum()),
                            explanation=f"Conflicting {family} labels {sorted(pair)} on the same anchor.",
                            metadata={"labels": sorted(pair)},
                        )
                        _insert_members(conn, group_id, subset["claim_id"].tolist(), "conflicting_claim")

                by_base = family_df.groupby("base_anchor_key")
                for base_anchor, base_df in by_base:
                    labels = set(base_df["claim_label"])
                    unique_anchors = set(base_df["anchor_key"])
                    unique_sources = {
                        source
                        for source_blob in base_df["sources"].dropna().tolist()
                        for source in str(source_blob).split(",")
                        if source
                    }
                    batch_names = {
                        (ctx or {}).get("batch_name")
                        for ctx in base_df["context_json"].tolist()
                        if isinstance(ctx, dict) and (ctx or {}).get("batch_name")
                    }
                    for pair in incompatible:
                        if not pair.issubset(labels):
                            continue
                        subset = base_df[base_df["claim_label"].isin(pair)]
                        if len(unique_anchors) > 1:
                            group_key = f"context|{family}|{base_anchor}|{'-'.join(sorted(pair))}"
                            group_id = _insert_group(
                                conn,
                                build_id=build_id,
                                group_key=group_key,
                                contradiction_type="context_specificity",
                                claim_family=family,
                                anchor_key=base_anchor,
                                severity_tier=SEVERITY_BY_TYPE["context_specificity"],
                                discovery_score=7.5 + float(subset["n_evidence"].fillna(0).sum()),
                                explanation=f"{family} labels {sorted(pair)} diverge across contexts for the same base anchor.",
                                metadata={"labels": sorted(pair), "n_contexts": len(unique_anchors)},
                            )
                            _insert_members(conn, group_id, subset["claim_id"].tolist(), "context_claim")
                        min_year = subset["min_year"].dropna()
                        max_year = subset["max_year"].dropna()
                        if not min_year.empty and not max_year.empty and len(set(subset["claim_label"])) > 1:
                            group_key = f"temporal|{family}|{base_anchor}|{'-'.join(sorted(pair))}"
                            group_id = _insert_group(
                                conn,
                                build_id=build_id,
                                group_key=group_key,
                                contradiction_type="temporal_revision",
                                claim_family=family,
                                anchor_key=base_anchor,
                                severity_tier=SEVERITY_BY_TYPE["temporal_revision"],
                                discovery_score=9.0 + float(max_year.max() - min_year.min()),
                                explanation=f"{family} labels {sorted(pair)} shift across time for the same base anchor.",
                                metadata={"labels": sorted(pair), "min_year": int(min_year.min()), "max_year": int(max_year.max())},
                            )
                            _insert_members(conn, group_id, subset["claim_id"].tolist(), "temporal_claim")
                        if len(unique_sources) > 1 or len(batch_names) > 1:
                            group_key = f"source|{family}|{base_anchor}|{'-'.join(sorted(pair))}"
                            group_id = _insert_group(
                                conn,
                                build_id=build_id,
                                group_key=group_key,
                                contradiction_type="source_or_batch_disagreement",
                                claim_family=family,
                                anchor_key=base_anchor,
                                severity_tier=SEVERITY_BY_TYPE["source_or_batch_disagreement"],
                                discovery_score=6.0 + len(unique_sources) + len(batch_names),
                                explanation=f"{family} disagreement spans multiple sources or batches.",
                                metadata={"labels": sorted(pair), "sources": sorted(unique_sources), "batches": sorted(batch_names)},
                            )
                            _insert_members(conn, group_id, subset["claim_id"].tolist(), "source_claim")

            vp_conflicts = claims[
                (claims["claim_family"] == "classification")
                & (claims["has_conflict"].fillna(0).astype(int) > 0)
            ]
            for _, row in vp_conflicts.iterrows():
                group_key = f"vp_conflict|{row['claim_id']}"
                group_id = _insert_group(
                    conn,
                    build_id=build_id,
                    group_key=group_key,
                    contradiction_type="submission_conflict",
                    claim_family="classification",
                    anchor_key=row["anchor_key"],
                    severity_tier=SEVERITY_BY_TYPE["submission_conflict"],
                    discovery_score=9.5 + float(row.get("n_evidence", 0) or 0),
                    explanation="VP benign claim carries an explicit submission-level conflict flag.",
                    metadata={"claim_id": int(row["claim_id"])},
                )
                _insert_members(conn, group_id, [int(row["claim_id"])], "conflicting_claim")

        if not rollups.empty:
            for _, row in rollups.iterrows():
                summary = row["summary_json"] or {}
                mixed = (
                    summary.get("consensus") == "mixed"
                    or summary.get("consensus_class") == "context_dependent"
                    or summary.get("has_conflict") in (1, True, "1")
                )
                if not mixed:
                    continue
                claim_ids = claims[
                    (claims["claim_family"] == row["claim_family"])
                    & (claims["base_anchor_key"] == row["anchor_key"])
                ]["claim_id"].tolist()
                group_key = f"mixed|{row['claim_family']}|{row['anchor_key']}|{row['source_table']}|{row['source_row_id']}"
                group_id = _insert_group(
                    conn,
                    build_id=build_id,
                    group_key=group_key,
                    contradiction_type="mixed_consensus",
                    claim_family=row["claim_family"],
                    anchor_key=row["anchor_key"],
                    severity_tier=SEVERITY_BY_TYPE["mixed_consensus"],
                    discovery_score=6.5 + len(claim_ids),
                    explanation=f"Aggregate rollup indicates mixed consensus for {row['claim_family']}.",
                    metadata=summary,
                )
                _insert_members(conn, group_id, claim_ids, "rollup_claim")

        group_count = int(conn.execute("SELECT COUNT(*) FROM graph_contradiction_groups").fetchone()[0])
        member_count = int(conn.execute("SELECT COUNT(*) FROM graph_contradiction_members").fetchone()[0])
        conn.commit()
        return {
            "graph_db_path": str(graph_db_path),
            "group_count": group_count,
            "member_count": member_count,
        }
    finally:
        conn.close()
