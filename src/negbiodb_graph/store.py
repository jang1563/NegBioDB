"""Storage helpers for writing normalized graph records."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from negbiodb_graph.utils import as_jsonable, stable_json_dumps


@dataclass(frozen=True)
class ResolvedEntity:
    entity_id: int
    entity_type: str
    canonical_key: str


class GraphStore:
    """Thin helper around graph-table upserts with in-memory caches."""

    def __init__(self, conn: sqlite3.Connection, build_id: int | None = None):
        self.conn = conn
        self.build_id = build_id
        self._entity_cache: dict[tuple[str, str], ResolvedEntity] = {}
        self._alias_cache: dict[tuple[str, str, str], ResolvedEntity] = {}
        self._claim_cache: dict[tuple[str, str, str, str, str], int] = {}

    def _json(self, value: Any) -> str | None:
        if value is None:
            return None
        return stable_json_dumps(as_jsonable(value))

    def find_entity(self, entity_type: str, canonical_key: str) -> ResolvedEntity | None:
        cache_key = (entity_type, canonical_key)
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]
        row = self.conn.execute(
            """SELECT entity_id, entity_type, canonical_key
               FROM graph_entities
               WHERE entity_type = ? AND canonical_key = ?""",
            (entity_type, canonical_key),
        ).fetchone()
        if row is None:
            return None
        entity = ResolvedEntity(int(row["entity_id"]), row["entity_type"], row["canonical_key"])
        self._entity_cache[cache_key] = entity
        return entity

    def find_entity_by_alias(
        self,
        entity_type: str,
        alias_type: str,
        alias_value: str,
    ) -> ResolvedEntity | None:
        cache_key = (entity_type, alias_type, alias_value)
        if cache_key in self._alias_cache:
            return self._alias_cache[cache_key]
        row = self.conn.execute(
            """SELECT e.entity_id, e.entity_type, e.canonical_key
               FROM graph_entity_aliases a
               JOIN graph_entities e ON a.entity_id = e.entity_id
               WHERE e.entity_type = ? AND a.alias_type = ? AND a.alias_value = ?
               ORDER BY a.confidence_score DESC, a.alias_id ASC
               LIMIT 1""",
            (entity_type, alias_type, alias_value),
        ).fetchone()
        if row is None:
            return None
        entity = ResolvedEntity(int(row["entity_id"]), row["entity_type"], row["canonical_key"])
        self._alias_cache[cache_key] = entity
        self._entity_cache[(entity.entity_type, entity.canonical_key)] = entity
        return entity

    def upsert_entity(
        self,
        entity_type: str,
        canonical_key: str,
        *,
        display_name: str | None = None,
        primary_domain: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> ResolvedEntity:
        existing = self.find_entity(entity_type, canonical_key)
        attrs_json = self._json(attrs)
        if existing is not None:
            if display_name is not None or primary_domain is not None or attrs_json is not None:
                self.conn.execute(
                    """UPDATE graph_entities
                       SET display_name = COALESCE(?, display_name),
                           primary_domain = COALESCE(?, primary_domain),
                           attrs_json = COALESCE(?, attrs_json)
                       WHERE entity_id = ?""",
                    (display_name, primary_domain, attrs_json, existing.entity_id),
                )
            return existing

        self.conn.execute(
            """INSERT INTO graph_entities
               (entity_type, canonical_key, display_name, primary_domain, attrs_json)
               VALUES (?, ?, ?, ?, ?)""",
            (entity_type, canonical_key, display_name, primary_domain, attrs_json),
        )
        entity_id = int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        entity = ResolvedEntity(entity_id, entity_type, canonical_key)
        self._entity_cache[(entity_type, canonical_key)] = entity
        return entity

    def add_alias(
        self,
        entity: ResolvedEntity,
        alias_type: str,
        alias_value: str | None,
        *,
        source_domain: str = "",
        source_table: str = "",
        source_record_id: str = "",
        confidence_score: float = 1.0,
    ) -> None:
        if not alias_value:
            return
        self.conn.execute(
            """INSERT OR IGNORE INTO graph_entity_aliases
               (entity_id, alias_type, alias_value, source_domain, source_table,
                source_record_id, confidence_score)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entity.entity_id,
                alias_type,
                alias_value,
                source_domain,
                source_table,
                source_record_id,
                float(confidence_score),
            ),
        )
        self._alias_cache[(entity.entity_type, alias_type, alias_value)] = entity

    def add_bridge(
        self,
        left_entity: ResolvedEntity,
        right_entity: ResolvedEntity,
        *,
        bridge_type: str,
        source_domain: str | None,
        method: str,
        confidence_score: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        left_id, right_id = sorted((left_entity.entity_id, right_entity.entity_id))
        bridge_key = f"{bridge_type}|{left_id}|{right_id}|{source_domain or ''}|{method}"
        self.conn.execute(
            """INSERT OR IGNORE INTO graph_bridges
               (left_entity_id, right_entity_id, bridge_type, source_domain,
                method, bridge_key, confidence_score, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                left_id,
                right_id,
                bridge_type,
                source_domain,
                method,
                bridge_key,
                float(confidence_score),
                self._json(metadata),
            ),
        )
        return bridge_key

    def upsert_claim(
        self,
        *,
        domain_code: str,
        claim_family: str,
        claim_label: str,
        anchor_key: str,
        base_anchor_key: str,
        context_hash: str,
        context: dict[str, Any] | None = None,
        claim_text: str | None = None,
        claim_level: str = "raw",
    ) -> int:
        cache_key = (claim_family, claim_label, anchor_key, context_hash, claim_level)
        if cache_key in self._claim_cache:
            return self._claim_cache[cache_key]
        self.conn.execute(
            """INSERT OR IGNORE INTO graph_claims
               (domain_code, claim_family, claim_label, claim_level, anchor_key,
                base_anchor_key, context_hash, context_json, claim_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                domain_code,
                claim_family,
                claim_label,
                claim_level,
                anchor_key,
                base_anchor_key,
                context_hash,
                self._json(context),
                claim_text,
            ),
        )
        row = self.conn.execute(
            """SELECT claim_id FROM graph_claims
               WHERE claim_family = ? AND claim_label = ?
                 AND anchor_key = ? AND context_hash = ?
                 AND claim_level = ?""",
            (claim_family, claim_label, anchor_key, context_hash, claim_level),
        ).fetchone()
        claim_id = int(row["claim_id"])
        self._claim_cache[cache_key] = claim_id
        return claim_id

    def add_claim_entity(
        self,
        claim_id: int,
        entity: ResolvedEntity,
        *,
        role: str,
        ordinal: int = 0,
    ) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO graph_claim_entities
               (claim_id, entity_id, role, ordinal)
               VALUES (?, ?, ?, ?)""",
            (claim_id, entity.entity_id, role, int(ordinal)),
        )

    def add_evidence(
        self,
        claim_id: int,
        *,
        source_domain: str,
        source_table: str,
        source_record_id: str,
        source_db: str | None = None,
        extraction_method: str | None = None,
        confidence_tier: str | None = None,
        evidence_type: str | None = None,
        publication_year: int | None = None,
        event_date: str | None = None,
        reference_name: str | None = None,
        payload: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO graph_evidence
               (claim_id, build_id, source_domain, source_db, source_table, source_record_id,
                extraction_method, confidence_tier, evidence_type, publication_year,
                event_date, reference_name, payload_json, provenance_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim_id,
                self.build_id,
                source_domain,
                source_db,
                source_table,
                source_record_id,
                extraction_method,
                confidence_tier,
                evidence_type,
                publication_year,
                event_date,
                reference_name,
                self._json(payload),
                self._json(provenance),
            ),
        )

    def add_rollup(
        self,
        *,
        domain_code: str,
        source_table: str,
        source_row_id: str,
        claim_family: str,
        anchor_key: str,
        rollup_type: str,
        summary: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO graph_claim_rollups
               (build_id, domain_code, source_table, source_row_id,
                claim_family, anchor_key, rollup_type, summary_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.build_id,
                domain_code,
                source_table,
                source_row_id,
                claim_family,
                anchor_key,
                rollup_type,
                self._json(summary),
            ),
        )
