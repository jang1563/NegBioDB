PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS graph_builds (
    build_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    build_tag        TEXT,
    started_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    completed_at     TEXT,
    status           TEXT NOT NULL DEFAULT 'running'
                     CHECK (status IN ('running', 'complete', 'failed')),
    strict_mode      INTEGER NOT NULL DEFAULT 0,
    manifest_path    TEXT,
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS graph_build_inputs (
    input_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    build_id         INTEGER NOT NULL REFERENCES graph_builds(build_id) ON DELETE CASCADE,
    input_name       TEXT NOT NULL,
    input_kind       TEXT NOT NULL,
    domain_code      TEXT,
    path             TEXT,
    is_required      INTEGER NOT NULL DEFAULT 1,
    is_available     INTEGER NOT NULL DEFAULT 0,
    checksum_sha256  TEXT,
    status           TEXT NOT NULL DEFAULT 'discovered',
    metadata_json    TEXT,
    created_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS graph_entities (
    entity_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type      TEXT NOT NULL,
    canonical_key    TEXT NOT NULL,
    display_name     TEXT,
    primary_domain   TEXT,
    attrs_json       TEXT,
    created_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(entity_type, canonical_key)
);

CREATE TABLE IF NOT EXISTS graph_entity_aliases (
    alias_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id        INTEGER NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
    alias_type       TEXT NOT NULL,
    alias_value      TEXT NOT NULL,
    source_domain    TEXT NOT NULL DEFAULT '',
    source_table     TEXT NOT NULL DEFAULT '',
    source_record_id TEXT NOT NULL DEFAULT '',
    confidence_score REAL NOT NULL DEFAULT 1.0,
    created_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(entity_id, alias_type, alias_value, source_domain, source_table, source_record_id)
);

CREATE TABLE IF NOT EXISTS graph_bridges (
    bridge_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    left_entity_id    INTEGER NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
    right_entity_id   INTEGER NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
    bridge_type       TEXT NOT NULL,
    source_domain     TEXT,
    method            TEXT NOT NULL,
    bridge_key        TEXT NOT NULL UNIQUE,
    confidence_score  REAL NOT NULL DEFAULT 1.0,
    metadata_json     TEXT,
    created_at        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS graph_claims (
    claim_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_code       TEXT NOT NULL,
    claim_family      TEXT NOT NULL,
    claim_label       TEXT NOT NULL,
    claim_level       TEXT NOT NULL DEFAULT 'raw'
                      CHECK (claim_level IN ('raw', 'reference')),
    anchor_key        TEXT NOT NULL,
    base_anchor_key   TEXT NOT NULL,
    context_hash      TEXT NOT NULL,
    context_json      TEXT,
    claim_text        TEXT,
    created_at        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(claim_family, claim_label, anchor_key, context_hash, claim_level)
);

CREATE TABLE IF NOT EXISTS graph_claim_entities (
    claim_id          INTEGER NOT NULL REFERENCES graph_claims(claim_id) ON DELETE CASCADE,
    entity_id         INTEGER NOT NULL REFERENCES graph_entities(entity_id) ON DELETE CASCADE,
    role              TEXT NOT NULL,
    ordinal           INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (claim_id, entity_id, role, ordinal)
);

CREATE TABLE IF NOT EXISTS graph_evidence (
    evidence_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id          INTEGER NOT NULL REFERENCES graph_claims(claim_id) ON DELETE CASCADE,
    build_id          INTEGER REFERENCES graph_builds(build_id) ON DELETE SET NULL,
    source_domain     TEXT NOT NULL,
    source_db         TEXT,
    source_table      TEXT NOT NULL,
    source_record_id  TEXT NOT NULL,
    extraction_method TEXT,
    confidence_tier   TEXT,
    evidence_type     TEXT,
    publication_year  INTEGER,
    event_date        TEXT,
    reference_name    TEXT,
    payload_json      TEXT,
    provenance_json   TEXT,
    created_at        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS graph_claim_rollups (
    rollup_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    build_id          INTEGER REFERENCES graph_builds(build_id) ON DELETE SET NULL,
    domain_code       TEXT NOT NULL,
    source_table      TEXT NOT NULL,
    source_row_id     TEXT NOT NULL,
    claim_family      TEXT NOT NULL,
    anchor_key        TEXT NOT NULL,
    rollup_type       TEXT NOT NULL,
    summary_json      TEXT NOT NULL,
    created_at        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(domain_code, source_table, source_row_id, rollup_type)
);

CREATE TABLE IF NOT EXISTS graph_contradiction_groups (
    group_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    build_id          INTEGER REFERENCES graph_builds(build_id) ON DELETE SET NULL,
    group_key         TEXT NOT NULL UNIQUE,
    contradiction_type TEXT NOT NULL,
    claim_family      TEXT NOT NULL,
    anchor_key        TEXT,
    bridge_key        TEXT,
    severity_tier     TEXT NOT NULL CHECK (severity_tier IN ('high', 'medium', 'low')),
    discovery_score   REAL NOT NULL DEFAULT 0.0,
    explanation       TEXT,
    metadata_json     TEXT,
    created_at        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS graph_contradiction_members (
    group_id          INTEGER NOT NULL REFERENCES graph_contradiction_groups(group_id) ON DELETE CASCADE,
    claim_id          INTEGER NOT NULL REFERENCES graph_claims(claim_id) ON DELETE CASCADE,
    member_role       TEXT NOT NULL,
    notes             TEXT,
    PRIMARY KEY (group_id, claim_id, member_role)
);

CREATE INDEX IF NOT EXISTS idx_graph_inputs_build ON graph_build_inputs(build_id);
CREATE INDEX IF NOT EXISTS idx_graph_entities_type ON graph_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_graph_entities_display ON graph_entities(display_name);
CREATE INDEX IF NOT EXISTS idx_graph_alias_lookup ON graph_entity_aliases(alias_type, alias_value);
CREATE INDEX IF NOT EXISTS idx_graph_alias_entity ON graph_entity_aliases(entity_id);
CREATE INDEX IF NOT EXISTS idx_graph_bridges_left ON graph_bridges(left_entity_id);
CREATE INDEX IF NOT EXISTS idx_graph_bridges_right ON graph_bridges(right_entity_id);
CREATE INDEX IF NOT EXISTS idx_graph_bridges_type ON graph_bridges(bridge_type);
CREATE INDEX IF NOT EXISTS idx_graph_claims_domain ON graph_claims(domain_code);
CREATE INDEX IF NOT EXISTS idx_graph_claims_family ON graph_claims(claim_family);
CREATE INDEX IF NOT EXISTS idx_graph_claims_anchor ON graph_claims(anchor_key);
CREATE INDEX IF NOT EXISTS idx_graph_claims_base_anchor ON graph_claims(base_anchor_key);
CREATE INDEX IF NOT EXISTS idx_graph_claim_entities_entity ON graph_claim_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_graph_claim_entities_role ON graph_claim_entities(role);
CREATE INDEX IF NOT EXISTS idx_graph_evidence_claim ON graph_evidence(claim_id);
CREATE INDEX IF NOT EXISTS idx_graph_evidence_source ON graph_evidence(source_domain, source_table);
CREATE INDEX IF NOT EXISTS idx_graph_rollups_anchor ON graph_claim_rollups(anchor_key);
CREATE INDEX IF NOT EXISTS idx_graph_contradictions_family ON graph_contradiction_groups(claim_family, contradiction_type);
CREATE INDEX IF NOT EXISTS idx_graph_contradiction_members_claim ON graph_contradiction_members(claim_id);

INSERT INTO schema_migrations (version) VALUES ('001');
