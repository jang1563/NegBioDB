-- ============================================================
-- NegBioDB Schema v1.1
-- Migration: 002_target_variants
-- Purpose:
--   1) Separate target variants from canonical UniProt targets
--   2) Link negative results to optional variant context
-- ============================================================

CREATE TABLE target_variants (
    variant_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id            INTEGER NOT NULL REFERENCES targets(target_id),
    variant_label        TEXT NOT NULL,             -- e.g., E255K, T315I
    raw_gene_name        TEXT,                      -- e.g., ABL1(E255K)-phosphorylated
    source_db            TEXT NOT NULL CHECK (source_db IN (
                            'davis', 'pubchem', 'chembl', 'bindingdb',
                            'literature', 'community')),
    source_record_id     TEXT NOT NULL DEFAULT '',
    created_at           TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(target_id, variant_label, source_db, source_record_id)
);

CREATE INDEX idx_target_variants_target ON target_variants(target_id);
CREATE INDEX idx_target_variants_label ON target_variants(variant_label);

ALTER TABLE negative_results ADD COLUMN variant_id INTEGER REFERENCES target_variants(variant_id);
CREATE INDEX idx_results_variant ON negative_results(variant_id);

-- Record this migration
INSERT INTO schema_migrations (version, description, sql_up)
    VALUES ('002', 'Add target_variants table and negative_results.variant_id', 'Full DDL');
