-- NegBioDB Gene Essentiality (GE) Domain — Initial Schema
-- Migration 001: Core tables for DepMap CRISPR/RNAi gene essentiality negatives
--
-- Design decisions:
--   - Asymmetric pairs: gene + cell_line (not symmetric like PPI)
--   - Separate genes/cell_lines tables (separate DB from DTI/CT/PPI)
--   - Confidence tiers: gold/silver/bronze (same framework as DTI/CT/PPI)
--   - PRISM bridge tables: cross-domain link to DTI via InChIKey/ChEMBL
--   - Dedup: UNIQUE on (gene_id, cell_line_id, screen_id, source_db)
--   - Reference flags on genes table for common essential / nonessential sets

-- ============================================================
-- Common Layer tables (same as DTI/CT/PPI)
-- ============================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    version     TEXT NOT NULL,
    source_url  TEXT,
    download_date TEXT,
    file_hash   TEXT,
    row_count   INTEGER,
    notes       TEXT,
    created_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================
-- Domain-specific tables: Gene Essentiality
-- ============================================================

-- Genes table
CREATE TABLE genes (
    gene_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    entrez_id               INTEGER UNIQUE,
    gene_symbol             TEXT NOT NULL,
    ensembl_id              TEXT,
    description             TEXT,
    is_common_essential     INTEGER DEFAULT 0,
    is_reference_nonessential INTEGER DEFAULT 0,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_genes_symbol ON genes(gene_symbol);
CREATE INDEX idx_genes_entrez ON genes(entrez_id) WHERE entrez_id IS NOT NULL;

-- Cell lines table
CREATE TABLE cell_lines (
    cell_line_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id                TEXT NOT NULL UNIQUE,
    ccle_name               TEXT,
    stripped_name            TEXT,
    lineage                 TEXT,
    primary_disease         TEXT,
    subtype                 TEXT,
    sex                     TEXT,
    primary_or_metastasis   TEXT,
    sample_collection_site  TEXT,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_cell_lines_ccle ON cell_lines(ccle_name) WHERE ccle_name IS NOT NULL;
CREATE INDEX idx_cell_lines_stripped ON cell_lines(stripped_name) WHERE stripped_name IS NOT NULL;
CREATE INDEX idx_cell_lines_lineage ON cell_lines(lineage) WHERE lineage IS NOT NULL;

-- Screen configurations
CREATE TABLE ge_screens (
    screen_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_db               TEXT NOT NULL CHECK (source_db IN (
        'depmap', 'project_score', 'demeter2')),
    depmap_release          TEXT NOT NULL,
    screen_type             TEXT NOT NULL CHECK (screen_type IN ('crispr', 'rnai')),
    library                 TEXT,
    algorithm               TEXT,
    notes                   TEXT,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE UNIQUE INDEX idx_ge_screens_source
    ON ge_screens(source_db, depmap_release, screen_type);

-- Core fact table: GE negative results (non-essential gene-cell_line pairs)
CREATE TABLE ge_negative_results (
    result_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    gene_id                 INTEGER NOT NULL REFERENCES genes(gene_id),
    cell_line_id            INTEGER NOT NULL REFERENCES cell_lines(cell_line_id),
    screen_id               INTEGER REFERENCES ge_screens(screen_id),

    gene_effect_score       REAL,
    dependency_probability  REAL,

    evidence_type           TEXT NOT NULL CHECK (evidence_type IN (
        'crispr_nonessential',
        'rnai_nonessential',
        'multi_screen_concordant',
        'reference_nonessential',
        'context_nonessential')),

    confidence_tier         TEXT NOT NULL CHECK (confidence_tier IN (
        'gold', 'silver', 'bronze')),

    source_db               TEXT NOT NULL,
    source_record_id        TEXT NOT NULL,
    extraction_method       TEXT NOT NULL CHECK (extraction_method IN (
        'score_threshold', 'reference_set', 'multi_source_concordance')),

    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_ge_nr_gene ON ge_negative_results(gene_id);
CREATE INDEX idx_ge_nr_cell_line ON ge_negative_results(cell_line_id);
CREATE INDEX idx_ge_nr_pair ON ge_negative_results(gene_id, cell_line_id);
CREATE INDEX idx_ge_nr_tier ON ge_negative_results(confidence_tier);
CREATE INDEX idx_ge_nr_source ON ge_negative_results(source_db);

CREATE UNIQUE INDEX idx_ge_nr_unique_source ON ge_negative_results(
    gene_id, cell_line_id,
    COALESCE(screen_id, -1),
    source_db);

-- ============================================================
-- Aggregation table
-- ============================================================

CREATE TABLE gene_cell_pairs (
    pair_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    gene_id                 INTEGER NOT NULL REFERENCES genes(gene_id),
    cell_line_id            INTEGER NOT NULL REFERENCES cell_lines(cell_line_id),
    num_screens             INTEGER NOT NULL,
    num_sources             INTEGER NOT NULL,
    best_confidence         TEXT NOT NULL,
    best_evidence_type      TEXT,
    min_gene_effect         REAL,
    max_gene_effect         REAL,
    mean_gene_effect        REAL,
    gene_degree             INTEGER,
    cell_line_degree        INTEGER,
    UNIQUE(gene_id, cell_line_id)
);

CREATE INDEX idx_gcp_gene ON gene_cell_pairs(gene_id);
CREATE INDEX idx_gcp_cell_line ON gene_cell_pairs(cell_line_id);
CREATE INDEX idx_gcp_confidence ON gene_cell_pairs(best_confidence);

-- ============================================================
-- Benchmark split tables
-- ============================================================

CREATE TABLE ge_split_definitions (
    split_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    split_name              TEXT NOT NULL,
    split_strategy          TEXT NOT NULL CHECK (split_strategy IN (
        'random', 'cold_gene', 'cold_cell_line',
        'cold_both', 'degree_balanced')),
    description             TEXT,
    random_seed             INTEGER,
    train_ratio             REAL DEFAULT 0.7,
    val_ratio               REAL DEFAULT 0.1,
    test_ratio              REAL DEFAULT 0.2,
    date_created            TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    version                 TEXT DEFAULT '1.0',
    UNIQUE(split_name, version)
);

CREATE TABLE ge_split_assignments (
    pair_id                 INTEGER NOT NULL REFERENCES gene_cell_pairs(pair_id),
    split_id                INTEGER NOT NULL REFERENCES ge_split_definitions(split_id),
    fold                    TEXT NOT NULL CHECK (fold IN ('train', 'val', 'test')),
    PRIMARY KEY (pair_id, split_id)
);

CREATE INDEX idx_ge_splits_fold ON ge_split_assignments(split_id, fold);

-- ============================================================
-- PRISM drug sensitivity bridge tables
-- ============================================================

CREATE TABLE prism_compounds (
    compound_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    broad_id                TEXT UNIQUE,
    name                    TEXT,
    smiles                  TEXT,
    inchikey                TEXT,
    chembl_id               TEXT,
    pubchem_cid             INTEGER,
    mechanism_of_action     TEXT,
    target_name             TEXT,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_prism_inchikey ON prism_compounds(inchikey)
    WHERE inchikey IS NOT NULL;
CREATE INDEX idx_prism_chembl ON prism_compounds(chembl_id)
    WHERE chembl_id IS NOT NULL;

CREATE TABLE prism_sensitivity (
    sensitivity_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_id             INTEGER NOT NULL REFERENCES prism_compounds(compound_id),
    cell_line_id            INTEGER NOT NULL REFERENCES cell_lines(cell_line_id),
    screen_type             TEXT CHECK (screen_type IN ('primary', 'secondary')),
    log_fold_change         REAL,
    auc                     REAL,
    ic50                    REAL,
    ec50                    REAL,
    depmap_release          TEXT,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_prism_sens_compound ON prism_sensitivity(compound_id);
CREATE INDEX idx_prism_sens_cell_line ON prism_sensitivity(cell_line_id);

-- Record migration
INSERT INTO schema_migrations (version) VALUES ('001');
