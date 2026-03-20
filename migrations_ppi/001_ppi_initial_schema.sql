-- NegBioDB PPI Domain — Initial Schema
-- Migration 001: Core tables for protein-protein interaction negatives
--
-- Design decisions:
--   - Symmetric pairs: CHECK (protein1_id < protein2_id) canonical ordering
--   - Separate proteins table (not reusing DTI targets — separate DB)
--   - Confidence tiers: gold/silver/bronze/copper (same as DTI/CT)
--   - Dedup: COALESCE(experiment_id, -1) pattern (same as DTI/CT)

-- ============================================================
-- Common Layer tables (same as DTI/CT)
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
-- Domain-specific tables: PPI Negatives
-- ============================================================

-- Proteins table
CREATE TABLE proteins (
    protein_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    uniprot_accession   TEXT NOT NULL,
    uniprot_entry_name  TEXT,
    gene_symbol         TEXT,
    amino_acid_sequence TEXT,
    sequence_length     INTEGER,
    organism            TEXT DEFAULT 'Homo sapiens',
    taxonomy_id         INTEGER DEFAULT 9606,
    subcellular_location TEXT,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE UNIQUE INDEX idx_proteins_uniprot ON proteins(uniprot_accession);
CREATE INDEX idx_proteins_gene ON proteins(gene_symbol)
    WHERE gene_symbol IS NOT NULL;

-- PPI experiments / evidence sources
CREATE TABLE ppi_experiments (
    experiment_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    source_db           TEXT NOT NULL CHECK (source_db IN (
        'huri', 'intact', 'humap', 'string', 'biogrid', 'pdb_derived', 'literature')),
    source_experiment_id TEXT NOT NULL,
    experiment_type     TEXT,
    detection_method    TEXT,
    detection_method_id TEXT,
    pubmed_id           INTEGER,
    doi                 TEXT,
    description         TEXT,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE UNIQUE INDEX idx_ppi_exp_source
    ON ppi_experiments(source_db, source_experiment_id);

-- Core fact table: PPI negative results
CREATE TABLE ppi_negative_results (
    result_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    protein1_id         INTEGER NOT NULL REFERENCES proteins(protein_id),
    protein2_id         INTEGER NOT NULL REFERENCES proteins(protein_id),
    experiment_id       INTEGER REFERENCES ppi_experiments(experiment_id),

    evidence_type       TEXT NOT NULL CHECK (evidence_type IN (
        'experimental_non_interaction',
        'ml_predicted_negative',
        'low_score_negative',
        'compartment_separated',
        'literature_reported')),

    confidence_tier     TEXT NOT NULL CHECK (confidence_tier IN (
        'gold', 'silver', 'bronze', 'copper')),

    interaction_score   REAL,
    score_type          TEXT,
    num_evidence_types  INTEGER,

    detection_method    TEXT,
    detection_method_id TEXT,
    organism_tested     TEXT DEFAULT 'Homo sapiens',

    source_db           TEXT NOT NULL,
    source_record_id    TEXT NOT NULL,
    extraction_method   TEXT NOT NULL CHECK (extraction_method IN (
        'database_direct', 'score_threshold',
        'ml_classifier', 'text_mining',
        'community_submitted')),
    curator_validated   INTEGER DEFAULT 0,
    publication_year    INTEGER,

    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),

    CHECK (protein1_id < protein2_id)
);

CREATE INDEX idx_ppi_nr_protein1 ON ppi_negative_results(protein1_id);
CREATE INDEX idx_ppi_nr_protein2 ON ppi_negative_results(protein2_id);
CREATE INDEX idx_ppi_nr_pair ON ppi_negative_results(protein1_id, protein2_id);
CREATE INDEX idx_ppi_nr_tier ON ppi_negative_results(confidence_tier);
CREATE INDEX idx_ppi_nr_source ON ppi_negative_results(source_db);

CREATE UNIQUE INDEX idx_ppi_nr_unique_source ON ppi_negative_results(
    protein1_id, protein2_id,
    COALESCE(experiment_id, -1),
    source_db, source_record_id);

-- ============================================================
-- Aggregation table
-- ============================================================

CREATE TABLE protein_protein_pairs (
    pair_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    protein1_id         INTEGER NOT NULL REFERENCES proteins(protein_id),
    protein2_id         INTEGER NOT NULL REFERENCES proteins(protein_id),
    num_experiments     INTEGER NOT NULL,
    num_sources         INTEGER NOT NULL,
    best_confidence     TEXT NOT NULL,
    best_evidence_type  TEXT,
    earliest_year       INTEGER,
    min_interaction_score REAL,
    max_interaction_score REAL,
    protein1_degree     INTEGER,
    protein2_degree     INTEGER,
    UNIQUE(protein1_id, protein2_id),
    CHECK (protein1_id < protein2_id)
);

CREATE INDEX idx_ppp_protein1 ON protein_protein_pairs(protein1_id);
CREATE INDEX idx_ppp_protein2 ON protein_protein_pairs(protein2_id);
CREATE INDEX idx_ppp_confidence ON protein_protein_pairs(best_confidence);

-- ============================================================
-- Benchmark split tables
-- ============================================================

CREATE TABLE ppi_split_definitions (
    split_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    split_name          TEXT NOT NULL,
    split_strategy      TEXT NOT NULL CHECK (split_strategy IN (
        'random', 'cold_protein', 'cold_both',
        'bfs_cluster', 'degree_balanced')),
    description         TEXT,
    random_seed         INTEGER,
    train_ratio         REAL DEFAULT 0.7,
    val_ratio           REAL DEFAULT 0.1,
    test_ratio          REAL DEFAULT 0.2,
    date_created        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    version             TEXT DEFAULT '1.0',
    UNIQUE(split_name, version)
);

CREATE TABLE ppi_split_assignments (
    pair_id             INTEGER NOT NULL REFERENCES protein_protein_pairs(pair_id),
    split_id            INTEGER NOT NULL REFERENCES ppi_split_definitions(split_id),
    fold                TEXT NOT NULL CHECK (fold IN ('train', 'val', 'test')),
    PRIMARY KEY (pair_id, split_id)
);

CREATE INDEX idx_ppi_splits_fold ON ppi_split_assignments(split_id, fold);

-- Record migration
INSERT INTO schema_migrations (version) VALUES ('001');
