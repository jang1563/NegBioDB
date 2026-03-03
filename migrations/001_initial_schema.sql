-- ============================================================
-- NegBioDB Schema v1.0
-- Migration: 001_initial_schema
-- Database: SQLite 3.35+
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA encoding = 'UTF-8';

-- ============================================================
-- COMMON LAYER
-- ============================================================

CREATE TABLE compounds (
    compound_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_smiles    TEXT NOT NULL,
    inchikey            TEXT NOT NULL,
    inchikey_connectivity TEXT NOT NULL,
    inchi               TEXT,
    pubchem_cid         INTEGER,
    chembl_id           TEXT,
    bindingdb_id        INTEGER,
    molecular_weight    REAL,
    logp                REAL,
    hbd                 INTEGER,
    hba                 INTEGER,
    tpsa                REAL,
    rotatable_bonds     INTEGER,
    num_heavy_atoms     INTEGER,
    qed                 REAL,
    pains_alert         INTEGER DEFAULT 0,
    aggregator_alert    INTEGER DEFAULT 0,
    lipinski_violations INTEGER DEFAULT 0,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE UNIQUE INDEX idx_compounds_inchikey ON compounds(inchikey);
CREATE INDEX idx_compounds_connectivity ON compounds(inchikey_connectivity);
CREATE INDEX idx_compounds_pubchem ON compounds(pubchem_cid) WHERE pubchem_cid IS NOT NULL;
CREATE INDEX idx_compounds_chembl ON compounds(chembl_id) WHERE chembl_id IS NOT NULL;
CREATE INDEX idx_compounds_smiles ON compounds(canonical_smiles);

CREATE TABLE targets (
    target_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    uniprot_accession   TEXT NOT NULL,
    uniprot_entry_name  TEXT,
    amino_acid_sequence TEXT,
    sequence_length     INTEGER,
    chembl_target_id    TEXT,
    gene_symbol         TEXT,
    ncbi_gene_id        INTEGER,
    target_family       TEXT,
    target_subfamily    TEXT,
    dto_class           TEXT,
    development_level   TEXT CHECK (development_level IN ('Tclin', 'Tchem', 'Tbio', 'Tdark')),
    organism            TEXT DEFAULT 'Homo sapiens',
    taxonomy_id         INTEGER DEFAULT 9606,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE UNIQUE INDEX idx_targets_uniprot ON targets(uniprot_accession);
CREATE INDEX idx_targets_chembl ON targets(chembl_target_id) WHERE chembl_target_id IS NOT NULL;
CREATE INDEX idx_targets_gene ON targets(gene_symbol) WHERE gene_symbol IS NOT NULL;
CREATE INDEX idx_targets_family ON targets(target_family);
CREATE INDEX idx_targets_dev_level ON targets(development_level);

CREATE TABLE assays (
    assay_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_db           TEXT NOT NULL CHECK (source_db IN (
                            'pubchem', 'chembl', 'bindingdb', 'literature', 'community')),
    source_assay_id     TEXT NOT NULL,
    assay_type          TEXT,
    assay_format        TEXT CHECK (assay_format IN (
                            'biochemical', 'cell-based', 'in_vivo', 'unknown')),
    assay_technology    TEXT,
    detection_method    TEXT,
    screen_type         TEXT CHECK (screen_type IN (
                            'primary_single_point', 'confirmatory_dose_response',
                            'counter_screen', 'orthogonal_assay',
                            'literature_assay', 'unknown')),
    z_factor            REAL,
    ssmd                REAL,
    cell_line           TEXT,
    description         TEXT,
    pubmed_id           INTEGER,
    doi                 TEXT,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE UNIQUE INDEX idx_assays_source ON assays(source_db, source_assay_id);
CREATE INDEX idx_assays_format ON assays(assay_format);
CREATE INDEX idx_assays_screen ON assays(screen_type);

-- ============================================================
-- DTI DOMAIN LAYER
-- ============================================================

CREATE TABLE negative_results (
    result_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_id         INTEGER NOT NULL REFERENCES compounds(compound_id),
    target_id           INTEGER NOT NULL REFERENCES targets(target_id),
    assay_id            INTEGER REFERENCES assays(assay_id),
    result_type         TEXT NOT NULL CHECK (result_type IN (
                            'hard_negative', 'conditional_negative',
                            'methodological_negative', 'dose_time_negative',
                            'hypothesis_negative')),
    confidence_tier     TEXT NOT NULL CHECK (confidence_tier IN (
                            'gold', 'silver', 'bronze', 'copper')),
    activity_type       TEXT,
    activity_value      REAL,
    activity_unit       TEXT,
    activity_relation   TEXT DEFAULT '=',
    pchembl_value       REAL,
    inactivity_threshold        REAL,
    inactivity_threshold_unit   TEXT DEFAULT 'nM',
    max_concentration_tested    REAL,
    num_replicates      INTEGER,
    species_tested      TEXT DEFAULT 'Homo sapiens',
    source_db           TEXT NOT NULL,
    source_record_id    TEXT NOT NULL,
    extraction_method   TEXT NOT NULL CHECK (extraction_method IN (
                            'database_direct', 'text_mining',
                            'llm_extracted', 'community_submitted')),
    curator_validated   INTEGER DEFAULT 0,
    publication_year    INTEGER,
    deposition_date     TEXT,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_results_compound ON negative_results(compound_id);
CREATE INDEX idx_results_target ON negative_results(target_id);
CREATE INDEX idx_results_pair ON negative_results(compound_id, target_id);
CREATE INDEX idx_results_tier ON negative_results(confidence_tier);
CREATE INDEX idx_results_source ON negative_results(source_db);
CREATE INDEX idx_results_year ON negative_results(publication_year);
CREATE INDEX idx_results_type ON negative_results(result_type);

-- COALESCE handles NULL assay_id: SQLite treats NULL as distinct in UNIQUE indexes,
-- which would allow duplicate rows when assay_id is missing.
CREATE UNIQUE INDEX idx_results_unique_source ON negative_results(
    compound_id, target_id, COALESCE(assay_id, -1), source_db, source_record_id);

CREATE TABLE dti_context (
    result_id           INTEGER PRIMARY KEY REFERENCES negative_results(result_id),
    binding_site        TEXT CHECK (binding_site IN (
                            'orthosteric', 'allosteric', 'unknown')),
    selectivity_panel   INTEGER DEFAULT 0,
    counterpart_active  INTEGER DEFAULT 0,
    cell_permeability_issue INTEGER DEFAULT 0,
    compound_solubility REAL,
    compound_stability  TEXT
);

-- ============================================================
-- AGGREGATION LAYER (for ML export)
-- ============================================================

CREATE TABLE compound_target_pairs (
    pair_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_id         INTEGER NOT NULL REFERENCES compounds(compound_id),
    target_id           INTEGER NOT NULL REFERENCES targets(target_id),
    num_assays          INTEGER NOT NULL,
    num_sources         INTEGER NOT NULL,
    best_confidence     TEXT NOT NULL,
    best_result_type    TEXT,
    earliest_year       INTEGER,
    median_pchembl      REAL,
    min_activity_value  REAL,
    max_activity_value  REAL,
    has_conflicting_results INTEGER DEFAULT 0,
    compound_degree     INTEGER,
    target_degree       INTEGER,
    UNIQUE(compound_id, target_id)
);

CREATE INDEX idx_pairs_compound ON compound_target_pairs(compound_id);
CREATE INDEX idx_pairs_target ON compound_target_pairs(target_id);
CREATE INDEX idx_pairs_confidence ON compound_target_pairs(best_confidence);

-- ============================================================
-- BENCHMARK / ML LAYER
-- ============================================================

CREATE TABLE split_definitions (
    split_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    split_name          TEXT NOT NULL,
    split_strategy      TEXT NOT NULL CHECK (split_strategy IN (
                            'random', 'cold_compound', 'cold_target', 'cold_both',
                            'temporal', 'scaffold', 'degree_balanced')),
    description         TEXT,
    random_seed         INTEGER,
    train_ratio         REAL DEFAULT 0.7,
    val_ratio           REAL DEFAULT 0.1,
    test_ratio          REAL DEFAULT 0.2,
    date_created        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    version             TEXT DEFAULT '1.0',
    UNIQUE(split_name, version)
);

CREATE TABLE split_assignments (
    pair_id             INTEGER NOT NULL REFERENCES compound_target_pairs(pair_id),
    split_id            INTEGER NOT NULL REFERENCES split_definitions(split_id),
    fold                TEXT NOT NULL CHECK (fold IN ('train', 'val', 'test')),
    PRIMARY KEY (pair_id, split_id)
);

CREATE INDEX idx_splits_fold ON split_assignments(split_id, fold);

-- ============================================================
-- METADATA LAYER
-- ============================================================

CREATE TABLE dataset_versions (
    version_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    version_tag         TEXT NOT NULL UNIQUE,
    description         TEXT,
    num_compounds       INTEGER,
    num_targets         INTEGER,
    num_pairs           INTEGER,
    num_results         INTEGER,
    schema_version      TEXT,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    checksum_sha256     TEXT
);

CREATE TABLE schema_migrations (
    migration_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    version             TEXT NOT NULL,
    description         TEXT,
    applied_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    sql_up              TEXT,
    sql_down            TEXT
);

-- Record this migration
INSERT INTO schema_migrations (version, description, sql_up)
    VALUES ('001', 'Initial NegBioDB schema', 'Full DDL');
