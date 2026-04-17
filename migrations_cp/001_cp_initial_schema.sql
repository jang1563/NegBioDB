-- NegBioDB Cell Painting (CP) Domain — Initial Schema
-- Migration 001: Core tables for JUMP-first Cell Painting perturbation results

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     TEXT PRIMARY KEY,
    applied_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    dataset_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    annotation_mode TEXT NOT NULL DEFAULT 'annotated'
                    CHECK (annotation_mode IN ('annotated', 'plate_proxy')),
    source_url      TEXT,
    download_date   TEXT,
    file_hash       TEXT,
    row_count       INTEGER,
    notes           TEXT,
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE compounds (
    compound_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_name           TEXT,
    canonical_smiles        TEXT,
    inchikey                TEXT,
    inchikey_connectivity   TEXT,
    inchi                   TEXT,
    pubchem_cid             INTEGER,
    chembl_id               TEXT,
    molecular_weight        REAL,
    logp                    REAL,
    hbd                     INTEGER,
    hba                     INTEGER,
    tpsa                    REAL,
    rotatable_bonds         INTEGER,
    num_heavy_atoms         INTEGER,
    qed                     REAL,
    pains_alert             INTEGER DEFAULT 0,
    lipinski_violations     INTEGER DEFAULT 0,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE UNIQUE INDEX idx_cp_compounds_inchikey
    ON compounds(inchikey) WHERE inchikey IS NOT NULL;
CREATE INDEX idx_cp_compounds_connectivity
    ON compounds(inchikey_connectivity) WHERE inchikey_connectivity IS NOT NULL;
CREATE INDEX idx_cp_compounds_smiles
    ON compounds(canonical_smiles) WHERE canonical_smiles IS NOT NULL;
CREATE INDEX idx_cp_compounds_name
    ON compounds(compound_name) WHERE compound_name IS NOT NULL;

CREATE TABLE cp_cell_lines (
    cell_line_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_line_name          TEXT NOT NULL UNIQUE,
    organism                TEXT DEFAULT 'Homo sapiens',
    tissue                  TEXT,
    disease                 TEXT,
    assay_protocol_version  TEXT DEFAULT 'Cell Painting v3',
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE cp_assay_contexts (
    assay_context_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_line_id            INTEGER NOT NULL REFERENCES cp_cell_lines(cell_line_id),
    assay_name              TEXT NOT NULL DEFAULT 'Cell Painting',
    cell_painting_version   TEXT NOT NULL DEFAULT 'v3',
    timepoint_h             REAL NOT NULL,
    stain_channels          TEXT,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(cell_line_id, assay_name, cell_painting_version, timepoint_h)
);

CREATE TABLE cp_batches (
    batch_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id              INTEGER REFERENCES dataset_versions(dataset_id),
    batch_name              TEXT NOT NULL UNIQUE,
    source_name             TEXT,
    source_uri              TEXT,
    acquisition_date        TEXT,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE cp_plates (
    plate_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id                INTEGER NOT NULL REFERENCES cp_batches(batch_id),
    plate_name              TEXT NOT NULL,
    plate_barcode           TEXT,
    dmso_well_count         INTEGER DEFAULT 0,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(batch_id, plate_name)
);

CREATE TABLE cp_observations (
    observation_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_id                 INTEGER NOT NULL REFERENCES compounds(compound_id),
    plate_id                    INTEGER NOT NULL REFERENCES cp_plates(plate_id),
    assay_context_id            INTEGER NOT NULL REFERENCES cp_assay_contexts(assay_context_id),
    well_id                     TEXT NOT NULL,
    site_id                     INTEGER,
    replicate_id                INTEGER,
    dose                        REAL,
    dose_unit                   TEXT NOT NULL DEFAULT 'uM',
    timepoint_h                 REAL NOT NULL,
    control_type                TEXT NOT NULL DEFAULT 'perturbation'
                                CHECK (control_type IN ('perturbation', 'dmso', 'other')),
    dmso_distance               REAL,
    replicate_reproducibility   REAL,
    viability_ratio             REAL,
    qc_pass                     INTEGER NOT NULL DEFAULT 1 CHECK (qc_pass IN (0, 1)),
    image_uri                   TEXT,
    source_record_id            TEXT,
    created_at                  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_cp_obs_plate ON cp_observations(plate_id);
CREATE INDEX idx_cp_obs_compound ON cp_observations(compound_id);
CREATE INDEX idx_cp_obs_qc ON cp_observations(qc_pass);
CREATE INDEX idx_cp_obs_control ON cp_observations(control_type);
CREATE UNIQUE INDEX idx_cp_obs_unique_source
    ON cp_observations(plate_id, well_id, COALESCE(site_id, -1), COALESCE(replicate_id, -1), compound_id);

CREATE TABLE cp_perturbation_results (
    cp_result_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_id                  INTEGER NOT NULL REFERENCES compounds(compound_id),
    cell_line_id                 INTEGER NOT NULL REFERENCES cp_cell_lines(cell_line_id),
    assay_context_id             INTEGER NOT NULL REFERENCES cp_assay_contexts(assay_context_id),
    batch_id                     INTEGER NOT NULL REFERENCES cp_batches(batch_id),
    dose                         REAL,
    dose_unit                    TEXT NOT NULL DEFAULT 'uM',
    timepoint_h                  REAL NOT NULL,
    num_observations             INTEGER NOT NULL,
    num_valid_observations       INTEGER NOT NULL,
    dmso_distance_mean           REAL,
    replicate_reproducibility    REAL,
    viability_ratio              REAL,
    outcome_label                TEXT NOT NULL CHECK (outcome_label IN (
                                    'inactive', 'weak_phenotype',
                                    'strong_phenotype', 'toxic_or_artifact')),
    confidence_tier              TEXT NOT NULL CHECK (confidence_tier IN (
                                    'gold', 'silver', 'bronze', 'copper')),
    has_orthogonal_evidence      INTEGER NOT NULL DEFAULT 0 CHECK (has_orthogonal_evidence IN (0, 1)),
    created_at                   TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(compound_id, cell_line_id, dose, dose_unit, timepoint_h, batch_id)
);

CREATE INDEX idx_cp_results_compound ON cp_perturbation_results(compound_id);
CREATE INDEX idx_cp_results_batch ON cp_perturbation_results(batch_id);
CREATE INDEX idx_cp_results_label ON cp_perturbation_results(outcome_label);
CREATE INDEX idx_cp_results_tier ON cp_perturbation_results(confidence_tier);

CREATE TABLE cp_profile_features (
    feature_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    cp_result_id             INTEGER NOT NULL UNIQUE
                              REFERENCES cp_perturbation_results(cp_result_id) ON DELETE CASCADE,
    feature_source           TEXT,
    storage_uri              TEXT,
    feature_json             TEXT NOT NULL,
    n_features               INTEGER,
    created_at               TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE cp_image_features (
    feature_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    cp_result_id             INTEGER NOT NULL UNIQUE
                              REFERENCES cp_perturbation_results(cp_result_id) ON DELETE CASCADE,
    feature_source           TEXT,
    storage_uri              TEXT,
    feature_json             TEXT NOT NULL,
    n_features               INTEGER,
    created_at               TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE cp_raw_image_assets (
    image_asset_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id           INTEGER REFERENCES cp_observations(observation_id) ON DELETE CASCADE,
    cp_result_id             INTEGER REFERENCES cp_perturbation_results(cp_result_id) ON DELETE CASCADE,
    channel_name             TEXT,
    site_id                  INTEGER,
    image_uri                TEXT NOT NULL,
    sha256                   TEXT,
    subset_tag               TEXT
);

CREATE TABLE cp_orthogonal_evidence (
    evidence_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cp_result_id             INTEGER NOT NULL
                              REFERENCES cp_perturbation_results(cp_result_id) ON DELETE CASCADE,
    evidence_domain          TEXT NOT NULL,
    evidence_label           TEXT NOT NULL,
    source_name              TEXT,
    source_record_id         TEXT,
    match_key                TEXT,
    notes                    TEXT
);

CREATE TABLE cp_split_definitions (
    split_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    split_name               TEXT NOT NULL,
    split_strategy           TEXT NOT NULL CHECK (split_strategy IN (
                                   'random', 'cold_compound', 'scaffold', 'batch_holdout')),
    description              TEXT,
    random_seed              INTEGER,
    train_ratio              REAL DEFAULT 0.7,
    val_ratio                REAL DEFAULT 0.1,
    test_ratio               REAL DEFAULT 0.2,
    created_at               TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    version                  TEXT DEFAULT '1.0',
    UNIQUE(split_name, version)
);

CREATE TABLE cp_split_assignments (
    cp_result_id             INTEGER NOT NULL REFERENCES cp_perturbation_results(cp_result_id) ON DELETE CASCADE,
    split_id                 INTEGER NOT NULL REFERENCES cp_split_definitions(split_id) ON DELETE CASCADE,
    fold                     TEXT NOT NULL CHECK (fold IN ('train', 'val', 'test')),
    PRIMARY KEY (cp_result_id, split_id)
);

CREATE INDEX idx_cp_splits_fold ON cp_split_assignments(split_id, fold);

INSERT INTO schema_migrations (version) VALUES ('001');
