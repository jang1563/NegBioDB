-- NegBioDB Clinical Trial Failure Domain — Initial Schema
-- Migration 001: Core tables for clinical trial failure tracking
--
-- Reuses Common Layer patterns from DTI domain:
--   - schema_migrations for version tracking
--   - dataset_versions for provenance
--   - Confidence tiers (gold/silver/bronze/copper)
--   - WAL journal mode + FK enforcement (set by ct_db.py)

-- ============================================================
-- Common Layer tables (same as DTI)
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
-- Domain-specific tables: Clinical Trial Failure
-- ============================================================

-- Interventions (drugs, biologics, devices, etc.)
CREATE TABLE interventions (
    intervention_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_type   TEXT NOT NULL CHECK (intervention_type IN (
        'drug', 'biologic', 'device', 'procedure', 'behavioral',
        'dietary', 'genetic', 'radiation', 'combination', 'other')),
    intervention_name   TEXT NOT NULL,
    canonical_name      TEXT,
    drugbank_id         TEXT,
    pubchem_cid         INTEGER,
    chembl_id           TEXT,
    mesh_id             TEXT,
    atc_code            TEXT,
    mechanism_of_action TEXT,
    canonical_smiles    TEXT,
    canonical_sequence  TEXT,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Conditions (diseases/indications)
CREATE TABLE conditions (
    condition_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_name      TEXT NOT NULL,
    canonical_name      TEXT,
    mesh_id             TEXT,
    icd10_code          TEXT,
    icd11_code          TEXT,
    do_id               TEXT,
    therapeutic_area    TEXT,
    condition_class     TEXT,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Molecular targets (bridge to DTI domain)
CREATE TABLE intervention_targets (
    intervention_id     INTEGER NOT NULL REFERENCES interventions(intervention_id),
    uniprot_accession   TEXT NOT NULL,
    gene_symbol         TEXT,
    target_role         TEXT CHECK (target_role IN (
        'primary', 'secondary', 'off_target')),
    action_type         TEXT,
    source              TEXT NOT NULL,
    PRIMARY KEY (intervention_id, uniprot_accession)
);

-- Clinical trials
CREATE TABLE clinical_trials (
    trial_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_db           TEXT NOT NULL CHECK (source_db IN (
        'clinicaltrials_gov', 'eu_ctr', 'who_ictrp', 'literature')),
    source_trial_id     TEXT NOT NULL,
    overall_status      TEXT NOT NULL,
    trial_phase         TEXT CHECK (trial_phase IN (
        'early_phase_1', 'phase_1', 'phase_1_2', 'phase_2',
        'phase_2_3', 'phase_3', 'phase_4', 'not_applicable')),
    study_type          TEXT,
    study_design        TEXT,
    blinding            TEXT,
    randomized          INTEGER DEFAULT 0,
    enrollment_target   INTEGER,
    enrollment_actual   INTEGER,
    primary_endpoint    TEXT,
    primary_endpoint_type TEXT,
    control_type        TEXT,
    sponsor_type        TEXT CHECK (sponsor_type IN (
        'industry', 'academic', 'government', 'other')),
    sponsor_name        TEXT,
    start_date          TEXT,
    primary_completion_date TEXT,
    completion_date     TEXT,
    results_posted_date TEXT,
    why_stopped         TEXT,
    has_results         INTEGER DEFAULT 0,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(source_db, source_trial_id)
);

-- Junction: trials <-> interventions (many-to-many)
CREATE TABLE trial_interventions (
    trial_id        INTEGER NOT NULL REFERENCES clinical_trials(trial_id),
    intervention_id INTEGER NOT NULL REFERENCES interventions(intervention_id),
    arm_group       TEXT,
    arm_role        TEXT CHECK (arm_role IN (
        'experimental', 'active_comparator', 'placebo_comparator', 'no_intervention')),
    dose_regimen    TEXT,
    PRIMARY KEY (trial_id, intervention_id)
);

-- Junction: trials <-> conditions (many-to-many)
CREATE TABLE trial_conditions (
    trial_id     INTEGER NOT NULL REFERENCES clinical_trials(trial_id),
    condition_id INTEGER NOT NULL REFERENCES conditions(condition_id),
    PRIMARY KEY (trial_id, condition_id)
);

-- Junction: trials <-> publications
CREATE TABLE trial_publications (
    trial_id   INTEGER NOT NULL REFERENCES clinical_trials(trial_id),
    pubmed_id  INTEGER NOT NULL,
    pub_type   TEXT,
    PRIMARY KEY (trial_id, pubmed_id)
);

-- Combination therapy decomposition
CREATE TABLE combination_components (
    combination_id  INTEGER NOT NULL REFERENCES interventions(intervention_id),
    component_id    INTEGER NOT NULL REFERENCES interventions(intervention_id),
    role            TEXT CHECK (role IN ('experimental', 'backbone', 'comparator')),
    PRIMARY KEY (combination_id, component_id)
);

-- Trial failure results (core fact table)
CREATE TABLE trial_failure_results (
    result_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_id     INTEGER NOT NULL REFERENCES interventions(intervention_id),
    condition_id        INTEGER NOT NULL REFERENCES conditions(condition_id),
    trial_id            INTEGER REFERENCES clinical_trials(trial_id),

    -- Hierarchical failure classification
    failure_category    TEXT NOT NULL CHECK (failure_category IN (
        'efficacy', 'safety', 'pharmacokinetic', 'enrollment',
        'strategic', 'regulatory', 'design', 'other')),
    failure_subcategory TEXT,
    failure_detail      TEXT,

    -- Confidence tier (reused from DTI)
    confidence_tier     TEXT NOT NULL CHECK (confidence_tier IN (
        'gold', 'silver', 'bronze', 'copper')),

    -- Arm-level context (multi-arm trials)
    arm_description     TEXT,
    arm_type            TEXT CHECK (arm_type IN (
        'experimental', 'active_comparator', 'placebo_comparator', 'overall')),

    -- Quantitative outcome data
    primary_endpoint_met    INTEGER,
    p_value_primary         REAL,
    effect_size             REAL,
    effect_size_type        TEXT,
    ci_lower                REAL,
    ci_upper                REAL,
    sample_size_treatment   INTEGER,
    sample_size_control     INTEGER,

    -- Safety signals
    serious_adverse_events  INTEGER,
    deaths_treatment        INTEGER,
    deaths_control          INTEGER,
    dsmb_stopped            INTEGER DEFAULT 0,

    -- Phase context
    highest_phase_reached   TEXT,
    prior_phase_succeeded   INTEGER DEFAULT 0,

    -- Provenance
    source_db           TEXT NOT NULL,
    source_record_id    TEXT NOT NULL,
    extraction_method   TEXT NOT NULL CHECK (extraction_method IN (
        'database_direct', 'nlp_classified', 'text_mining',
        'llm_extracted', 'community_submitted')),
    curator_validated   INTEGER DEFAULT 0,
    publication_year    INTEGER,

    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Aggregation table
CREATE TABLE intervention_condition_pairs (
    pair_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_id     INTEGER NOT NULL REFERENCES interventions(intervention_id),
    condition_id        INTEGER NOT NULL REFERENCES conditions(condition_id),
    num_trials          INTEGER NOT NULL,
    num_sources         INTEGER NOT NULL,
    best_confidence     TEXT NOT NULL,
    primary_failure_category TEXT,
    earliest_year       INTEGER,
    highest_phase_reached TEXT,
    has_any_approval    INTEGER DEFAULT 0,
    intervention_degree INTEGER,
    condition_degree    INTEGER,
    UNIQUE(intervention_id, condition_id)
);

-- Trial failure context (extended metadata)
CREATE TABLE trial_failure_context (
    result_id           INTEGER PRIMARY KEY REFERENCES trial_failure_results(result_id),
    patient_population  TEXT,
    biomarker_stratified INTEGER DEFAULT 0,
    companion_diagnostic TEXT,
    prior_treatment_lines INTEGER,
    comparator_drug     TEXT,
    geographic_regions  TEXT,
    regulatory_pathway  TEXT,
    genetic_evidence    INTEGER DEFAULT 0,
    class_effect_known  INTEGER DEFAULT 0,
    has_negbiodb_dti_data INTEGER DEFAULT 0
);

-- ============================================================
-- Indices for performance
-- ============================================================

CREATE INDEX idx_tfr_failure_category ON trial_failure_results(failure_category);
CREATE INDEX idx_tfr_confidence ON trial_failure_results(confidence_tier);
CREATE INDEX idx_tfr_intervention ON trial_failure_results(intervention_id);
CREATE INDEX idx_tfr_condition ON trial_failure_results(condition_id);
CREATE INDEX idx_tfr_trial ON trial_failure_results(trial_id);
CREATE INDEX idx_ct_status ON clinical_trials(overall_status);
CREATE INDEX idx_ct_phase ON clinical_trials(trial_phase);
CREATE INDEX idx_ct_completion ON clinical_trials(primary_completion_date);
CREATE INDEX idx_ct_source ON clinical_trials(source_db);
CREATE INDEX idx_interv_chembl ON interventions(chembl_id);
CREATE INDEX idx_interv_drugbank ON interventions(drugbank_id);
CREATE INDEX idx_interv_name ON interventions(intervention_name);
CREATE INDEX idx_cond_mesh ON conditions(mesh_id);
CREATE INDEX idx_cond_icd10 ON conditions(icd10_code);
CREATE INDEX idx_icp_intervention ON intervention_condition_pairs(intervention_id);
CREATE INDEX idx_icp_condition ON intervention_condition_pairs(condition_id);

-- Record migration
INSERT INTO schema_migrations (version) VALUES ('001');
