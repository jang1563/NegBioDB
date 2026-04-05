-- NegBioDB Drug Combination Synergy (DC) Domain — Initial Schema
-- Migration 001: Core tables for DrugComb/NCI-ALMANAC/AZ-DREAM drug combination data
--
-- Design decisions:
--   - Symmetric pairs: Drug A x Drug B (compound_a_id < compound_b_id enforced)
--   - Tripartite detail: Drug A x Drug B x Cell Line
--   - Separate DB from DTI/CT/PPI/GE/VP (negbiodb_dc.db)
--   - Confidence tiers: gold/silver/bronze/copper (multi-source to single-conc)
--   - Entity tables unprefixed (compounds, cell_lines, drug_targets)
--   - Domain-specific tables prefixed (dc_synergy_results, dc_split_*)
--   - Aggregation tables unprefixed (drug_drug_pairs, drug_drug_cell_line_triples)
--   - Cross-domain bridges: dc_cross_domain_compounds (→DTI/CT), dc_cross_domain_cell_lines (→GE)

-- ============================================================
-- Common Layer tables (same as CT/PPI/GE/VP)
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
-- Entity tables (UNPREFIXED)
-- ============================================================

-- Drugs (compound-level, mapped to PubChem for DTI cross-linking)
CREATE TABLE compounds (
    compound_id INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_name TEXT NOT NULL,
    pubchem_cid INTEGER,
    inchikey TEXT,
    canonical_smiles TEXT,
    chembl_id TEXT,
    drugbank_id TEXT,
    molecular_weight REAL,
    molecular_formula TEXT,
    -- Known targets (comma-separated gene symbols for quick lookup)
    known_targets TEXT,
    -- ATC code for therapeutic classification
    atc_code TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(drug_name)
);

-- Cell lines (mapped to DepMap for GE cross-linking)
CREATE TABLE cell_lines (
    cell_line_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_line_name TEXT NOT NULL,
    cosmic_id INTEGER,
    depmap_model_id TEXT,
    tissue TEXT,
    cancer_type TEXT,
    -- From DepMap if available
    primary_disease TEXT,
    lineage TEXT,
    lineage_subtype TEXT,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(cell_line_name)
);

-- Drug targets (for target overlap features)
CREATE TABLE drug_targets (
    compound_id INTEGER NOT NULL REFERENCES compounds(compound_id),
    gene_symbol TEXT NOT NULL,
    uniprot_accession TEXT,
    source TEXT CHECK (source IN ('chembl', 'drugbank', 'dgidb')),
    PRIMARY KEY (compound_id, gene_symbol, source)
);

-- ============================================================
-- Domain fact table (PREFIXED with dc_)
-- ============================================================

-- Raw synergy measurements (one row per drug_pair x cell_line x source)
CREATE TABLE dc_synergy_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_a_id INTEGER NOT NULL REFERENCES compounds(compound_id),
    compound_b_id INTEGER NOT NULL REFERENCES compounds(compound_id),
    cell_line_id INTEGER NOT NULL REFERENCES cell_lines(cell_line_id),
    -- Synergy scores (NULL if not computed by source)
    zip_score REAL,
    bliss_score REAL,
    loewe_score REAL,
    hsa_score REAL,
    css_score REAL,          -- Combination Sensitivity Score
    combo_score REAL,        -- NCI-ALMANAC ComboScore
    s_score REAL,            -- DrugComb S score
    -- Classification (derived from scores)
    synergy_class TEXT CHECK (synergy_class IN (
        'strongly_synergistic', 'synergistic', 'additive',
        'antagonistic', 'strongly_antagonistic'
    )),
    -- Tier assignment
    confidence_tier TEXT CHECK (confidence_tier IN ('gold', 'silver', 'bronze', 'copper')),
    evidence_type TEXT CHECK (evidence_type IN (
        'multi_source_concordant', 'multi_cell_line', 'dose_response_matrix',
        'single_concentration', 'computational'
    )),
    -- Source tracking
    source_db TEXT NOT NULL CHECK (source_db IN ('drugcomb', 'nci_almanac', 'az_dream')),
    source_study_id TEXT,
    -- Dose-response metadata
    num_concentrations INTEGER,
    has_dose_matrix INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    -- Ordering constraint: compound_a_id < compound_b_id (symmetric pair normalization)
    CHECK (compound_a_id < compound_b_id)
);

-- ============================================================
-- Aggregation tables (UNPREFIXED — follows gene_cell_pairs / variant_disease_pairs pattern)
-- ============================================================

-- Primary analysis unit: Drug A x Drug B (aggregated across cell lines)
CREATE TABLE drug_drug_pairs (
    pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_a_id INTEGER NOT NULL REFERENCES compounds(compound_id),
    compound_b_id INTEGER NOT NULL REFERENCES compounds(compound_id),
    -- Aggregation stats
    num_cell_lines INTEGER,
    num_sources INTEGER,
    num_measurements INTEGER,
    -- Consensus synergy (median across cell lines)
    median_zip REAL,
    median_bliss REAL,
    -- Concordance: fraction of cell lines showing antagonism
    antagonism_fraction REAL,
    synergy_fraction REAL,
    -- Consensus class (majority vote across cell lines)
    consensus_class TEXT CHECK (consensus_class IN (
        'synergistic', 'additive', 'antagonistic', 'context_dependent'
    )),
    best_confidence TEXT CHECK (best_confidence IN ('gold', 'silver', 'bronze', 'copper')),
    -- Target overlap (precomputed)
    num_shared_targets INTEGER DEFAULT 0,
    target_jaccard REAL DEFAULT 0.0,
    -- Degrees
    compound_a_degree INTEGER,  -- How many partners does drug A have?
    compound_b_degree INTEGER,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    -- Ordering constraint
    CHECK (compound_a_id < compound_b_id),
    UNIQUE(compound_a_id, compound_b_id)
);

-- Detailed triple for cell-line-specific analysis
CREATE TABLE drug_drug_cell_line_triples (
    triple_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id INTEGER NOT NULL REFERENCES drug_drug_pairs(pair_id),
    cell_line_id INTEGER NOT NULL REFERENCES cell_lines(cell_line_id),
    -- Best measurement for this triple
    best_zip REAL,
    best_bliss REAL,
    num_measurements INTEGER,
    synergy_class TEXT,
    confidence_tier TEXT,
    UNIQUE(pair_id, cell_line_id)
);

-- ============================================================
-- Benchmark split tables (PREFIXED with dc_)
-- ============================================================

CREATE TABLE dc_split_definitions (
    split_id INTEGER PRIMARY KEY AUTOINCREMENT,
    split_name TEXT NOT NULL,
    split_strategy TEXT CHECK (split_strategy IN (
        'random', 'cold_compound', 'cold_cell_line', 'cold_both',
        'scaffold', 'leave_one_tissue_out'
    )),
    description TEXT,
    random_seed INTEGER,
    train_ratio REAL DEFAULT 0.7,
    val_ratio REAL DEFAULT 0.1,
    test_ratio REAL DEFAULT 0.2,
    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    version TEXT DEFAULT '1.0',
    UNIQUE(split_name, version)
);

CREATE TABLE dc_split_assignments (
    split_id INTEGER NOT NULL REFERENCES dc_split_definitions(split_id),
    pair_id INTEGER NOT NULL REFERENCES drug_drug_pairs(pair_id),
    fold TEXT CHECK (fold IN ('train', 'val', 'test')),
    PRIMARY KEY (split_id, pair_id)
);

-- ============================================================
-- Cross-domain bridge tables (following VP pattern: external_id TEXT, no cross-DB FK)
-- ============================================================

-- Link DC compounds to DTI/CT compounds (resolves in Python, not DB-level FK)
CREATE TABLE dc_cross_domain_compounds (
    bridge_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_id     INTEGER NOT NULL REFERENCES compounds(compound_id),
    domain          TEXT NOT NULL CHECK (domain IN ('dti', 'ct')),
    external_id     TEXT NOT NULL,     -- InChIKey (DTI), or intervention name (CT)
    external_table  TEXT,              -- e.g., 'compounds', 'interventions'
    mapping_method  TEXT DEFAULT 'inchikey',
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(compound_id, domain, external_id)
);

-- Link DC cell lines to GE cell lines (resolves in Python, not DB-level FK)
CREATE TABLE dc_cross_domain_cell_lines (
    bridge_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    cell_line_id    INTEGER NOT NULL REFERENCES cell_lines(cell_line_id),
    domain          TEXT NOT NULL CHECK (domain IN ('ge')),
    external_id     TEXT NOT NULL,     -- DepMap model_id or COSMIC ID
    external_table  TEXT DEFAULT 'cell_lines',
    mapping_method  TEXT DEFAULT 'depmap_model_id',
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(cell_line_id, domain, external_id)
);

-- ============================================================
-- Indexes (following VP pattern: ~25 indexes for query performance)
-- ============================================================

-- Entity indexes
CREATE INDEX idx_compounds_pubchem ON compounds(pubchem_cid) WHERE pubchem_cid IS NOT NULL;
CREATE INDEX idx_compounds_inchikey ON compounds(inchikey) WHERE inchikey IS NOT NULL;
CREATE INDEX idx_compounds_chembl ON compounds(chembl_id) WHERE chembl_id IS NOT NULL;
CREATE INDEX idx_cell_lines_cosmic ON cell_lines(cosmic_id) WHERE cosmic_id IS NOT NULL;
CREATE INDEX idx_cell_lines_depmap ON cell_lines(depmap_model_id) WHERE depmap_model_id IS NOT NULL;
CREATE INDEX idx_cell_lines_tissue ON cell_lines(tissue);

-- Drug targets indexes
CREATE INDEX idx_drug_targets_gene ON drug_targets(gene_symbol);

-- Fact table indexes
CREATE INDEX idx_dc_sr_compound_a ON dc_synergy_results(compound_a_id);
CREATE INDEX idx_dc_sr_compound_b ON dc_synergy_results(compound_b_id);
CREATE INDEX idx_dc_sr_pair ON dc_synergy_results(compound_a_id, compound_b_id);
CREATE INDEX idx_dc_sr_cell_line ON dc_synergy_results(cell_line_id);
CREATE INDEX idx_dc_sr_tier ON dc_synergy_results(confidence_tier);
CREATE INDEX idx_dc_sr_source ON dc_synergy_results(source_db);
CREATE INDEX idx_dc_sr_class ON dc_synergy_results(synergy_class);

-- Pair table indexes
CREATE INDEX idx_ddp_compound_a ON drug_drug_pairs(compound_a_id);
CREATE INDEX idx_ddp_compound_b ON drug_drug_pairs(compound_b_id);
CREATE INDEX idx_ddp_confidence ON drug_drug_pairs(best_confidence);
CREATE INDEX idx_ddp_class ON drug_drug_pairs(consensus_class);

-- Triple table indexes
CREATE INDEX idx_ddclt_pair ON drug_drug_cell_line_triples(pair_id);
CREATE INDEX idx_ddclt_cell_line ON drug_drug_cell_line_triples(cell_line_id);

-- Split indexes
CREATE INDEX idx_dc_splits_fold ON dc_split_assignments(split_id, fold);

-- Cross-domain bridge indexes
CREATE INDEX idx_dc_bridge_compound ON dc_cross_domain_compounds(compound_id);
CREATE INDEX idx_dc_bridge_compound_domain ON dc_cross_domain_compounds(domain);
CREATE INDEX idx_dc_bridge_cell_line ON dc_cross_domain_cell_lines(cell_line_id);

-- Record migration
INSERT INTO schema_migrations (version) VALUES ('001');
