-- NegBioDB Metabolomics-Disease (MD) Domain — Initial Schema
-- Migration 001: Core tables for metabolite-disease non-association data
--
-- Design decisions:
--   - Core negative fact: metabolite X measured in disease context with no
--     significant differential expression (p > 0.05 or FDR > 0.1)
--   - Both positives AND negatives stored (is_significant flag); ML uses same studies
--   - Confidence tiers only assigned for negatives (is_significant=FALSE)
--   - Entity tables: md_metabolites, md_diseases, md_studies
--   - Fact table: md_biomarker_results (prefixed md_)
--   - Aggregation table: md_metabolite_disease_pairs (consensus across studies)
--   - Separate DB from other domains (negbiodb_md.db)
--   - HMDB data used ONLY for internal standardization; not redistributed (CC BY-NC 4.0)
--   - All exported metabolite identifiers sourced from PubChem (open license)
--   - Cross-domain bridge to DTI domain via InChIKey deferred to post-launch

-- ============================================================
-- Common Layer tables (same pattern as CT/PPI/GE/VP/DC/CP)
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
-- Entity tables (PREFIXED with md_ to avoid collisions)
-- ============================================================

-- Metabolites (endogenous small molecules and xenobiotics measured in studies)
-- License note: pubchem_cid + canonical_smiles from PubChem (open); hmdb_id for
--   internal cross-reference only — NOT redistributed as part of the benchmark.
CREATE TABLE md_metabolites (
    metabolite_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    -- Identifiers (PubChem = redistributable; HMDB = internal only)
    pubchem_cid     INTEGER,
    hmdb_id         TEXT,           -- internal standardization cache, NOT redistributed
    kegg_id         TEXT,
    chebi_id        TEXT,
    inchikey        TEXT,
    canonical_smiles TEXT,
    formula         TEXT,
    -- Chemical taxonomy (ClassyFire CC BY 4.0 — redistributable)
    metabolite_class TEXT,          -- ClassyFire superclass: Lipids, Amino acids, etc.
    metabolite_subclass TEXT,       -- ClassyFire class (more specific)
    -- Physicochemical properties (computed via RDKit)
    molecular_weight REAL,
    logp             REAL,
    tpsa             REAL,
    hbd              INTEGER,       -- H-bond donors
    hba              INTEGER,       -- H-bond acceptors
    created_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(inchikey),
    UNIQUE(pubchem_cid)
);

-- Diseases (standardized via MONDO ontology, CC0)
CREATE TABLE md_diseases (
    disease_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    -- Ontology identifiers
    mondo_id        TEXT,           -- MONDO Disease Ontology (CC0)
    mesh_id         TEXT,           -- MeSH (NLM, redistributable)
    doid            TEXT,           -- Disease Ontology ID
    icd10_code      TEXT,
    -- Category for ML features (one-hot)
    disease_category TEXT CHECK (disease_category IN (
        'cancer', 'metabolic', 'neurological', 'cardiovascular', 'other'
    )),
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(name)
);

-- Studies (MetaboLights / Metabolomics Workbench experimental studies)
-- Only studies with complete statistical tables (all measured metabolites with
-- p-values) are ingested — not just "top hits" lists.
CREATE TABLE md_studies (
    study_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL CHECK (source IN ('metabolights', 'nmdr')),
    external_id     TEXT NOT NULL,  -- MTBLS123 (MetaboLights) or ST000123 (NMDR)
    title           TEXT,
    description     TEXT,           -- Used as L2 context source
    -- Experimental design
    biofluid        TEXT CHECK (biofluid IN ('blood', 'urine', 'csf', 'tissue', 'other')),
    platform        TEXT CHECK (platform IN ('nmr', 'lc_ms', 'gc_ms', 'other')),
    comparison      TEXT,           -- e.g., 'disease_vs_healthy', 'treated_vs_control'
    -- Sample sizes
    n_disease       INTEGER,
    n_control       INTEGER,
    -- Reference
    pmid            INTEGER,
    doi             TEXT,
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(source, external_id)
);

-- ============================================================
-- Domain fact table (PREFIXED with md_)
-- ============================================================

-- Core fact: one row per metabolite × disease × study measurement
-- Both significant (is_significant=TRUE) and non-significant (FALSE) rows stored.
-- ML positives: is_significant=TRUE; ML negatives: is_significant=FALSE.
-- tier: only set for negatives (is_significant=FALSE); NULL for positives.
-- p_value/fdr: NULL for copper-tier rows where no statistics were reported.
CREATE TABLE md_biomarker_results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    metabolite_id   INTEGER NOT NULL REFERENCES md_metabolites(metabolite_id),
    disease_id      INTEGER NOT NULL REFERENCES md_diseases(disease_id),
    study_id        INTEGER NOT NULL REFERENCES md_studies(study_id),
    -- Effect statistics (NULL if not reported)
    fold_change     REAL,
    log2_fc         REAL,
    p_value         REAL,
    fdr             REAL,
    -- Classification
    is_significant  INTEGER NOT NULL DEFAULT 0 CHECK (is_significant IN (0, 1)),
    -- Tier: gold/silver/bronze/copper — only for negatives (is_significant=0)
    --   gold:   FDR > 0.1, n >= 50/group, replicated in >= 2 studies
    --   silver: p > 0.05, n >= 20/group
    --   bronze: p > 0.05, n < 20/group
    --   copper: metabolite present in study but no statistics reported
    tier            TEXT CHECK (tier IN ('gold', 'silver', 'bronze', 'copper')),
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================
-- Aggregation table: consensus across studies
-- ============================================================

-- Summary per metabolite × disease pair aggregated across all studies.
-- Computed by etl_aggregate.py / scripts_md/05_aggregate_pairs.py.
CREATE TABLE md_metabolite_disease_pairs (
    pair_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    metabolite_id       INTEGER NOT NULL REFERENCES md_metabolites(metabolite_id),
    disease_id          INTEGER NOT NULL REFERENCES md_diseases(disease_id),
    -- Study counts
    n_studies_total     INTEGER NOT NULL DEFAULT 0,
    n_studies_negative  INTEGER NOT NULL DEFAULT 0,  -- is_significant=0
    n_studies_positive  INTEGER NOT NULL DEFAULT 0,  -- is_significant=1
    -- Consensus (majority across studies)
    consensus           TEXT CHECK (consensus IN ('negative', 'positive', 'mixed')),
    best_tier           TEXT CHECK (best_tier IN ('gold', 'silver', 'bronze', 'copper')),
    -- Metabolite degree (number of diseases tested for this metabolite)
    metabolite_degree   INTEGER DEFAULT 0,
    -- Disease degree (number of metabolites tested for this disease)
    disease_degree      INTEGER DEFAULT 0,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(metabolite_id, disease_id)
);

-- ============================================================
-- Benchmark split tables (PREFIXED with md_)
-- ============================================================

CREATE TABLE md_split_definitions (
    split_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    split_name      TEXT NOT NULL,
    split_strategy  TEXT CHECK (split_strategy IN (
        'random', 'cold_metabolite', 'cold_disease', 'cold_both'
    )),
    description     TEXT,
    random_seed     INTEGER,
    train_ratio     REAL DEFAULT 0.7,
    val_ratio       REAL DEFAULT 0.1,
    test_ratio      REAL DEFAULT 0.2,
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    version         TEXT DEFAULT '1.0',
    UNIQUE(split_name, version)
);

CREATE TABLE md_split_assignments (
    split_id        INTEGER NOT NULL REFERENCES md_split_definitions(split_id),
    pair_id         INTEGER NOT NULL REFERENCES md_metabolite_disease_pairs(pair_id),
    fold            TEXT CHECK (fold IN ('train', 'val', 'test')),
    PRIMARY KEY (split_id, pair_id)
);

-- ============================================================
-- Indexes
-- ============================================================

-- Metabolite indexes
CREATE INDEX idx_md_metabolites_inchikey ON md_metabolites(inchikey) WHERE inchikey IS NOT NULL;
CREATE INDEX idx_md_metabolites_pubchem  ON md_metabolites(pubchem_cid) WHERE pubchem_cid IS NOT NULL;
CREATE INDEX idx_md_metabolites_hmdb     ON md_metabolites(hmdb_id) WHERE hmdb_id IS NOT NULL;
CREATE INDEX idx_md_metabolites_class    ON md_metabolites(metabolite_class);

-- Disease indexes
CREATE INDEX idx_md_diseases_mondo    ON md_diseases(mondo_id) WHERE mondo_id IS NOT NULL;
CREATE INDEX idx_md_diseases_mesh     ON md_diseases(mesh_id) WHERE mesh_id IS NOT NULL;
CREATE INDEX idx_md_diseases_category ON md_diseases(disease_category);

-- Study indexes
CREATE INDEX idx_md_studies_source      ON md_studies(source);
CREATE INDEX idx_md_studies_external    ON md_studies(source, external_id);
CREATE INDEX idx_md_studies_pmid        ON md_studies(pmid) WHERE pmid IS NOT NULL;
CREATE INDEX idx_md_studies_platform    ON md_studies(platform);
CREATE INDEX idx_md_studies_biofluid    ON md_studies(biofluid);

-- Fact table indexes
CREATE INDEX idx_md_br_metabolite    ON md_biomarker_results(metabolite_id);
CREATE INDEX idx_md_br_disease       ON md_biomarker_results(disease_id);
CREATE INDEX idx_md_br_study         ON md_biomarker_results(study_id);
CREATE INDEX idx_md_br_pair          ON md_biomarker_results(metabolite_id, disease_id);
CREATE INDEX idx_md_br_significant   ON md_biomarker_results(is_significant);
CREATE INDEX idx_md_br_tier          ON md_biomarker_results(tier) WHERE tier IS NOT NULL;

-- Pair table indexes
CREATE INDEX idx_md_pairs_metabolite ON md_metabolite_disease_pairs(metabolite_id);
CREATE INDEX idx_md_pairs_disease    ON md_metabolite_disease_pairs(disease_id);
CREATE INDEX idx_md_pairs_consensus  ON md_metabolite_disease_pairs(consensus);
CREATE INDEX idx_md_pairs_tier       ON md_metabolite_disease_pairs(best_tier);

-- Split indexes
CREATE INDEX idx_md_splits_fold      ON md_split_assignments(split_id, fold);

-- Record migration
INSERT INTO schema_migrations (version) VALUES ('001');
