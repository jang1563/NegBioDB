-- NegBioDB Variant Pathogenicity (VP) Domain — Initial Schema
-- Migration 001: Core tables for ClinVar/gnomAD variant pathogenicity negatives
--
-- Design decisions:
--   - Asymmetric pairs: variant + disease (like DTI's compound x target)
--   - Entity pair: Variant x Disease
--   - Separate DB from DTI/CT/PPI/GE (negbiodb_vp.db)
--   - Confidence tiers: gold/silver/bronze/copper (ACMG-aligned + gnomAD BA1)
--   - Entity tables unprefixed (genes, variants, diseases)
--   - Domain-specific tables prefixed (vp_submissions, vp_negative_results, etc.)
--   - Aggregation table unprefixed (variant_disease_pairs)
--   - has_conflict flag on both fact table and aggregation table
--   - gnomAD AFs and computational scores stored on variants table (per-variant)
--   - Cross-domain bridge: vp_cross_domain_genes links to GE/PPI/DTI via Entrez ID

-- ============================================================
-- Common Layer tables (same as CT/PPI/GE)
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
-- Entity tables: Genes, Variants, Diseases (UNPREFIXED)
-- ============================================================

-- Genes table (gene-level context for variants)
CREATE TABLE genes (
    gene_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    entrez_id               INTEGER UNIQUE,
    gene_symbol             TEXT NOT NULL,
    hgnc_id                 TEXT,
    ensembl_id              TEXT,
    description             TEXT,
    -- gnomAD gene constraint metrics (populated by gnomAD ETL)
    pli_score               REAL,
    loeuf_score             REAL,
    missense_z              REAL,
    -- ClinGen gene-disease validity (populated by ClinGen ETL)
    clingen_validity        TEXT CHECK (clingen_validity IS NULL OR clingen_validity IN (
        'Definitive', 'Strong', 'Moderate', 'Limited',
        'Disputed', 'Refuted', 'No Known Disease Relationship')),
    gene_moi                TEXT CHECK (gene_moi IS NULL OR gene_moi IN (
        'AD', 'AR', 'XL', 'XLD', 'XLR', 'MT', 'Other', 'Unknown')),
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_genes_symbol ON genes(gene_symbol);
CREATE INDEX idx_genes_entrez ON genes(entrez_id) WHERE entrez_id IS NOT NULL;
CREATE INDEX idx_genes_hgnc ON genes(hgnc_id) WHERE hgnc_id IS NOT NULL;

-- Variants table (genomic variant loci with annotations)
CREATE TABLE variants (
    variant_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    clinvar_variation_id    INTEGER,           -- NULL for copper-tier gnomAD-only variants
    chromosome              TEXT NOT NULL,
    position                INTEGER NOT NULL,
    ref_allele              TEXT NOT NULL,
    alt_allele              TEXT NOT NULL,
    variant_type            TEXT NOT NULL CHECK (variant_type IN (
        'single nucleotide variant', 'Deletion', 'Insertion',
        'Indel', 'Duplication', 'other')),
    gene_id                 INTEGER REFERENCES genes(gene_id),
    rs_id                   INTEGER,           -- dbSNP rsID
    hgvs_coding             TEXT,              -- e.g., NM_000059.4:c.5123C>A
    hgvs_protein            TEXT,              -- e.g., NP_000050.3:p.Ala1708Asp
    consequence_type        TEXT CHECK (consequence_type IS NULL OR consequence_type IN (
        'missense', 'nonsense', 'synonymous', 'frameshift',
        'splice', 'inframe_indel', 'intronic', 'other')),
    -- gnomAD allele frequencies (populated by gnomAD ETL)
    gnomad_af_global        REAL,
    gnomad_af_afr           REAL,
    gnomad_af_amr           REAL,
    gnomad_af_asj           REAL,
    gnomad_af_eas           REAL,
    gnomad_af_fin           REAL,
    gnomad_af_nfe           REAL,
    gnomad_af_sas           REAL,
    gnomad_af_oth           REAL,
    -- Computational scores (populated by score ETL on HPC)
    cadd_phred              REAL,
    revel_score             REAL,              -- missense only; NULL for non-missense
    alphamissense_score     REAL,              -- missense only; NULL for non-missense
    alphamissense_class     TEXT CHECK (alphamissense_class IS NULL OR alphamissense_class IN (
        'likely_pathogenic', 'ambiguous', 'likely_benign')),
    phylop_score            REAL,
    gerp_score              REAL,
    sift_score              REAL,
    polyphen2_score         REAL,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE UNIQUE INDEX idx_variants_locus ON variants(chromosome, position, ref_allele, alt_allele);
CREATE INDEX idx_variants_clinvar ON variants(clinvar_variation_id) WHERE clinvar_variation_id IS NOT NULL;
CREATE INDEX idx_variants_gene ON variants(gene_id) WHERE gene_id IS NOT NULL;
CREATE INDEX idx_variants_rs ON variants(rs_id) WHERE rs_id IS NOT NULL;
CREATE INDEX idx_variants_type ON variants(variant_type);
CREATE INDEX idx_variants_consequence ON variants(consequence_type) WHERE consequence_type IS NOT NULL;

-- Diseases table (conditions / phenotypes)
CREATE TABLE diseases (
    disease_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    medgen_cui              TEXT,              -- MedGen concept unique identifier
    omim_id                 TEXT,
    orphanet_id             TEXT,
    mondo_id                TEXT,
    canonical_name          TEXT NOT NULL,
    inheritance_pattern     TEXT CHECK (inheritance_pattern IS NULL OR inheritance_pattern IN (
        'AD', 'AR', 'XL', 'XLD', 'XLR', 'MT',
        'multifactorial', 'complex', 'other', 'unknown')),
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_diseases_medgen ON diseases(medgen_cui) WHERE medgen_cui IS NOT NULL;
CREATE INDEX idx_diseases_omim ON diseases(omim_id) WHERE omim_id IS NOT NULL;
CREATE INDEX idx_diseases_name ON diseases(canonical_name);

-- ============================================================
-- Domain-specific tables (PREFIXED with vp_)
-- ============================================================

-- Per-submission provenance (SCV accessions from ClinVar submission_summary)
CREATE TABLE vp_submissions (
    submission_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    scv_accession           TEXT UNIQUE,       -- SCV accession ID
    variant_id              INTEGER NOT NULL REFERENCES variants(variant_id),
    submitter_name          TEXT,
    submitter_id            INTEGER,           -- ClinVar OrgID
    classification          TEXT NOT NULL CHECK (classification IN (
        'benign', 'likely_benign', 'uncertain_significance',
        'likely_pathogenic', 'pathogenic',
        'benign/likely_benign', 'pathogenic/likely_pathogenic',
        'conflicting', 'other')),
    review_status           TEXT NOT NULL,
    date_last_evaluated     TEXT,              -- ISO date
    submission_year         INTEGER,
    reported_phenotype      TEXT,
    acmg_criteria           TEXT,              -- JSON array e.g. '["BA1","BS1","BP4"]'
    method                  TEXT,
    comment                 TEXT,
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_vp_sub_variant ON vp_submissions(variant_id);
CREATE INDEX idx_vp_sub_submitter ON vp_submissions(submitter_name) WHERE submitter_name IS NOT NULL;
CREATE INDEX idx_vp_sub_classification ON vp_submissions(classification);

-- Core fact table: VP negative results (benign/likely benign variant-disease pairs)
CREATE TABLE vp_negative_results (
    result_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_id              INTEGER NOT NULL REFERENCES variants(variant_id),
    disease_id              INTEGER NOT NULL REFERENCES diseases(disease_id),
    submission_id           INTEGER REFERENCES vp_submissions(submission_id),

    classification          TEXT NOT NULL CHECK (classification IN (
        'benign', 'likely_benign', 'benign/likely_benign')),

    evidence_type           TEXT NOT NULL CHECK (evidence_type IN (
        'expert_reviewed',
        'multi_submitter_concordant',
        'single_submitter',
        'population_frequency',
        'computational_only')),

    confidence_tier         TEXT NOT NULL CHECK (confidence_tier IN (
        'gold', 'silver', 'bronze', 'copper')),

    source_db               TEXT NOT NULL CHECK (source_db IN (
        'clinvar', 'gnomad')),
    source_record_id        TEXT NOT NULL,     -- VariationID or gnomAD locus key
    extraction_method       TEXT NOT NULL CHECK (extraction_method IN (
        'review_status', 'submitter_concordance',
        'single_submission', 'population_af_threshold')),

    submission_year         INTEGER,
    has_conflict            INTEGER DEFAULT 0, -- 1 if any P/LP submission exists for same variant-disease
    num_benign_criteria     INTEGER DEFAULT 0, -- count of ACMG benign criteria (from submissions)

    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX idx_vp_nr_variant ON vp_negative_results(variant_id);
CREATE INDEX idx_vp_nr_disease ON vp_negative_results(disease_id);
CREATE INDEX idx_vp_nr_pair ON vp_negative_results(variant_id, disease_id);
CREATE INDEX idx_vp_nr_tier ON vp_negative_results(confidence_tier);
CREATE INDEX idx_vp_nr_source ON vp_negative_results(source_db);
CREATE INDEX idx_vp_nr_year ON vp_negative_results(submission_year) WHERE submission_year IS NOT NULL;

CREATE UNIQUE INDEX idx_vp_nr_unique_source ON vp_negative_results(
    variant_id, disease_id,
    COALESCE(submission_id, -1),
    source_db);

-- ============================================================
-- Aggregation table (UNPREFIXED — follows gene_cell_pairs pattern)
-- ============================================================

CREATE TABLE variant_disease_pairs (
    pair_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_id              INTEGER NOT NULL REFERENCES variants(variant_id),
    disease_id              INTEGER NOT NULL REFERENCES diseases(disease_id),
    num_submissions         INTEGER NOT NULL,
    num_submitters          INTEGER NOT NULL,
    best_confidence         TEXT NOT NULL CHECK (best_confidence IN (
        'gold', 'silver', 'bronze', 'copper')),
    best_evidence_type      TEXT,
    best_classification     TEXT CHECK (best_classification IN (
        'benign', 'likely_benign', 'benign/likely_benign')),
    earliest_year           INTEGER,
    has_conflict            INTEGER DEFAULT 0,
    max_population_af       REAL,
    num_benign_criteria     INTEGER DEFAULT 0,
    variant_degree          INTEGER,           -- # diseases for this variant
    disease_degree          INTEGER,           -- # variants for this disease
    UNIQUE(variant_id, disease_id)
);

CREATE INDEX idx_vdp_variant ON variant_disease_pairs(variant_id);
CREATE INDEX idx_vdp_disease ON variant_disease_pairs(disease_id);
CREATE INDEX idx_vdp_confidence ON variant_disease_pairs(best_confidence);
CREATE INDEX idx_vdp_year ON variant_disease_pairs(earliest_year) WHERE earliest_year IS NOT NULL;

-- ============================================================
-- Benchmark split tables (PREFIXED)
-- ============================================================

CREATE TABLE vp_split_definitions (
    split_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    split_name              TEXT NOT NULL,
    split_strategy          TEXT NOT NULL CHECK (split_strategy IN (
        'random', 'cold_gene', 'cold_disease',
        'cold_both', 'degree_balanced', 'temporal')),
    description             TEXT,
    random_seed             INTEGER,
    train_ratio             REAL DEFAULT 0.7,
    val_ratio               REAL DEFAULT 0.1,
    test_ratio              REAL DEFAULT 0.2,
    date_created            TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    version                 TEXT DEFAULT '1.0',
    UNIQUE(split_name, version)
);

CREATE TABLE vp_split_assignments (
    pair_id                 INTEGER NOT NULL REFERENCES variant_disease_pairs(pair_id),
    split_id                INTEGER NOT NULL REFERENCES vp_split_definitions(split_id),
    fold                    TEXT NOT NULL CHECK (fold IN ('train', 'val', 'test')),
    PRIMARY KEY (pair_id, split_id)
);

CREATE INDEX idx_vp_splits_fold ON vp_split_assignments(split_id, fold);

-- ============================================================
-- Cross-domain bridge table
-- ============================================================

CREATE TABLE vp_cross_domain_genes (
    bridge_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    gene_id                 INTEGER NOT NULL REFERENCES genes(gene_id),
    domain                  TEXT NOT NULL CHECK (domain IN ('ge', 'ppi', 'dti')),
    external_id             TEXT NOT NULL,     -- Entrez ID (GE), UniProt (PPI), or target_id (DTI)
    external_table          TEXT,              -- e.g., 'genes', 'proteins', 'targets'
    mapping_method          TEXT DEFAULT 'entrez_id',
    created_at              TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(gene_id, domain, external_id)
);

CREATE INDEX idx_vp_bridge_gene ON vp_cross_domain_genes(gene_id);
CREATE INDEX idx_vp_bridge_domain ON vp_cross_domain_genes(domain);

-- Record migration
INSERT INTO schema_migrations (version) VALUES ('001');
