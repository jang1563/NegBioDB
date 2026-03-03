# Schema Design, ML Export Patterns, and NeurIPS Compliance

> Practical guidance for NegBioDB SQLite schema, ML-ready data exports, Croissant metadata, and Datasheet for Datasets.
> Last updated: 2026-03-02

---

## 1. SQLite Schema Design for NegBioDB

### 1.1 Design Principles

1. **Normalize for the database, denormalize for export.** The SQLite schema uses proper relational design (3NF). ML-ready CSV/Parquet exports are pre-joined flat tables.
2. **Common layer + domain layer.** Tables shared across all future domains (compounds, targets, results) are separated from DTI-specific tables. This matches the architecture in PROJECT_OVERVIEW.md.
3. **InChIKey is the compound dedup key; UniProt accession is the target dedup key.** These are indexed, not used as primary keys (because they can change or have edge cases). Surrogate integer PKs are used instead for performance.
4. **Every record tracks provenance.** Source database, source ID, extraction method, and curator validation status are first-class columns.

### 1.2 Complete SQLite Schema (DDL)

```sql
-- ============================================================
-- NegBioDB Schema v1.0
-- Database: SQLite 3.35+
-- ============================================================

PRAGMA journal_mode = WAL;          -- Better concurrent reads
PRAGMA foreign_keys = ON;           -- Enforce referential integrity
PRAGMA encoding = 'UTF-8';

-- ============================================================
-- COMMON LAYER: Shared across all scientific domains
-- ============================================================

-- ------- Compounds -------
CREATE TABLE compounds (
    compound_id     INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Structural identifiers (canonical, standardized by RDKit)
    canonical_smiles    TEXT NOT NULL,
    inchikey            TEXT NOT NULL,           -- Full 27-char InChIKey
    inchikey_connectivity TEXT NOT NULL,         -- First 14 chars (dedup key)
    inchi               TEXT,                    -- Full InChI string

    -- Cross-database identifiers (nullable; populated as available)
    pubchem_cid         INTEGER,
    chembl_id           TEXT,                    -- e.g., 'CHEMBL25'
    bindingdb_id        INTEGER,

    -- Computed properties (from RDKit at standardization time)
    molecular_weight    REAL,
    logp                REAL,                    -- Crippen LogP
    hbd                 INTEGER,                 -- H-bond donors
    hba                 INTEGER,                 -- H-bond acceptors
    tpsa                REAL,                    -- Topological PSA
    rotatable_bonds     INTEGER,
    num_heavy_atoms     INTEGER,
    qed                 REAL,                    -- Quantitative drug-likeness

    -- Quality flags
    pains_alert         INTEGER DEFAULT 0,       -- 0/1 boolean
    aggregator_alert    INTEGER DEFAULT 0,       -- 0/1 boolean
    lipinski_violations INTEGER DEFAULT 0,

    -- Metadata
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Compound indexes
CREATE UNIQUE INDEX idx_compounds_inchikey ON compounds(inchikey);
CREATE INDEX idx_compounds_connectivity ON compounds(inchikey_connectivity);
CREATE INDEX idx_compounds_pubchem ON compounds(pubchem_cid) WHERE pubchem_cid IS NOT NULL;
CREATE INDEX idx_compounds_chembl ON compounds(chembl_id) WHERE chembl_id IS NOT NULL;
CREATE INDEX idx_compounds_smiles ON compounds(canonical_smiles);

-- ------- Targets -------
CREATE TABLE targets (
    target_id           INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Canonical identifier
    uniprot_accession   TEXT NOT NULL,           -- e.g., 'P00533'
    uniprot_entry_name  TEXT,                    -- e.g., 'EGFR_HUMAN'

    -- Sequence data
    amino_acid_sequence TEXT,                    -- Full protein sequence
    sequence_length     INTEGER,

    -- Cross-database identifiers
    chembl_target_id    TEXT,                    -- e.g., 'CHEMBL203'
    gene_symbol         TEXT,                    -- e.g., 'EGFR'
    ncbi_gene_id        INTEGER,

    -- Target classification (DTO / IDG)
    target_family       TEXT,                    -- kinase, GPCR, ion_channel, nuclear_receptor, etc.
    target_subfamily    TEXT,                    -- e.g., 'tyrosine kinase'
    dto_class           TEXT,                    -- Drug Target Ontology class
    development_level   TEXT CHECK (development_level IN ('Tclin', 'Tchem', 'Tbio', 'Tdark')),

    -- Organism
    organism            TEXT DEFAULT 'Homo sapiens',
    taxonomy_id         INTEGER DEFAULT 9606,

    -- Metadata
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Target indexes
CREATE UNIQUE INDEX idx_targets_uniprot ON targets(uniprot_accession);
CREATE INDEX idx_targets_chembl ON targets(chembl_target_id) WHERE chembl_target_id IS NOT NULL;
CREATE INDEX idx_targets_gene ON targets(gene_symbol) WHERE gene_symbol IS NOT NULL;
CREATE INDEX idx_targets_family ON targets(target_family);
CREATE INDEX idx_targets_dev_level ON targets(development_level);

-- ------- Assays -------
CREATE TABLE assays (
    assay_id            INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Source assay identifier
    source_db           TEXT NOT NULL CHECK (source_db IN (
                            'pubchem', 'chembl', 'bindingdb', 'literature', 'community')),
    source_assay_id     TEXT NOT NULL,           -- e.g., AID 1234 or CHEMBL1234567

    -- Assay classification (BAO-based)
    assay_type          TEXT,                    -- BAO term if available
    assay_format        TEXT CHECK (assay_format IN (
                            'biochemical', 'cell-based', 'in_vivo', 'unknown')),
    assay_technology    TEXT,                    -- e.g., 'fluorescence', 'AlphaScreen'
    detection_method    TEXT,

    -- Screening tier
    screen_type         TEXT CHECK (screen_type IN (
                            'primary_single_point', 'confirmatory_dose_response',
                            'counter_screen', 'orthogonal_assay', 'literature_assay', 'unknown')),

    -- Quality metrics (assay-level)
    z_factor            REAL,
    ssmd                REAL,

    -- Assay context
    cell_line           TEXT,                    -- If cell-based
    description         TEXT,
    pubmed_id           INTEGER,                 -- Linked publication
    doi                 TEXT,

    -- Metadata
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE UNIQUE INDEX idx_assays_source ON assays(source_db, source_assay_id);
CREATE INDEX idx_assays_format ON assays(assay_format);
CREATE INDEX idx_assays_screen ON assays(screen_type);

-- ============================================================
-- DTI DOMAIN LAYER: Drug-Target Interaction specifics
-- ============================================================

-- ------- Negative Results (the core table) -------
-- Each row = one compound-target measurement from one assay
CREATE TABLE negative_results (
    result_id           INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Foreign keys
    compound_id         INTEGER NOT NULL REFERENCES compounds(compound_id),
    target_id           INTEGER NOT NULL REFERENCES targets(target_id),
    assay_id            INTEGER REFERENCES assays(assay_id),

    -- Result classification
    result_type         TEXT NOT NULL CHECK (result_type IN (
                            'hard_negative',           -- Confirmed no binding/activity
                            'conditional_negative',    -- Inactive under specific conditions
                            'methodological_negative', -- Inactive in this assay type only
                            'dose_time_negative',      -- Active only at extreme dose/time
                            'hypothesis_negative'      -- Predicted mechanism not observed
                        )),
    confidence_tier     TEXT NOT NULL CHECK (confidence_tier IN (
                            'gold', 'silver', 'bronze', 'copper')),

    -- Quantitative activity data
    activity_type       TEXT,                    -- IC50, Ki, Kd, EC50, %inhibition, etc.
    activity_value      REAL,                    -- Raw value
    activity_unit       TEXT,                    -- nM, uM, %, etc.
    activity_relation   TEXT DEFAULT '=',        -- =, >, <, ~
    pchembl_value       REAL,                    -- Negative log molar (-log10(M))

    -- Inactivity determination
    inactivity_threshold    REAL,                -- Threshold used (e.g., 10000 nM)
    inactivity_threshold_unit TEXT DEFAULT 'nM',
    max_concentration_tested REAL,               -- Highest conc. tested

    -- Experimental context
    num_replicates      INTEGER,
    species_tested      TEXT DEFAULT 'Homo sapiens',

    -- Provenance
    source_db           TEXT NOT NULL,
    source_record_id    TEXT NOT NULL,           -- Original record ID in source DB
    extraction_method   TEXT NOT NULL CHECK (extraction_method IN (
                            'database_direct',     -- Direct DB query
                            'text_mining',         -- Automated text extraction
                            'llm_extracted',       -- LLM pipeline extraction
                            'community_submitted'  -- User contribution
                        )),
    curator_validated   INTEGER DEFAULT 0,       -- 0/1 boolean

    -- Temporal data (for temporal splits)
    publication_year    INTEGER,                 -- Year of original publication/deposition
    deposition_date     TEXT,                    -- ISO date when deposited to source DB

    -- Metadata
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Core query indexes
CREATE INDEX idx_results_compound ON negative_results(compound_id);
CREATE INDEX idx_results_target ON negative_results(target_id);
CREATE INDEX idx_results_pair ON negative_results(compound_id, target_id);
CREATE INDEX idx_results_tier ON negative_results(confidence_tier);
CREATE INDEX idx_results_source ON negative_results(source_db);
CREATE INDEX idx_results_year ON negative_results(publication_year);
CREATE INDEX idx_results_type ON negative_results(result_type);

-- Prevent exact duplicate records from same source
-- NOTE: COALESCE handles NULL assay_id — SQLite treats NULL as distinct
-- in UNIQUE indexes, which would allow duplicate rows when assay_id is missing.
CREATE UNIQUE INDEX idx_results_unique_source ON negative_results(
    compound_id, target_id, COALESCE(assay_id, -1), source_db, source_record_id);

-- ------- DTI-Specific Context (extends negative_results) -------
CREATE TABLE dti_context (
    result_id           INTEGER PRIMARY KEY REFERENCES negative_results(result_id),

    binding_site        TEXT CHECK (binding_site IN (
                            'orthosteric', 'allosteric', 'unknown')),
    selectivity_panel   INTEGER DEFAULT 0,       -- Part of selectivity screen?
    counterpart_active  INTEGER DEFAULT 0,       -- Active in other species/conditions?
    cell_permeability_issue INTEGER DEFAULT 0,
    compound_solubility REAL,                    -- uM
    compound_stability  TEXT                     -- stable, unstable, unknown
);

-- ------- Compound-Target Pair Aggregation -------
-- Materialized view: one row per unique compound-target pair,
-- aggregating across all assays. Rebuilt during export.
CREATE TABLE compound_target_pairs (
    pair_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    compound_id         INTEGER NOT NULL REFERENCES compounds(compound_id),
    target_id           INTEGER NOT NULL REFERENCES targets(target_id),

    -- Aggregated from negative_results
    num_assays          INTEGER NOT NULL,        -- How many assays tested this pair
    num_sources         INTEGER NOT NULL,        -- How many DBs confirm this
    best_confidence     TEXT NOT NULL,            -- Highest tier achieved
    best_result_type    TEXT,                     -- Most specific result type (for export)
    earliest_year       INTEGER,                 -- Earliest publication year (for temporal splits + export)

    -- Consensus activity
    median_pchembl      REAL,
    min_activity_value  REAL,                    -- Most potent (lowest IC50)
    max_activity_value  REAL,                    -- Least potent (highest IC50)

    -- Conflict detection
    has_conflicting_results INTEGER DEFAULT 0,   -- Some assays say active?

    -- Pre-computed for ML splits
    compound_degree     INTEGER,                 -- Number of targets for this compound
    target_degree       INTEGER,                 -- Number of compounds for this target

    UNIQUE(compound_id, target_id)
);

CREATE INDEX idx_pairs_compound ON compound_target_pairs(compound_id);
CREATE INDEX idx_pairs_target ON compound_target_pairs(target_id);
CREATE INDEX idx_pairs_confidence ON compound_target_pairs(best_confidence);

-- ============================================================
-- BENCHMARK / ML LAYER: Split assignments and ML metadata
-- ============================================================

-- ------- Split Definitions -------
CREATE TABLE split_definitions (
    split_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    split_name          TEXT NOT NULL,            -- e.g., 'random_v1', 'cold_compound_v1'
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

-- ------- Split Assignments -------
-- Many-to-many: each pair can be in multiple splits
CREATE TABLE split_assignments (
    pair_id             INTEGER NOT NULL REFERENCES compound_target_pairs(pair_id),
    split_id            INTEGER NOT NULL REFERENCES split_definitions(split_id),
    fold                TEXT NOT NULL CHECK (fold IN ('train', 'val', 'test')),

    PRIMARY KEY (pair_id, split_id)
);

CREATE INDEX idx_splits_fold ON split_assignments(split_id, fold);

-- ------- Dataset Versions -------
CREATE TABLE dataset_versions (
    version_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    version_tag         TEXT NOT NULL UNIQUE,     -- e.g., 'v0.1-alpha', 'v1.0'
    description         TEXT,
    num_compounds       INTEGER,
    num_targets         INTEGER,
    num_pairs           INTEGER,
    num_results         INTEGER,
    schema_version      TEXT,                     -- e.g., '1.0'
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    checksum_sha256     TEXT                      -- Of the exported DB file
);

-- ============================================================
-- METADATA LAYER: Schema version tracking
-- ============================================================

CREATE TABLE schema_migrations (
    migration_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    version             TEXT NOT NULL,
    description         TEXT,
    applied_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    sql_up              TEXT,                     -- SQL that was applied
    sql_down            TEXT                      -- SQL to reverse (if possible)
);

-- Insert initial migration (version matches filename prefix '001')
INSERT INTO schema_migrations (version, description, sql_up) VALUES
    ('001', 'Initial NegBioDB schema', 'Full DDL');
```

### 1.3 Indexing Strategy Rationale

The indexes above are designed for the most common query patterns:

| Query Pattern | Index Used |
|---|---|
| Look up compound by InChIKey | `idx_compounds_inchikey` (UNIQUE) |
| Deduplicate compounds by connectivity | `idx_compounds_connectivity` |
| Look up compound by PubChem CID or ChEMBL ID | `idx_compounds_pubchem`, `idx_compounds_chembl` |
| Look up target by UniProt accession | `idx_targets_uniprot` (UNIQUE) |
| Find all results for a compound | `idx_results_compound` |
| Find all results for a target | `idx_results_target` |
| Find result for specific compound-target pair | `idx_results_pair` (compound_id, target_id) |
| Filter by confidence tier | `idx_results_tier` |
| Filter by source database | `idx_results_source` |
| Temporal split queries | `idx_results_year` |
| Prevent duplicate source records | `idx_results_unique_source` (UNIQUE) |
| ML split generation | `idx_pairs_compound`, `idx_pairs_target` |
| Export by split + fold | `idx_splits_fold` |

**SQLite-specific notes:**
- SQLite uses B-tree indexes by default; these are efficient for both exact lookups and range queries.
- `WHERE column IS NOT NULL` partial indexes save space for sparse cross-reference columns.
- WAL mode (`PRAGMA journal_mode = WAL`) allows concurrent readers during data loading.

### 1.4 Handling the Many-to-Many Relationship

The compound-target relationship is inherently many-to-many, with additional complexity:

```
Compound 1 ←→ Target A (via Assay X, Assay Y, Assay Z)
Compound 1 ←→ Target B (via Assay X)
Compound 2 ←→ Target A (via Assay W)
```

The schema handles this with a **three-level structure**:

1. **`negative_results`**: One row per measurement (compound + target + assay). This is the finest granularity. The same compound-target pair can have multiple rows if tested in multiple assays.

2. **`compound_target_pairs`**: One row per unique compound-target pair. Aggregates across all assays. This is the unit used for ML benchmarking (each "sample" in the dataset is one pair).

3. **`split_assignments`**: Links pairs to benchmark splits. A pair appears once per split strategy (not once per fold).

**Populating compound_target_pairs** (run after all data ingestion):

```sql
INSERT INTO compound_target_pairs
    (compound_id, target_id, num_assays, num_sources, best_confidence,
     best_result_type, earliest_year,
     median_pchembl, min_activity_value, max_activity_value, has_conflicting_results,
     compound_degree, target_degree)
SELECT
    nr.compound_id,
    nr.target_id,
    COUNT(DISTINCT nr.assay_id) AS num_assays,
    COUNT(DISTINCT nr.source_db) AS num_sources,
    -- Best confidence: gold > silver > bronze > copper
    CASE
        WHEN SUM(CASE WHEN nr.confidence_tier = 'gold' THEN 1 ELSE 0 END) > 0 THEN 'gold'
        WHEN SUM(CASE WHEN nr.confidence_tier = 'silver' THEN 1 ELSE 0 END) > 0 THEN 'silver'
        WHEN SUM(CASE WHEN nr.confidence_tier = 'bronze' THEN 1 ELSE 0 END) > 0 THEN 'bronze'
        ELSE 'copper'
    END AS best_confidence,
    -- Best result type: hard > conditional > methodological > dose_time > hypothesis
    CASE
        WHEN SUM(CASE WHEN nr.result_type = 'hard_negative' THEN 1 ELSE 0 END) > 0 THEN 'hard_negative'
        WHEN SUM(CASE WHEN nr.result_type = 'conditional_negative' THEN 1 ELSE 0 END) > 0 THEN 'conditional_negative'
        WHEN SUM(CASE WHEN nr.result_type = 'methodological_negative' THEN 1 ELSE 0 END) > 0 THEN 'methodological_negative'
        WHEN SUM(CASE WHEN nr.result_type = 'dose_time_negative' THEN 1 ELSE 0 END) > 0 THEN 'dose_time_negative'
        ELSE 'hypothesis_negative'
    END AS best_result_type,
    MIN(nr.publication_year) AS earliest_year,
    -- Aggregated activity
    NULL AS median_pchembl,  -- Computed in Python (SQLite lacks MEDIAN)
    MIN(nr.activity_value) AS min_activity_value,
    MAX(nr.activity_value) AS max_activity_value,
    0 AS has_conflicting_results,
    -- Degree (populated in a second pass)
    0 AS compound_degree,
    0 AS target_degree
FROM negative_results nr
GROUP BY nr.compound_id, nr.target_id;

-- Update degrees
UPDATE compound_target_pairs SET compound_degree = (
    SELECT COUNT(DISTINCT target_id) FROM compound_target_pairs cp2
    WHERE cp2.compound_id = compound_target_pairs.compound_id
);

UPDATE compound_target_pairs SET target_degree = (
    SELECT COUNT(DISTINCT compound_id) FROM compound_target_pairs cp2
    WHERE cp2.target_id = compound_target_pairs.target_id
);
```

---

## 2. ML-Ready Export Patterns

### 2.1 How Successful Projects Structure Their Data

**TDC (Therapeutics Data Commons) -- NeurIPS 2021:**
- DataFrame columns: `Drug_ID`, `Drug` (SMILES), `Target_ID`, `Target` (amino acid sequence), `Y` (label/score)
- Split via: `data.get_split()` returns `{'train': df, 'val': df, 'test': df}`
- Supports: Random, Cold Drug, Cold Protein splits
- Distribution: Python package (`pip install PyTDC`)
- Source: [TDC DTI Tasks](https://tdcommons.ai/multi_pred_tasks/dti/)

**MoleculeNet / DeepChem -- 2018:**
- SMILES + property columns in CSV
- Loader returns `(tasks, (train, val, test), transformers)` tuple
- Multi-task: multiple Y columns per compound
- Built-in splitters: random, scaffold, stratified
- Source: [MoleculeNet](https://moleculenet.org/datasets-1)

**WelQrate -- NeurIPS 2024:**
- Multiple molecular representations: SDF, 2D Graph, 3D Graph
- Isomeric SMILES + InChI for text representations
- Activity labels (binary: active/inactive) with extreme class imbalance
- Scaffold-based and random splits
- Source: [WelQrate Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/5f2f8305cd1c5be7e8319aea306388ce-Paper-Datasets_and_Benchmarks_Track.pdf)

### 2.2 NegBioDB Standard Column Names

Following conventions from TDC and MoleculeNet, the exported flat CSV/Parquet files use these column names:

```
# Primary ML export: negbiodb_dti_pairs.csv
#
# One row per compound-target pair (the ML "sample")
#
pair_id,                    # Unique integer ID
compound_id,                # Internal compound ID
smiles,                     # Canonical SMILES (RDKit-standardized)
inchikey,                   # Full InChIKey
target_id,                  # Internal target ID
uniprot_id,                 # UniProt accession (e.g., P00533)
target_sequence,            # Full amino acid sequence
gene_symbol,                # e.g., EGFR
Y,                          # Label: 0 = inactive (negative), 1 = active
confidence_tier,            # gold, silver, bronze, copper
num_assays,                 # Number of supporting assays
num_sources,                # Number of independent databases
result_type,                # hard_negative, conditional_negative, etc.
target_family,              # kinase, GPCR, ion_channel, etc.
publication_year,           # Earliest publication year
split_random,               # train / val / test
split_cold_compound,        # train / val / test
split_cold_target,          # train / val / test
split_temporal,             # train / val / test
split_scaffold,             # train / val / test
split_ddb,                  # train / val / test (Degree Distribution Balanced)
```

**Column naming conventions in cheminformatics ML:**

| Convention | Examples | Used By |
|---|---|---|
| `Drug` / `Target` / `Y` | TDC standard | TDC |
| `smiles` / `activity` / `split` | MoleculeNet style | DeepChem |
| `SMILES` / `Label` | Kaggle convention | Various competitions |
| `mol_smiles` / `protein_seq` / `label` | Explicit naming | Research papers |

NegBioDB uses lowercase_snake_case throughout (consistent with Python/pandas conventions) and provides both TDC-compatible column aliases in the Python library.

### 2.3 CSV vs Parquet vs HDF5: Recommendation

| Format | Read Speed | Size | Schema | Ecosystem | Best For |
|---|---|---|---|---|---|
| **CSV** | Slowest | Largest | None | Universal | Human inspection, small datasets |
| **Parquet** | Fast (columnar) | 2-10x smaller | Embedded | Pandas, Spark, Arrow | Primary ML distribution |
| **HDF5** | Fast (row) | Small | Embedded | NumPy, PyTorch | Pre-computed embeddings, tensors |

**NegBioDB recommendation: Ship CSV + Parquet, generate HDF5 on demand.**

```python
# Export script: scripts/09_export_benchmark.py

import sqlite3
import pandas as pd

def export_ml_dataset(db_path: str, output_dir: str):
    """Export NegBioDB SQLite to ML-ready flat files."""
    conn = sqlite3.connect(db_path)

    # NOTE: This query exports NegBioDB negatives (Y=0) only.
    # For M1 binary DTI prediction, positive data (Y=1) must be merged separately
    # from ChEMBL (pChEMBL >= 6). See ROADMAP §Positive Data Protocol and
    # merge_positive_negative() below.
    query = """
    SELECT
        ctp.pair_id,
        ctp.compound_id,                     -- Internal compound ID
        c.canonical_smiles AS smiles,
        c.inchikey,
        ctp.target_id,                       -- Internal target ID
        t.uniprot_accession AS uniprot_id,
        t.amino_acid_sequence AS target_sequence,
        t.gene_symbol,
        0 AS Y,                              -- All NegBioDB entries are negatives (inactive)
        ctp.best_confidence AS confidence_tier,
        ctp.best_result_type AS result_type,
        ctp.num_assays,
        ctp.num_sources,
        t.target_family,
        ctp.earliest_year AS publication_year,
        -- Split assignments (pivoted)
        MAX(CASE WHEN sd.split_strategy = 'random' THEN sa.fold END) AS split_random,
        MAX(CASE WHEN sd.split_strategy = 'cold_compound' THEN sa.fold END) AS split_cold_compound,
        MAX(CASE WHEN sd.split_strategy = 'cold_target' THEN sa.fold END) AS split_cold_target,
        MAX(CASE WHEN sd.split_strategy = 'temporal' THEN sa.fold END) AS split_temporal,
        MAX(CASE WHEN sd.split_strategy = 'scaffold' THEN sa.fold END) AS split_scaffold,
        MAX(CASE WHEN sd.split_strategy = 'degree_balanced' THEN sa.fold END) AS split_ddb
    FROM compound_target_pairs ctp
    JOIN compounds c ON ctp.compound_id = c.compound_id
    JOIN targets t ON ctp.target_id = t.target_id
    LEFT JOIN split_assignments sa ON ctp.pair_id = sa.pair_id
    LEFT JOIN split_definitions sd ON sa.split_id = sd.split_id
    GROUP BY ctp.pair_id
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # CSV (human-readable, universal compatibility)
    csv_path = f"{output_dir}/negbiodb_dti_pairs.csv"
    df.to_csv(csv_path, index=False)

    # Parquet (ML-optimized: smaller, faster, typed)
    parquet_path = f"{output_dir}/negbiodb_dti_pairs.parquet"
    df.to_parquet(parquet_path, index=False, engine='pyarrow')

    # Also export a splits-only file (lightweight for users who
    # already downloaded the main data and just need new splits)
    splits_df = df[['pair_id', 'smiles', 'uniprot_id',
                     'split_random', 'split_cold_compound',
                     'split_cold_target', 'split_temporal',
                     'split_scaffold', 'split_ddb']]
    splits_df.to_csv(f"{output_dir}/negbiodb_splits.csv", index=False)

    return df


def merge_positive_negative(negatives_path: str, positives_path: str,
                             output_dir: str, ratio: str = 'balanced'):
    """Merge NegBioDB negatives (Y=0) with ChEMBL positives (Y=1) for M1 task.

    Positive data must be pre-extracted from ChEMBL (pChEMBL >= 6, shared
    target pool). See ROADMAP §Positive Data Protocol for extraction SQL.

    Args:
        negatives_path: Path to negbiodb_dti_pairs.parquet (Y=0)
        positives_path: Path to chembl_positives.parquet (Y=1)
        output_dir: Output directory
        ratio: 'balanced' (1:1) or 'realistic' (1:10 active:inactive)
    """
    neg = pd.read_parquet(negatives_path)
    pos = pd.read_parquet(positives_path)

    # Verify no compound-target overlap
    neg_pairs = set(zip(neg['inchikey'], neg['uniprot_id']))
    pos_pairs = set(zip(pos['inchikey'], pos['uniprot_id']))
    overlap = neg_pairs & pos_pairs
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping pairs!"

    # Sample according to ratio
    if ratio == 'balanced':
        n = min(len(neg), len(pos))
        neg_sample = neg.sample(n=n, random_state=42)
        pos_sample = pos.sample(n=n, random_state=42)
    elif ratio == 'realistic':
        # 1:10 active:inactive
        n_pos = min(len(pos), len(neg) // 10)
        pos_sample = pos.sample(n=n_pos, random_state=42)
        neg_sample = neg.sample(n=n_pos * 10, random_state=42)

    combined = pd.concat([neg_sample, pos_sample], ignore_index=True).sample(frac=1, random_state=42)
    combined.to_parquet(f"{output_dir}/negbiodb_m1_{ratio}.parquet", index=False)
    combined.to_csv(f"{output_dir}/negbiodb_m1_{ratio}.csv", index=False)
    print(f"Exported {len(combined)} pairs ({ratio}): {len(pos_sample)} pos + {len(neg_sample)} neg")
    return combined
```

### 2.4 Embedding Split Assignments in the Schema

There are two common patterns for storing splits. NegBioDB uses **Pattern B** (separate table) in the database, but **Pattern A** (inline columns) in the exported CSV/Parquet.

**Pattern A: Inline columns (in export files)**
```
pair_id, smiles, uniprot_id, Y, split_random, split_cold_compound, ...
1,       CC(=O)..., P00533,  0, train,        test,                 ...
2,       CC(O)...,  P04629,  0, val,          train,                ...
```
- Pros: Single file, easy to filter in pandas (`df[df.split_random == 'train']`)
- Cons: Adding a new split requires re-exporting
- Used by: Most Kaggle datasets, WelQrate

**Pattern B: Separate split table (in database)**
```
-- split_assignments table
pair_id | split_id | fold
1       | 1        | train
1       | 2        | test
2       | 1        | val
2       | 2        | train
```
- Pros: Can add splits without modifying main data; versioning per split
- Cons: Requires a join for ML loading
- Used by: NegBioDB internal, OpenML

**NegBioDB ships both**: Pattern B in SQLite (for extensibility), Pattern A in CSV/Parquet (for convenience). The export script above performs the pivot.

### 2.5 Python Library Usage Pattern (TDC-compatible)

```python
# User-facing API (future pip install negbiodb)
from negbiodb import NegBioDB

# Load dataset
db = NegBioDB()  # Downloads + caches locally

# Get full data as DataFrame
df = db.get_data()
# Columns: smiles, inchikey, uniprot_id, target_sequence, gene_symbol,
#           Y, confidence_tier, num_assays, num_sources, target_family

# Get pre-defined splits
splits = db.get_split(strategy='cold_compound')
# Returns: {'train': pd.DataFrame, 'val': pd.DataFrame, 'test': pd.DataFrame}

# Filter by confidence
gold_data = db.get_data(confidence='gold')

# Filter by target family
kinase_negatives = db.get_data(target_family='kinase')

# TDC-compatible column names (alias)
df_tdc = db.get_data(format='tdc')
# Columns: Drug_ID, Drug, Target_ID, Target, Y
```

---

## 3. Extensibility Considerations

### 3.1 Domain Layer Pattern

The schema separates concerns into layers that can grow independently:

```
┌─────────────────────────────────────────────────────────┐
│  COMMON LAYER (shared tables)                           │
│  ┌───────────┐  ┌──────────┐  ┌───────────┐           │
│  │ compounds │  │ targets  │  │ assays    │           │
│  └───────────┘  └──────────┘  └───────────┘           │
│  ┌───────────────────────┐  ┌───────────────────────┐  │
│  │ dataset_versions      │  │ schema_migrations     │  │
│  └───────────────────────┘  └───────────────────────┘  │
└─────────────────────────────┬───────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌──────────────────┐  ┌─────────────────┐
│ DTI Domain    │   │ Gene Function    │  │ Clinical Trial  │
│ Layer         │   │ Domain Layer     │  │ Domain Layer    │
│               │   │                  │  │                 │
│ negative_     │   │ ko_results       │  │ trial_outcomes  │
│   results     │   │ kd_results       │  │ trial_compounds │
│ dti_context   │   │ phenotype_data   │  │ patient_cohorts │
│ compound_     │   │ gene_context     │  │ clinical_       │
│   target_     │   │                  │  │   context       │
│   pairs       │   │                  │  │                 │
│ split_        │   │                  │  │                 │
│   definitions │   │                  │  │                 │
│ split_        │   │                  │  │                 │
│   assignments │   │                  │  │                 │
└───────────────┘   └──────────────────┘  └─────────────────┘
```

**Rules for adding a new domain:**

1. **Reuse `compounds` and `targets` tables.** New domains may add compounds or targets that do not yet exist. Insert them into the common tables.

2. **Create domain-specific result tables** that reference `compounds` and/or `targets` via foreign keys.

3. **Create domain-specific context tables** for additional metadata (analogous to `dti_context`).

4. **Create domain-specific `split_definitions` and `split_assignments`** if the domain has its own benchmark (or share the existing tables with a domain column).

**Example: Gene Function Domain Layer (Phase 2)**

```sql
-- Domain: Gene Function (CRISPR KO/KD negatives)
CREATE TABLE ko_results (
    result_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id       INTEGER NOT NULL REFERENCES targets(target_id),

    -- KO/KD specifics
    perturbation_type   TEXT CHECK (perturbation_type IN ('knockout', 'knockdown', 'overexpression')),
    cell_line           TEXT,
    phenotype_expected  TEXT,        -- What was expected
    phenotype_observed  TEXT,        -- What was observed (negative = no effect)
    effect_size         REAL,
    p_value             REAL,

    -- Provenance
    source_db           TEXT,        -- DepMap, literature, etc.
    source_id           TEXT,
    confidence_tier     TEXT CHECK (confidence_tier IN ('gold', 'silver', 'bronze', 'copper')),

    created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
```

### 3.2 Schema Versioning

For a SQLite-based project, we use a lightweight migration system rather than Alembic (which is overkill for a single-file database distributed as a dataset).

**Strategy: Numbered SQL migration files + a migrations table.**

```
migrations/
├── 001_initial_schema.sql
├── 002_add_scaffold_column.sql
├── 003_add_gene_function_domain.sql
└── ...
```

Each migration file:
```sql
-- migrations/002_add_scaffold_column.sql
-- Description: Add Murcko scaffold to compounds for scaffold splitting

ALTER TABLE compounds ADD COLUMN murcko_scaffold TEXT;
CREATE INDEX idx_compounds_scaffold ON compounds(murcko_scaffold)
    WHERE murcko_scaffold IS NOT NULL;

-- Record migration (version MUST match filename prefix for idempotency)
INSERT INTO schema_migrations (version, description) VALUES
    ('002', 'Add Murcko scaffold column to compounds');
```

> **IMPORTANT — Migration version format:** The `version` value in `schema_migrations` MUST use the filename prefix format (`001`, `002`, ...), NOT semantic versioning (`1.0`, `1.1`). The Python runner extracts the prefix from the filename and checks it against the DB. A format mismatch will cause migrations to be reapplied on every run.

**Python migration runner:**

```python
import sqlite3
import glob
import os

def run_migrations(db_path: str, migrations_dir: str):
    """Apply pending migrations to a NegBioDB SQLite database."""
    conn = sqlite3.connect(db_path)

    # Get already-applied migrations
    applied = set()
    try:
        rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
        applied = {row[0] for row in rows}
    except sqlite3.OperationalError:
        pass  # Table does not exist yet; first migration will create it

    # Apply pending migrations in order
    migration_files = sorted(glob.glob(os.path.join(migrations_dir, "*.sql")))
    for mf in migration_files:
        version = os.path.basename(mf).split("_")[0]  # e.g., "002"
        if version not in applied:
            with open(mf) as f:
                sql = f.read()
            conn.executescript(sql)
            print(f"Applied migration {version}: {os.path.basename(mf)}")

    conn.close()
```

**For distributed dataset files:** The `dataset_versions` table records each release. Users can check `SELECT version_tag FROM dataset_versions ORDER BY created_at DESC LIMIT 1` to know which version they have.

---

## 4. Croissant Metadata Format

### 4.1 What Is Required for NeurIPS 2025/2026

Per the [NeurIPS 2025 Datasets & Benchmarks Call for Papers](https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks) and the [NeurIPS 2025 blog post on raising the bar](https://blog.neurips.cc/2025/03/10/neurips-datasets-benchmarks-raising-the-bar-for-dataset-submissions/):

**Mandatory:**
- Dataset must be hosted on a public repository (Hugging Face, Kaggle, OpenML, Dataverse, or custom hosting with long-term access)
- A **Croissant machine-readable metadata file** (JSON-LD) must be included
- Dataset and code must be accessible to reviewers at submission time
- Code must be on GitHub/Bitbucket in executable format

**Recommended but not strictly mandatory:**
- Datasheet for Datasets (Gebru et al.)
- Dataset Nutrition Label
- Data cards

**If using Hugging Face, Kaggle, OpenML, or Dataverse:** Croissant metadata is auto-generated. For custom hosting, you must generate it yourself and can validate it with the [Croissant online validator](https://docs.mlcommons.org/croissant/).

### 4.2 Croissant Structure

Croissant has four layers ([spec](https://docs.mlcommons.org/croissant/docs/croissant-spec.html)):

1. **Metadata**: Dataset-level info (name, description, license, creators)
2. **Resources**: Files that contain the data (`FileObject`, `FileSet`)
3. **Structure**: How raw data maps to records and fields (`RecordSet`, `Field`)
4. **ML Semantics**: How the data is used in ML (`default_slice`, split info)

### 4.3 NegBioDB Croissant JSON-LD Template

```json
{
  "@context": {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "data": {
      "@id": "cr:data",
      "@type": "@json"
    },
    "dataType": {
      "@id": "cr:dataType",
      "@type": "@vocab"
    },
    "dct": "http://purl.org/dc/terms/",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileObject": "cr:fileObject",
    "fileProperty": "cr:fileProperty",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "sha256": "cr:sha256",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform"
  },
  "@type": "sc:Dataset",
  "conformsTo": "http://mlcommons.org/croissant/1.0",
  "name": "NegBioDB",
  "description": "NegBioDB is a curated database of experimentally confirmed negative drug-target interactions (DTIs), designed as both a scientific resource and an ML-ready benchmark. It addresses publication bias in DTI prediction by providing high-quality inactive compound-target pairs sourced from PubChem BioAssay, ChEMBL, BindingDB, and the DAVIS kinase matrix. Each entry includes confidence tiers (gold/silver/bronze/copper), assay context, and provenance. Pre-defined train/val/test splits support multiple evaluation strategies.",
  "license": "https://creativecommons.org/licenses/by-sa/4.0/",
  "url": "https://huggingface.co/datasets/negbiodb/negbiodb-dti",
  "citeAs": "@inproceedings{negbiodb2026, title={NegBioDB: A Negative Results Database and Benchmark for Drug-Target Interaction Prediction}, author={...}, booktitle={NeurIPS Datasets and Benchmarks Track}, year={2026}}",
  "datePublished": "2026-05-01",
  "creator": [
    {
      "@type": "Person",
      "name": "Author Name",
      "affiliation": "Institution"
    }
  ],
  "keywords": [
    "drug-target interaction",
    "negative results",
    "cheminformatics",
    "benchmark",
    "publication bias",
    "virtual screening"
  ],
  "isLiveDataset": false,
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "dti-pairs-csv",
      "name": "negbiodb_dti_pairs.csv",
      "description": "Main dataset: one row per compound-target pair with SMILES, UniProt ID, sequence, labels, confidence tiers, and split assignments.",
      "contentUrl": "https://huggingface.co/datasets/negbiodb/negbiodb-dti/resolve/main/negbiodb_dti_pairs.csv",
      "encodingFormat": "text/csv",
      "sha256": "PLACEHOLDER_SHA256"
    },
    {
      "@type": "cr:FileObject",
      "@id": "dti-pairs-parquet",
      "name": "negbiodb_dti_pairs.parquet",
      "description": "Same data as CSV in Apache Parquet format for faster loading.",
      "contentUrl": "https://huggingface.co/datasets/negbiodb/negbiodb-dti/resolve/main/negbiodb_dti_pairs.parquet",
      "encodingFormat": "application/x-parquet",
      "sha256": "PLACEHOLDER_SHA256"
    },
    {
      "@type": "cr:FileObject",
      "@id": "sqlite-db",
      "name": "negbiodb.sqlite",
      "description": "Full relational database with all assay-level detail, provenance, and domain context.",
      "contentUrl": "https://huggingface.co/datasets/negbiodb/negbiodb-dti/resolve/main/negbiodb.sqlite",
      "encodingFormat": "application/x-sqlite3",
      "sha256": "PLACEHOLDER_SHA256"
    }
  ],
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "@id": "dti_pairs",
      "name": "dti_pairs",
      "description": "Drug-target interaction pairs with activity labels, confidence tiers, and split assignments.",
      "field": [
        {
          "@type": "cr:Field",
          "@id": "dti_pairs/pair_id",
          "name": "pair_id",
          "description": "Unique identifier for this compound-target pair.",
          "dataType": "sc:Integer",
          "source": {
            "fileObject": { "@id": "dti-pairs-csv" },
            "extract": { "column": "pair_id" }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "dti_pairs/smiles",
          "name": "smiles",
          "description": "Canonical SMILES of the compound (RDKit-standardized).",
          "dataType": "sc:Text",
          "source": {
            "fileObject": { "@id": "dti-pairs-csv" },
            "extract": { "column": "smiles" }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "dti_pairs/inchikey",
          "name": "inchikey",
          "description": "InChIKey of the compound (27 characters).",
          "dataType": "sc:Text",
          "source": {
            "fileObject": { "@id": "dti-pairs-csv" },
            "extract": { "column": "inchikey" }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "dti_pairs/uniprot_id",
          "name": "uniprot_id",
          "description": "UniProt accession of the target protein.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": { "@id": "dti-pairs-csv" },
            "extract": { "column": "uniprot_id" }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "dti_pairs/target_sequence",
          "name": "target_sequence",
          "description": "Full amino acid sequence of the target protein.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": { "@id": "dti-pairs-csv" },
            "extract": { "column": "target_sequence" }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "dti_pairs/Y",
          "name": "Y",
          "description": "Activity label: 0 = inactive (confirmed negative). When combined with positive data, 1 = active.",
          "dataType": "sc:Integer",
          "source": {
            "fileObject": { "@id": "dti-pairs-csv" },
            "extract": { "column": "Y" }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "dti_pairs/confidence_tier",
          "name": "confidence_tier",
          "description": "Quality confidence: gold (dose-response, multiple sources), silver (dose-response, single source), bronze (single-point, filtered), copper (minimal evidence).",
          "dataType": "sc:Text",
          "source": {
            "fileObject": { "@id": "dti-pairs-csv" },
            "extract": { "column": "confidence_tier" }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "dti_pairs/split_random",
          "name": "split_random",
          "description": "Random split assignment: train, val, or test (70/10/20).",
          "dataType": "sc:Text",
          "source": {
            "fileObject": { "@id": "dti-pairs-csv" },
            "extract": { "column": "split_random" }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "dti_pairs/split_cold_compound",
          "name": "split_cold_compound",
          "description": "Cold-compound split: test compounds never seen in training.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": { "@id": "dti-pairs-csv" },
            "extract": { "column": "split_cold_compound" }
          }
        },
        {
          "@type": "cr:Field",
          "@id": "dti_pairs/split_cold_target",
          "name": "split_cold_target",
          "description": "Cold-target split: test targets never seen in training.",
          "dataType": "sc:Text",
          "source": {
            "fileObject": { "@id": "dti-pairs-csv" },
            "extract": { "column": "split_cold_target" }
          }
        }
      ]
    }
  ]
}
```

### 4.4 Generating Croissant from SQLite

**Option 1: Use `mlcroissant` Python library (recommended)**

```python
import mlcroissant as mlc

# Validate an existing Croissant file
dataset = mlc.Dataset("metadata.json")
# If this runs without error, the Croissant file is valid

# Programmatically build a Croissant file
# (mlcroissant supports creating metadata from scratch)
# See: https://github.com/mlcommons/croissant/blob/main/python/mlcroissant/recipes/introduction.ipynb
```

**Option 2: Upload to Hugging Face, get auto-generated Croissant**

If you host on Hugging Face (recommended for NeurIPS), Croissant metadata is [auto-generated](https://huggingface.co/docs/dataset-viewer/en/croissant). You can then download and customize the generated file.

**Option 3: Write JSON-LD manually** (use the template above)

**Validation:** Use the [Croissant online editor/validator](https://docs.mlcommons.org/croissant/) or:
```bash
pip install mlcroissant
python -c "import mlcroissant as mlc; mlc.Dataset('metadata.json')"
```

### 4.5 Hosting Recommendation for NeurIPS

Given the NeurIPS 2025/2026 requirements:

**Primary: Hugging Face** (auto-generates Croissant, widely used, free, DOI support via Zenodo link)
- Upload CSV + Parquet + SQLite to a Hugging Face Dataset repo
- Create a dataset card (README.md in HF format)
- Croissant auto-generated, customizable

**Secondary: Zenodo** (for DOI and long-term archival)
- Upload the same files to Zenodo for a permanent DOI
- Link the DOI in the Croissant metadata and paper

---

## 5. Datasheet for Datasets (Gebru et al.)

### 5.1 Required Sections

The [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) framework (Gebru et al., CACM 2021) contains seven sections with specific questions. NeurIPS D&B track recommends (not strictly requires) including one. Below is each section with the key questions and guidance for NegBioDB.

### 5.2 NegBioDB Datasheet Template

---

#### Section 1: Motivation

**For what purpose was the dataset created?**
NegBioDB was created to address the systematic absence of experimentally confirmed negative (inactive) drug-target interactions in ML training and benchmark datasets. Existing DTI benchmarks (DAVIS, KIBA, BioSNAP, LIT-PCBA) either use random negative sampling (generating false negatives), contain only positive interactions, or have known data leakage issues. NegBioDB provides curated, experimentally confirmed negatives with confidence tiers and assay context.

**Who created the dataset and on behalf of which entity?**
[Authors], [Institution]. Created as an independent research project.

**Who funded the creation of the dataset?**
No external funding. All data sources are publicly available under open licenses. Infrastructure uses free-tier services only.

---

#### Section 2: Composition

**What do the instances that comprise the dataset represent?**
Each instance represents a compound-target pair where the compound has been experimentally tested and found to be inactive against the target protein. Compounds are small molecules represented by SMILES and InChIKey. Targets are proteins represented by UniProt accession and amino acid sequence.

**How many instances are there in total?**
[To be filled: target 10K-20K+ compound-target pairs in v1.0, backed by N individual assay measurements from M unique compounds and P unique targets.]

**Does the dataset contain all possible instances or is it a sample?**
It is a curated subset of all experimentally confirmed negatives. The full universe of confirmed negatives across all public databases exceeds millions of records. NegBioDB applies quality filters (confirmatory dose-response assays preferred, target annotation required, compound standardization) to select the highest-confidence negatives.

**What data does each instance consist of?**
- Compound SMILES (canonical, RDKit-standardized)
- Compound InChIKey
- Compound properties (MW, LogP, HBD, HBA, TPSA, QED)
- Target UniProt accession
- Target amino acid sequence
- Target classification (family, development level)
- Activity label (Y = 0 for inactive)
- Confidence tier (gold/silver/bronze/copper)
- Number of supporting assays and independent database sources
- Result type classification
- Publication year
- Pre-assigned train/val/test splits for 6 strategies

**Is there a label or target associated with each instance?**
Yes. Y = 0 (inactive/negative) for all instances. When combined with positive DTI data for benchmarking, positives receive Y = 1.

**Are there recommended data splits?**
Yes. Six pre-defined splits: random, cold-compound, cold-target, temporal, scaffold, and degree-distribution-balanced (DDB). Each split has train/val/test assignments (70/10/20).

**Are there any errors, sources of noise, or redundancies?**
- Activity thresholds vary by source (10 uM for NegBioDB vs. 100 uM for HCDT 2.0)
- Some compound-target pairs may show conflicting results across different assay formats (flagged in `has_conflicting_results`)
- PAINS and aggregator compounds are flagged but not removed (their inactivity data may be valid)
- Cross-database deduplication reduces redundancy but some near-duplicates may remain

**Is the dataset self-contained?**
Yes. The CSV/Parquet exports are self-contained. The SQLite database is self-contained. No external resources are required for ML use.

---

#### Section 3: Collection Process

**How was the data associated with each instance acquired?**
Data was computationally extracted from four public databases:
1. PubChem BioAssay: FTP bulk download of confirmatory inactive records (`bioactivities.tsv.gz`)
2. ChEMBL: SQL queries on the downloaded SQLite database (pChEMBL < 5, activity_comment = "Not Active")
3. BindingDB: Bulk TSV download filtered for Kd/Ki > 10 uM
4. DAVIS: Complete kinase interaction matrix via TDC Python library

**What mechanisms or procedures were used to collect the data?**
Automated extraction scripts with compound standardization (RDKit: salt removal, normalization, canonical SMILES, InChIKey generation) and target standardization (UniProt accession resolution). Cross-database deduplication using InChIKey connectivity layer (first 14 characters).

**Was any preprocessing/cleaning/labeling of the data done?**
Yes:
- RDKit standardization pipeline (salt removal, charge neutralization, tautomer normalization)
- PAINS and aggregator flagging
- Confidence tier assignment based on assay quality
- Cross-database deduplication
- Quality filter exclusion of flagged records (`data_validity_comment IS NULL` in ChEMBL)

**Is the software used to preprocess/clean/label the instances available?**
Yes. All scripts are in the GitHub repository: [URL]

---

#### Section 4: Preprocessing, Cleaning, Labeling

**Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?**
The raw data is available from the original public databases. NegBioDB records the `source_db` and `source_record_id` for every entry, enabling full provenance tracing.

**Is the software used to preprocess/clean/label the instances available?**
Yes. The full pipeline (extraction, standardization, deduplication, confidence assignment, split generation) is open-source.

---

#### Section 5: Uses

**Has the dataset been used for any tasks already?**
Yes, in the accompanying paper: DTI binary prediction (active/inactive classification), negative confidence prediction, and LLM evaluation tasks.

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**
- The dataset contains only negative (inactive) interactions. Users must combine with positive DTI data for binary classification tasks.
- Confidence tiers reflect data quality; users should consider filtering by tier for high-stakes applications.
- The 10 uM inactivity threshold is more stringent than some alternatives (e.g., HCDT 2.0 uses 100 uM). Compounds between 10-100 uM are excluded by default.

**Are there tasks for which the dataset should not be used?**
- Should not be used as a standalone dataset for predicting drug efficacy (these are in vitro inactivity measurements, not clinical outcomes)
- Should not be used without understanding that "inactive" is context-dependent (assay format, concentration tested, cell line)

---

#### Section 6: Distribution

**How will the dataset be distributed?**
- Hugging Face Datasets (primary, with Croissant metadata)
- Zenodo (DOI for citation and long-term archival)
- GitHub Releases (code + data)
- Python package (`pip install negbiodb`)

**When was the dataset first released?**
[2026-05-01, to coincide with NeurIPS submission]

**What license is the dataset distributed under?**
CC BY-SA 4.0 (compatible with ChEMBL CC BY-SA 3.0; PubChem is public domain; BindingDB is CC BY)

---

#### Section 7: Maintenance

**Who is supporting/hosting/maintaining the dataset?**
[Authors], [Institution]. Hosted on Hugging Face (primary) and Zenodo (archival).

**How can the owner/curator/manager of the dataset be contacted?**
GitHub Issues (preferred), email: [contact]

**Will the dataset be updated?**
Yes. Planned updates:
- Quarterly data refreshes (new ChEMBL releases, new PubChem depositions)
- New domain layers (Gene Function, Clinical Trial)
- New split strategies and benchmark tasks
- Schema versioning tracked in `schema_migrations` table

**If the dataset relates to people, are there applicable limits on the retention of the data?**
Not applicable. The dataset contains chemical and protein data only, no human subjects data.

---

### 5.3 Examples from Accepted NeurIPS D&B Papers

The following accepted NeurIPS D&B papers include datasheets or equivalent documentation:

- **WelQrate (NeurIPS 2024)**: Includes detailed dataset documentation covering curation pipeline, quality control measures, PAINS filtering, and benchmark evaluation. Source: [WelQrate Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/5f2f8305cd1c5be7e8319aea306388ce-Paper-Datasets_and_Benchmarks_Track.pdf)

- **MassSpecGym (NeurIPS 2024)**: Benchmark for molecule discovery and identification with comprehensive documentation. Source: [MassSpecGym Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/c6c31413d5c53b7d1c343c1498734b0f-Paper-Datasets_and_Benchmarks_Track.pdf)

- **TDC (NeurIPS 2021)**: Includes task categorization, data quality assessment, and benchmark design documentation as part of the supplementary materials. Source: [TDC supplementary](https://zitniklab.hms.harvard.edu/publications/papers/TDC-neurips21-supp.pdf)

- **Croissant (NeurIPS 2024)**: The Croissant format paper itself was accepted at NeurIPS 2024 D&B track. Source: [Croissant Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/9547b09b722f2948ff3ddb5d86002bc0-Paper-Datasets_and_Benchmarks_Track.pdf)

The common pattern among these papers: they embed dataset documentation within the paper itself (typically in the appendix), rather than as a separate document. The documentation covers motivation, composition, collection process, and intended use, even if not explicitly labeled as a "Datasheet for Datasets."

---

## 6. PubChem Streaming Processing Pattern (Expert Panel DE1)

PubChem `bioactivities.tsv.gz` is ~3 GB compressed / ~12 GB uncompressed (~301M rows). Loading the full file into memory will cause OOM on most machines. **Streaming processing is mandatory.**

### Option A: Pandas chunked reading (recommended for simplicity)

```python
import pandas as pd
import gzip

CONFIRMATORY_AIDS = set()  # Pre-loaded from bioassays.tsv.gz

def process_pubchem_streaming(filepath: str, output_path: str, chunksize: int = 100_000):
    """Stream-process PubChem bioactivities without loading full file.

    IMPORTANT: Filtered results are written to disk incrementally, NOT
    accumulated in memory. Even after filtering, ~61M inactive rows would
    cause OOM if collected in a list and concatenated.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    total_processed = 0
    total_kept = 0
    writer = None

    with gzip.open(filepath, 'rt') as f:
        reader = pd.read_csv(f, sep='\t', chunksize=chunksize,
                             usecols=['AID', 'SID', 'CID', 'Activity_Outcome',
                                      'Activity_Value', 'Activity_Name'],
                             dtype={'AID': 'int64', 'SID': 'int64',
                                    'CID': 'float64',  # nullable
                                    'Activity_Outcome': 'str'})

        for i, chunk in enumerate(reader):
            # Filter: confirmatory assays + inactive outcome
            mask = (
                chunk['AID'].isin(CONFIRMATORY_AIDS) &
                (chunk['Activity_Outcome'] == 'Inactive')
            )
            filtered = chunk[mask]
            total_processed += len(chunk)
            total_kept += len(filtered)

            # Write each filtered chunk directly to Parquet (no memory accumulation)
            if len(filtered) > 0:
                table = pa.Table.from_pandas(filtered, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)

            if (i + 1) % 100 == 0:
                print(f"Processed {total_processed:,} rows, kept {total_kept:,}")

    if writer is not None:
        writer.close()
    print(f"Done. {total_kept:,} rows written to {output_path}")
```

### Option B: Polars lazy evaluation (faster, lower memory)

```python
import polars as pl

def process_pubchem_polars(filepath: str, confirmatory_aids: list, output_path: str):
    """Polars lazy scan — processes without loading full dataset."""
    df = (
        pl.scan_csv(filepath, separator='\t', has_header=True)
        .filter(
            pl.col('AID').is_in(confirmatory_aids) &
            (pl.col('Activity_Outcome') == 'Inactive')
        )
        .select(['AID', 'SID', 'CID', 'Activity_Outcome', 'Activity_Value'])
        .collect(streaming=True)  # Streaming execution
    )
    df.write_parquet(output_path)
    return df
```

### SID→CID Mapping: SQLite Temp Table Approach

`Sid2CidSMILES.gz` (122 MB) contains SID-to-CID-to-SMILES mappings. For efficient joins:

```python
import sqlite3

def load_sid_mapping(mapping_path: str, db_path: str):
    """Load SID→CID mapping into SQLite temp table for indexed joins."""
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS sid_cid_map (sid INTEGER PRIMARY KEY, cid INTEGER, smiles TEXT)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sid_cid ON sid_cid_map(cid)")

    # Stream insert in batches
    reader = pd.read_csv(mapping_path, sep='\t', chunksize=500_000,
                         names=['sid', 'cid', 'smiles'], compression='gzip')
    for chunk in reader:
        chunk.to_sql('sid_cid_map', conn, if_exists='append', index=False)

    conn.commit()
    return conn
```

---

## 7. Implementation Checklist for NeurIPS Sprint

### Week 1-2 Priority Actions (Schema + Export)

- [ ] Create `migrations/001_initial_schema.sql` with the DDL from Section 1.2
- [ ] Implement `scripts/create_database.py` that runs migrations and creates empty DB
- [ ] Implement compound standardization (Section 3.2 of `05_technical_deep_dive.md`)
- [ ] Implement target standardization
- [ ] Implement `scripts/09_export_benchmark.py` with CSV + Parquet export (Section 2.3)
- [ ] Create `croissant/metadata.json` from template in Section 4.3
- [ ] Draft datasheet sections 1-3 (Motivation, Composition, Collection)

### Week 3-5 Priority Actions (Benchmark Layer)

- [ ] Implement split generation scripts (random, cold_compound, cold_target minimum)
- [ ] Populate `split_definitions` and `split_assignments` tables
- [ ] Populate `compound_target_pairs` aggregation table
- [ ] Validate Croissant metadata with mlcroissant
- [ ] Complete datasheet sections 4-7

### Pre-Submission Checklist

- [ ] Croissant JSON-LD validates without errors
- [ ] CSV/Parquet exports load correctly in pandas
- [ ] SQLite database passes integrity checks (`PRAGMA integrity_check`)
- [ ] All split assignments sum to expected totals
- [ ] Datasheet for Datasets complete (all 7 sections)
- [ ] Dataset uploaded to Hugging Face
- [ ] DOI obtained from Zenodo
- [ ] SHA256 checksums updated in Croissant metadata
- [ ] **Replace all metadata placeholders**: Author name/institution in Croissant `creator` field, `citeAs` BibTeX, Datasheet `[Authors]`/`[Institution]`, `PLACEHOLDER_SHA256` values
- [ ] Implement `merge_positive_negative()` for M1 task (see Section 2.3)

---

## Sources

- [SQLite for chemical search](https://macinchem.org/2023/04/12/using-sqlite-for-exact-search/)
- [ChemicaLite (SQLite cheminformatics extension)](https://chemicalite.readthedocs.io/en/latest/tutorial_1st.html)
- [ChEMBL on SQLite](http://chembl.blogspot.com/2016/03/chembl-db-on-sqlite-is-that-even.html)
- [TDC](https://tdcommons.ai/) and [TDC GitHub](https://github.com/mims-harvard/TDC)
- [TDC DTI Tasks](https://tdcommons.ai/multi_pred_tasks/dti/)
- [MoleculeNet](https://moleculenet.org/datasets-1) and [MoleculeNet Paper](https://arxiv.org/abs/1703.00564)
- [WelQrate (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5f2f8305cd1c5be7e8319aea306388ce-Paper-Datasets_and_Benchmarks_Track.pdf)
- [Croissant specification](https://docs.mlcommons.org/croissant/docs/croissant-spec.html)
- [Croissant GitHub (mlcommons/croissant)](https://github.com/mlcommons/croissant)
- [Croissant Paper (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/9547b09b722f2948ff3ddb5d86002bc0-Paper-Datasets_and_Benchmarks_Track.pdf)
- [mlcroissant Python library](https://github.com/mlcommons/croissant/tree/main/python/mlcroissant)
- [Hugging Face Croissant integration](https://huggingface.co/docs/dataset-viewer/en/croissant)
- [NeurIPS 2025 D&B Call for Papers](https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks)
- [NeurIPS 2025 D&B blog post (raising the bar)](https://blog.neurips.cc/2025/03/10/neurips-datasets-benchmarks-raising-the-bar-for-dataset-submissions/)
- [NeurIPS 2025 Data Hosting Guidelines](https://neurips.cc/Conferences/2025/DataHostingGuidelines)
- [Datasheets for Datasets (Gebru et al.)](https://arxiv.org/abs/1803.09010)
- [Datasheets template (GitHub)](https://github.com/AudreyBeard/Datasheets-for-Datasets-Template)
- [Alembic SQLite batch migrations](https://alembic.sqlalchemy.org/en/latest/batch.html)
- [Parquet vs CSV vs HDF5 comparison](https://www.hopsworks.ai/post/guide-to-file-formats-for-machine-learning)
- [DrugBank SQL schema](https://dev.drugbank.com/guides/sql)
- [BioSNAP drug-target network](https://snap.stanford.edu/biodata/datasets/10015/10015-ChG-TargetDecagon.html)
