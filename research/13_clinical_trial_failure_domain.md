# Research Document 13: Clinical Trial Failure Domain — Deep Research & Feasibility Plan

> Created: 2026-03-17 | Status: Research Complete, Reviewed (3-agent audit, 30 issues found & fixed)

---

## Executive Summary

This document presents a comprehensive research-backed plan for extending NegBioDB to a **Clinical Trial Failure** domain. The research identifies a clear gap: **no existing free resource provides a curated, structured database of clinical trial failures** with standardized failure taxonomy, quantitative outcome data, drug-target linkage, ML-ready format, and LLM evaluation tasks — all combined in one resource.

**Key findings:**
- ~500K registered trials on ClinicalTrials.gov; ~25K terminated, ~200K+ completed-with-negative-results
- **No existing free resource** combines all of: failure taxonomy + quantitative outcomes + molecular linkage + ML/LLM benchmarks (closest partial matches: CTO 2/7, TrialBench 2/7)
- **18 existing resources** analyzed; best partial coverage scores only 2/7 on our criteria
- **All key data sources are free** with permissive licenses (public domain, CC BY, Apache 2.0)
- NegBioDB architecture (Common Layer) is ~70% reusable
- **Realistic timeline: 6-8 weeks** (drug-name entity resolution is the critical bottleneck)

---

## Table of Contents

1. [Competitive Landscape](#1-competitive-landscape)
2. [Gap Analysis](#2-gap-analysis)
3. [Data Sources & Access](#3-data-sources--access)
4. [Proposed Architecture](#4-proposed-architecture)
5. [Benchmark Design](#5-benchmark-design)
6. [Literature Foundation](#6-literature-foundation)
7. [Technical Feasibility](#7-technical-feasibility)
8. [Execution Plan](#8-execution-plan)
9. [Risk Assessment](#9-risk-assessment)
10. [References](#10-references)

---

## 1. Competitive Landscape

### 1.1 Feature Matrix — All Existing Resources

| Resource | Size | (a) Failure Taxonomy | (b) Quantitative Outcomes | (c) Drug-Target Linkage | (d) ML-ready | (e) LLM Tasks | (f) Confidence Tiers | (g) Cross-registry |
|----------|------|---------------------|--------------------------|------------------------|-------------|--------------|---------------------|--------------------|
| **AACT** (Duke/CTTI) | 500K+ trials | Free-text only | Raw results tables | Minimal | No | No | No | No (CT.gov only) |
| **CTO** (Nature Health 2026) | 125K trials | Binary only | No | No | Yes | No | No | No |
| **TrialBench** (Sci Data 2025) | 420K+ underlying trials, 23 datasets | 4 categories | No | SMILES only | **Yes** | No | No | No |
| **Open Targets Stop Reasons** | 3,747 curated | **17 categories** | No | In paper, not dataset | No | No | No | No |
| **ClinicalRisk** (CIKM 2023) | 12,717 trials | 4 categories | No | No | Yes | No | No | No |
| **TrialPanorama** (arXiv 2025) | 1.66M trials | No (binary completion) | Some | DrugBank partial | Yes | Some | No | **Yes (15 reg.)** |
| **Shi & Du 2024** (Sci Data) | 8,665 trials | No | **Yes (11K p-values)** | No | No | No | No | No |
| **CT-ADE** (Sci Data 2025) | 168K drug-ADE pairs | Safety only | No | Drug-ADE only | Yes | Yes | No | No |
| **CliniFact** (Sci Data 2025) | 1,970 claims | No | No | No | Yes | **Yes** | No | No |
| **CDEK** (Database 2019) | 127K trials, 22K APIs (Active Pharmaceutical Ingredients) | No | No | No | No | No | No | No |
| **ClinSR.org** (Nat Comms 2025) | 20K pipelines | No (aggregate only) | No | No | No | No | No | No |
| **openFDA CRLs** | ~300 letters | Regulatory only | No | No | No | No | No | No |
| **TOP/HINT** (Patterns 2022) | 17,538 trials | Binary only | No | SMILES+fingerprints | Yes | No | No | No |
| **PyTrial** (NeurIPS workshop) | Various | Binary only | No | Some | Yes | No | No | No |
| **CTKG** (Sci Reports 2022) | 1.5M nodes KG | No | No | No | No | No | No | No |
| **OpenTrials** (Goldacre) | — | — | — | — | — | — | — | — |
| Citeline Trialtrove | 400K+ trials | Expert-curated | Yes | Yes | Proprietary | No | — | Yes |
| Certara CODEX | 8K+ studies | Expert-curated | Yes | Some | Proprietary | No | — | Some |

**Score: No free resource exceeds 2/7 criteria (partial coverage — "Some," "partial" — not counted as meeting full criterion). Commercial resources (Citeline, Certara) score higher but are prohibitively expensive and non-redistributable.**

### 1.2 Closest Competitors — Detailed Assessment

#### CTO Benchmark (Nature Health, 2026) — **Highest risk**
- 125K trials, binary outcome labels via weak supervision (GPT-3.5 + news sentiment + stock prices)
- MIT license, HuggingFace available
- **Gap:** No failure reasons, no quantitative outcomes, no drug-target linkage, no confidence tiers
- **Risk:** If they add failure taxonomy + molecular features in v2, they become a direct competitor
- **Mitigation:** Our focus on *why* trials fail (not just binary outcome) is fundamentally different
- **Opportunity:** CTO's 125K binary outcome labels (MIT license) can serve as input to our pipeline — use their labels as a pre-filter for identifying negative-outcome trials, then enrich with our failure taxonomy and quantitative outcomes

#### TrialBench (Nature Scientific Data, 2025) — **Second highest risk**
- 23 datasets including failure reason identification (41K trials, 4 categories)
- Multi-modal: SMILES, MeSH, ICD-10, eligibility text
- **Gap:** Only 4 coarse failure categories, no quantitative outcomes, no target linkage
- **Risk:** Could deepen failure taxonomy in future
- **Mitigation:** Our hierarchical taxonomy (30+ categories) with quantitative evidence is far richer

#### Open Targets Stop Reasons (Nature Genetics, 2024)
- Best existing taxonomy (17 categories), but only 3,747 curated examples
- Apache 2.0 — we can incorporate and extend
- **Gap:** No molecular features, no ML format, no LLM tasks
- **Mitigation:** We build on their taxonomy and extend it with molecular linkage

### 1.3 DTI-Level Competitors (for NegBioDB paper context)

| Resource | Negative DTIs | Key Difference |
|----------|--------------|---------------|
| **NegBioDB** (ours) | **30.5M results, 25M pairs** | Multi-source, curated, ML+LLM benchmarks |
| HCDT 2.0 (Sci Data 2025) | 38,653 | Supplement to positive DB; CC BY-NC-ND |
| InertDB (J Cheminf 2025) | 3,205 compounds | Inert *compounds* (not pairs); decoy generation |
| CARA (Comms Chem 2024) | — | Activity benchmark; no negative focus |
| DUD-E (2012) | ~1.4M decoys | Computationally generated, known biases |

**NegBioDB has ~646× more negative compound-target pairs (25M pairs vs 38,653) than the next largest free resource (HCDT 2.0).**

---

## 2. Gap Analysis

### 2.1 The Missing Resource

No existing resource provides:

1. **Hierarchical failure reason taxonomy** — going beyond the 4 categories (TrialBench) or 17 categories (Open Targets) to include subcategories like:
   - Efficacy → insufficient vs placebo → primary endpoint missed → by specific margin
   - Safety → organ-specific toxicity → cardiac/hepatic/neurological
   - PK/PD → poor bioavailability → metabolic instability

2. **Quantitative outcome data linked to failure** — p-values, effect sizes, confidence intervals structured and connected to failure categories (Shi & Du 2024 has p-values but no failure classification)

3. **Drug-target molecular linkage** — connecting failed trial drugs to their molecular targets via UniProt/gene IDs with SMILES structures (nobody does this for failed trials)

4. **Preclinical-to-clinical bridge** — linking NegBioDB DTI negatives to clinical trial failures for the same drug-target pairs (completely novel)

5. **Dual ML + LLM benchmark** specifically for understanding *why* things fail

6. **Confidence-tiered evidence** with cross-registry deduplication

### 2.2 The Opportunity

| Dimension | Existing Best | Our Target | Improvement |
|-----------|-------------|------------|-------------|
| Failure taxonomy | 17 categories (Open Targets) | 30+ hierarchical categories | 2× richer, hierarchical |
| Curated failure records | 3,747 (Open Targets) | 20K-40K | 5-11× more |
| Quantitative outcomes | 11K results from 8.7K trials (Shi & Du) | 15K+ with structured p-values | ~1.5× more, linked to failure taxonomy |
| Drug-target linkage | 0 (nobody) | All drug trials mapped to targets | **First of its kind** |
| ML benchmark tasks | 1 task, 4 classes (TrialBench) | 3 tasks, 8 top-level + 37 hierarchical classes | 3× tasks, 2-8× classes |
| LLM evaluation tasks | 0 (nobody for failure reasoning) | 4 tasks (v1) | **First of its kind** |

### 2.3 Publication Bias Justification

From the literature:
- **Only 53% of all trials are published** (Showell et al., Cochrane 2024)
- Positive trials have **2.69× higher odds** of being published (OR 2.69, 95% CI 2.02-3.60)
- **~23% of Nordic trials** (2016-2019) never reported any results (Nilsonne et al., 2025)
- Terminated trials have a **35% results reporting rate** vs 78% for completed trials (Yerunkar et al., 2025; preprint)
- **~86% of drugs entering Phase I fail** to gain approval (LOA ~13.8%; Wong et al., 2019)
- **40-50% of failures are due to lack of efficacy** (Sun et al., 2022) — directly linkable to molecular target engagement

---

## 3. Data Sources & Access

### 3.1 Primary Sources (All Free, Permissive Licenses)

| Source | Content | Volume | License | Access Method |
|--------|---------|--------|---------|--------------|
| **AACT** (Duke/CTTI) | All ClinicalTrials.gov data in PostgreSQL | 500K+ trials | Public domain (US govt data) / MIT (code) | PostgreSQL direct / pipe files |
| **ClinicalTrials.gov API v2** | Full trial records in JSON | 500K+ trials | Public domain | REST API, 50 req/min, no auth |
| **Open Targets Stop Reasons** | 17-category failure taxonomy | 3,747 curated† + ~28.5K NLP-classified‡ | Apache 2.0 | HuggingFace Parquet |
| **openFDA CRLs** | FDA rejection reasons | ~300 letters | CC0 (public domain) | JSON API |
| **Shi & Du 2024** | Structured p-values/effect sizes | 11,216 efficacy results | CC BY 4.0 | GitHub CSV/JSON |
| **CT-ADE** | Drug-adverse event pairs | 168,984 pairs | MIT | GitHub/Figshare |
| **DrugBank** (open data) | Drug structures, targets, indications | 15K+ drugs | CC BY-NC 4.0 | XML download |
| **ChEMBL** | Drug-target mapping for approved drugs | 2.5M+ activities | CC BY-SA 3.0 | SQLite |
| **UniProt** | Target protein sequences | 250M+ entries | CC BY 4.0 | REST API / bulk |
| **WHO ICD-10/11** | Disease classification | Full taxonomy | WHO terms | API / download |

*† Open Targets curated count (3,747) comes from Razuvayevskaya et al. 2024 supplementary; methodology mixes manual expert annotation with semi-automated classification. ‡ NLP-classified count is approximate (28,561 in paper; HuggingFace dataset may differ slightly at ~28,842 due to versioning).*

### 3.2 License Compatibility

| Source License | Compatible with CC BY-SA 4.0? | Notes |
|----------------|-------------------------------|-------|
| Public domain (CT.gov, openFDA) | Yes | No restrictions |
| Public domain / MIT (AACT) | Yes | US govt data; no restrictions on data |
| Apache 2.0 (Open Targets) | Yes | Very permissive |
| CC BY 4.0 (Shi & Du, UniProt) | Yes | Attribution required |
| MIT (CT-ADE) | Yes | Very permissive |
| CC BY-SA 3.0 (ChEMBL) | Yes | Forces SA (already using SA) |
| CC BY-NC 4.0 (DrugBank open) | **NO — incompatible** | CC BY-NC cannot be included in CC BY-SA dataset; must be separate optional file |
| WHO terms (ICD-10/11) | **Conditional** | WHO allows free use of ICD codes for research but restricts redistribution of full classification; use codes only (not descriptions) or link via MeSH instead |
| CC BY-NC-SA 4.0 (CDEK) | **NO — incompatible** | Same NC restriction as DrugBank; CDEK drug mappings must be in separate optional file |

**Conclusion:** Core pipeline is fully compatible with CC BY-SA 4.0. **DrugBank-derived features must be in a separate optional enrichment file** with its own CC BY-NC 4.0 license — cannot be redistributed in the main CC BY-SA 4.0 release. Use ChEMBL (CC BY-SA 3.0) as primary source for drug-target linkage instead.

### 3.3 Data Volume Estimates

| Category | Estimated Count | Source |
|----------|----------------|--------|
| Total registered trials | ~500,000 | ClinicalTrials.gov |
| Terminated trials | ~25,000 | AACT `overall_status = 'Terminated'` |
| Withdrawn trials | ~5,000 | AACT |
| Completed with results posted | ~60,000 | AACT `has_results = true` |
| Completed drug trials (Phase II/III) | ~80,000 | AACT filtered |
| **Estimated curated failure records** | **20,000-40,000** | After filtering + NLP classification |
| With quantitative outcomes (p-values in AACT) | ~5,000-15,000 | Trials with `outcome_analyses` entries for primary endpoint |
| With drug-target mappable | ~15,000-30,000 | Drug trials with identifiable molecular targets |

### 3.4 Access Patterns

```
Pipeline Overview:

1. AACT PostgreSQL → Bulk download of studies, interventions, outcomes, reported_events
   (Daily snapshots: https://aact.ctti-clinicaltrials.org/pipe_files)

2. ClinicalTrials.gov API v2 → Supplement for results not in AACT
   (https://clinicaltrials.gov/api/v2/studies?filter.overallStatus=TERMINATED,COMPLETED)

3. Open Targets NLP model → Classify unstructured why_stopped text
   (https://github.com/LesyaR/stopReasons)

4. DrugBank + ChEMBL → Map intervention names to molecular structures/targets
   (Drug name → DrugBank ID → SMILES + UniProt targets)

5. openFDA CRLs → Regulatory rejection reasons for late-stage failures
   (https://open.fda.gov/apis/transparency/completeresponseletters/)

6. Shi & Du 2024 → Structured p-values/effect sizes for subset
   (https://github.com/xuanyshi/Finer-Grained-Clinical-Trial-Results)
```

---

## 4. Proposed Architecture

### 4.1 Schema Design

#### Reused from NegBioDB Common Layer (~70% reuse)
- `db.py` connection/migration infrastructure → 100% reuse
- `schema_migrations`, `dataset_versions` tables → 100% reuse
- Confidence tier system (gold/silver/bronze/copper) → 100% reuse
- Provenance tracking columns → 100% reuse
- Split framework (split_definitions + split_assignments) → 95% reuse
- Export pipeline pattern (chunked Parquet + CSV) → 90% reuse

#### New Domain Tables

```sql
-- Interventions (replaces compounds)
CREATE TABLE interventions (
    intervention_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_type   TEXT NOT NULL CHECK (intervention_type IN (
        'drug', 'biologic', 'device', 'procedure', 'behavioral',
        'dietary', 'genetic', 'radiation', 'combination', 'other')),
    intervention_name   TEXT NOT NULL,
    canonical_name      TEXT,           -- WHO INN standardized
    drugbank_id         TEXT,
    pubchem_cid         INTEGER,
    chembl_id           TEXT,
    mesh_id             TEXT,
    atc_code            TEXT,
    mechanism_of_action TEXT,
    canonical_smiles    TEXT,           -- if small molecule
    canonical_sequence  TEXT,           -- if biologic (antibody, peptide, etc.)
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Conditions (replaces targets)
CREATE TABLE conditions (
    condition_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_name      TEXT NOT NULL,
    canonical_name      TEXT,
    mesh_id             TEXT,
    icd10_code          TEXT,
    icd11_code          TEXT,
    do_id               TEXT,           -- Disease Ontology
    therapeutic_area    TEXT,
    condition_class     TEXT,           -- rare, common, orphan
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Molecular Targets (bridge to DTI domain — NEW)
CREATE TABLE intervention_targets (
    intervention_id     INTEGER NOT NULL REFERENCES interventions(intervention_id),
    uniprot_accession   TEXT NOT NULL,
    gene_symbol         TEXT,
    target_role         TEXT CHECK (target_role IN (
        'primary', 'secondary', 'off_target')),
    action_type         TEXT,           -- inhibitor, agonist, antagonist, etc.
    source              TEXT NOT NULL,  -- drugbank, chembl, literature
    PRIMARY KEY (intervention_id, uniprot_accession)
);

-- Clinical Trials (replaces assays)
CREATE TABLE clinical_trials (
    trial_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_db           TEXT NOT NULL CHECK (source_db IN (
        'clinicaltrials_gov', 'eu_ctr', 'who_ictrp', 'literature')),
    source_trial_id     TEXT NOT NULL,  -- NCT number, EudraCT number, etc.
    overall_status      TEXT NOT NULL,
    trial_phase         TEXT CHECK (trial_phase IN (
        'early_phase_1', 'phase_1', 'phase_1_2', 'phase_2',
        'phase_2_3', 'phase_3', 'phase_4', 'not_applicable')),
    study_type          TEXT,           -- interventional, observational
    study_design        TEXT,           -- RCT, single-arm, crossover
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
    primary_completion_date TEXT,       -- anchor for temporal splits
    completion_date     TEXT,
    results_posted_date TEXT,
    why_stopped         TEXT,           -- original free text
    has_results         INTEGER DEFAULT 0,
    created_at          TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(source_db, source_trial_id)
);

-- Junction: trials ↔ interventions (many-to-many)
CREATE TABLE trial_interventions (
    trial_id        INTEGER NOT NULL REFERENCES clinical_trials(trial_id),
    intervention_id INTEGER NOT NULL REFERENCES interventions(intervention_id),
    arm_group       TEXT,
    arm_role        TEXT CHECK (arm_role IN (
        'experimental', 'active_comparator', 'placebo_comparator', 'no_intervention')),
    dose_regimen    TEXT,              -- e.g., '200mg BID'
    PRIMARY KEY (trial_id, intervention_id)
);

-- Junction: trials ↔ conditions (many-to-many)
CREATE TABLE trial_conditions (
    trial_id     INTEGER NOT NULL REFERENCES clinical_trials(trial_id),
    condition_id INTEGER NOT NULL REFERENCES conditions(condition_id),
    PRIMARY KEY (trial_id, condition_id)
);

-- Junction: trials ↔ publications
CREATE TABLE trial_publications (
    trial_id   INTEGER NOT NULL REFERENCES clinical_trials(trial_id),
    pubmed_id  INTEGER NOT NULL,
    pub_type   TEXT,  -- primary, secondary, post_hoc
    PRIMARY KEY (trial_id, pubmed_id)
);

-- Combination therapy decomposition
CREATE TABLE combination_components (
    combination_id  INTEGER NOT NULL REFERENCES interventions(intervention_id),
    component_id    INTEGER NOT NULL REFERENCES interventions(intervention_id),
    role            TEXT CHECK (role IN ('experimental', 'backbone', 'comparator')),
    PRIMARY KEY (combination_id, component_id)
);

-- Trial Failure Results (replaces negative_results)
CREATE TABLE trial_failure_results (
    result_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_id     INTEGER NOT NULL REFERENCES interventions(intervention_id),
    condition_id        INTEGER NOT NULL REFERENCES conditions(condition_id),
    trial_id            INTEGER REFERENCES clinical_trials(trial_id),

    -- Hierarchical failure classification
    failure_category    TEXT NOT NULL CHECK (failure_category IN (
        'efficacy', 'safety', 'pharmacokinetic', 'enrollment',
        'strategic', 'regulatory', 'design', 'other')),
    failure_subcategory TEXT,           -- e.g., 'primary_endpoint_not_met'
    failure_detail      TEXT,           -- free text detail

    -- Confidence tier (reused from DTI)
    confidence_tier     TEXT NOT NULL CHECK (confidence_tier IN (
        'gold', 'silver', 'bronze', 'copper')),

    -- Arm-level context (multi-arm trials)
    arm_description     TEXT,           -- e.g., '200mg BID' or 'low dose'
    arm_type            TEXT CHECK (arm_type IN (
        'experimental', 'active_comparator', 'placebo_comparator', 'overall')),

    -- Quantitative outcome data
    primary_endpoint_met    INTEGER,    -- 0=not met, 1=met, NULL=unknown
    p_value_primary         REAL,
    effect_size             REAL,
    effect_size_type        TEXT,       -- OR, HR, RR, mean_diff, SMD
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

    -- Provenance (reused pattern)
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

-- Aggregation table (replaces compound_target_pairs)
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
    has_any_approval    INTEGER DEFAULT 0,  -- approved for *any* indication?
    intervention_degree INTEGER,            -- # conditions tested
    condition_degree    INTEGER,            -- # interventions tested
    UNIQUE(intervention_id, condition_id)
);

-- Trial Failure Context (replaces dti_context)
CREATE TABLE trial_failure_context (
    result_id           INTEGER PRIMARY KEY REFERENCES trial_failure_results(result_id),
    patient_population  TEXT,
    biomarker_stratified INTEGER DEFAULT 0,
    companion_diagnostic TEXT,
    prior_treatment_lines INTEGER,
    comparator_drug     TEXT,
    geographic_regions  TEXT,
    regulatory_pathway  TEXT,           -- accelerated, breakthrough, standard
    genetic_evidence    INTEGER DEFAULT 0,  -- target has GWAS support?
    class_effect_known  INTEGER DEFAULT 0,  -- related drugs known to fail?
    has_negbiodb_dti_data INTEGER DEFAULT 0  -- preclinical negative data exists?
);
```

### 4.2 Failure Taxonomy (Hierarchical, 30+ Categories)

Extending Open Targets' 17 categories with subcategorization:

```
EFFICACY (lack of therapeutic effect)
├── primary_endpoint_not_met      — Phase II/III: p > 0.05 on primary
├── insufficient_vs_placebo       — Numerically close but not significant
├── insufficient_vs_active        — Inferior to active comparator
├── futility_interim              — Interim analysis: unlikely to succeed
├── no_dose_response              — No clear dose-response relationship
└── biomarker_not_moved           — Target biomarker unchanged

SAFETY (unacceptable adverse effects)
├── hepatotoxicity                — Liver injury signal
├── cardiotoxicity                — QT prolongation, cardiac events
├── neurotoxicity                 — CNS adverse effects
├── immunotoxicity                — Immune-related AEs (esp. I/O)
├── general_toxicity              — Overall AE burden too high
├── mortality_signal              — Excess deaths in treatment arm
└── dsmb_stopped                  — DSMB recommended stopping

PHARMACOKINETIC (PK/PD failure)
├── poor_bioavailability          — Low oral absorption
├── rapid_metabolism              — Short half-life / clearance
├── poor_target_engagement        — PK OK but no PD effect
├── drug_drug_interaction         — Clinically relevant DDIs
└── formulation_failure           — Drug product issues

ENROLLMENT (recruitment failure)
├── insufficient_accrual          — Could not recruit enough patients
├── slow_enrollment               — Enrollment too slow to continue
└── competing_trials              — Lost patients to competitor trials

STRATEGIC (business/operational)
├── business_decision             — Portfolio prioritization, M&A
├── funding_exhausted             — Ran out of money
├── sponsor_withdrawn             — Sponsor pulled out
└── covid_disruption              — COVID-19 related

REGULATORY (authority decisions)
├── fda_crl                       — Complete Response Letter
├── ema_refusal                   — EMA negative opinion
├── clinical_hold                 — FDA placed on hold
└── post_market_withdrawal        — Approved then withdrawn

DESIGN (protocol/methodology flaw)
├── endpoint_selection            — Wrong primary endpoint
├── patient_selection             — Wrong patient population
├── underpowered                  — Sample size too small
├── protocol_amendment_burden     — Too many changes
└── data_quality                  — Integrity issues

OTHER (uncategorized/ambiguous)
├── insufficient_information      — Cannot determine failure reason
├── multiple_unresolvable         — Multiple equally likely reasons
└── not_applicable                — Non-failure termination (e.g., already approved)
```

### 4.3 Confidence Tier Definitions (Clinical Trial Domain)

| Tier | Definition | Example |
|------|-----------|---------|
| **Gold** | Phase III failure with full results posted (p-values, CIs, AE data) + peer-reviewed publication | AACT results + PubMed-linked paper |
| **Silver** | Phase II/III failure with results posted on ClinicalTrials.gov (quantitative) | AACT results, no publication |
| **Bronze** | Terminated trial with structured `why_stopped` classified by NLP model | Open Targets NLP classification |
| **Copper** | Press release, SEC filing, or news report only; or LLM-extracted | openFDA CRL or news-derived |

---

## 5. Benchmark Design

### 5.1 ML Track (CT-M Tasks)

| Task | Input | Output | Primary Metric | Data Source |
|------|-------|--------|----------------|-------------|
| **CT-M1: Trial Outcome Prediction** | (drug SMILES, disease ICD-10, trial design features) | Success / Failure | PR-AUC, F1 | Merged CTO + AACT |
| **CT-M2: Failure Reason Classification** | (drug features, disease, trial metadata) | 8 failure categories | Macro-F1, MCC | Our curated taxonomy |
| **CT-M3: Phase Transition + Failure Mode** | (drug features, Phase N results) | (a) Advance / Fail Phase N+1, (b) failure mode if failed | (a) AUROC, PR-AUC; (b) Macro-F1 | Phase-linked AACT data |

*CT-M2 note: efficacy failures dominate (40-50%). Provide class-weighted and oversampled variants to mitigate class imbalance. Report both macro-F1 (penalizes rare class errors) and per-class F1.*

*CT-M3 is multi-task by design: unlike HINT/TOP which only predict binary pass/fail, our task jointly predicts transition outcome AND failure mode. This provides both a direct comparison point with HINT/TOP baselines (subtask a) and a novel contribution (subtask b).*

**Split Strategies:**
| Strategy | Description | Purpose |
|----------|-------------|---------|
| `random` | Stratified 70/10/20 | Baseline |
| `cold_intervention` | Test on unseen drugs | Generalization to new drugs |
| `cold_condition` | Test on unseen diseases | Generalization to new indications |
| `temporal` | Train < 2018, val 2018-2020, test > 2020 | Temporal generalization |
| `cold_therapeutic_area` | Hold out entire TAs (e.g., oncology) | Domain transfer |
| `phase_stratified` | Balanced by phase distribution | Phase-fair evaluation |

### 5.2 LLM Track (CT-L Tasks)

| Task | Input | Output | Eval Method | Size Target |
|------|-------|--------|-------------|-------------|
| **CT-L1: Failure Reason MCQ** | Trial description + options | Failure category (MCQ) | Automated (accuracy, MCC) | 2,000 |
| **CT-L2: Trial Report Extraction** | Publication abstract | Structured JSON (drug, condition, outcome, p-value, failure reason) | Automated (entity F1) | 200 |
| **CT-L3: Failure Reasoning** | Trial failure + context | Scientific explanation of why it failed | LLM-as-Judge + human | 50 pilot |
| **CT-L4: Phase Transition Judgment** | Phase II results | Will Phase III succeed? + evidence | Automated + spot-check | 500 |
| ~~CT-L5: Preclinical-Clinical Bridge~~ | ~~Drug + DTI negative data from NegBioDB~~ | ~~Predict clinical failure risk~~ | ~~LLM-as-Judge~~ | ~~100 pilot~~ |

**CT-L5 deferred to future work.** The preclinical-clinical bridge is scientifically interesting but data-thin: the overlap between NegBioDB DTI compounds and clinically tested drugs is likely in the low hundreds. This task requires both databases to be mature and the actual compound overlap to be empirically measured. Better framed as a target-centric analysis (do targets with many preclinical negatives also have higher clinical failure rates?) in a future integrative paper.

**Core LLM tasks for v1: CT-L1 (MCQ) + CT-L3 (Reasoning).** CT-L2 and CT-L4 are secondary.

### 5.3 Comparison with Existing Benchmarks

| Feature | TOP/HINT | CTO | TrialBench | Shi & Du 2024 | ClinicalRisk | **Ours** |
|---------|---------|-----|-----------|-------------|-------------|---------|
| Trials | 17.5K | 125K | 420K+ | 8.7K | 12.7K | 20-50K curated (planned) |
| Outcome labels | Binary | Binary (weak) | 4 categories | No | 4 categories | **30+ hierarchical** (planned) |
| Quantitative evidence | No | No | No | **Yes (11K p-values)** | No | **Yes (p-values, CIs)** (planned) |
| Drug-target linkage | SMILES only | No | SMILES+MeSH | No | No | **SMILES + UniProt** (planned) |
| LLM tasks | No | No | No | No | No | **4 tasks** (planned) |
| Confidence tiers | No | No | No | No | No | **4 tiers** (planned) |
| Cross-registry | No | No | No | No | No | **(future work)** |
| Failure focus | Not primary | Not primary | 1 of 8 tasks | No | Yes | **Core focus** |

---

## 6. Literature Foundation

### 6.1 Clinical Trial Failure Rates (Key Statistics)

| Statistic | Value | Source |
|-----------|-------|--------|
| Overall Phase I-to-approval LOA | ~13.8% (median); 6.7% (Citeline) | Wong et al. 2019; Citeline 2014-2023 |
| Phase II success rate | ~28% | Citeline/Norstella 2014-2023 analysis |
| Oncology LOA | 3.4% | Wong et al. 2019 |
| AD trial failure rate | 99.6% | Cummings et al. 2014 |
| Primary cause of failure | 40-50% efficacy | Acta Pharma Sinica B 2022 |
| Secondary cause | 30% safety | Acta Pharma Sinica B 2022 |
| Genetic evidence success multiplier | 2.6× | Minikel et al., Nature 2024 |
| Biomarker usage success multiplier | ~2× | Wong et al. 2019 |
| Trials ever published | 53% | Showell et al., Cochrane 2024 |
| Publication bias (positive vs negative) | OR 2.69 (95% CI 2.02-3.60) | Showell et al. 2024 |
| Terminated trials reporting rate vs completed | 35% vs 78% | Yerunkar et al. 2025 (preprint) |

### 6.2 ML SOTA for Clinical Trial Prediction

| Model | Year | Phase II F1 | Phase III F1 | Key Innovation |
|-------|------|-----------|-------------|---------------|
| HINT | 2022 | 0.620 | 0.847 | Hierarchical interaction network |
| inClinico | 2023 | 79% acc (self-reported) | — | Transformer + omics; **proprietary** (Insilico Medicine, not reproducible) |
| MEXA-CTP | 2025 | +11.3% | +11.3% | Mode experts cross-attention |
| LIFTED | 2025 | Enhanced | Enhanced | LLM-based multimodal MoE |
| MMCTOP | 2025 | 75.98 | 80.27 | Current SOTA on TOP |
| CLaDMoP | 2025 | Strong | Comparable | LLM eligibility + drug fusion |
| AutoCT | 2025 | Comparable | Comparable | LLM agents + MCTS |

### 6.3 Key Papers to Cite

**Must-cite (core narrative):**
1. Wong, Siah & Lo (2019) — Clinical trial success rates estimation (Biostatistics)
2. Showell et al. (2024) — Publication bias Cochrane review
3. Minikel et al. (2024) — Genetic evidence and clinical success (Nature)
4. Razuvayevskaya et al. (2024) — Trial stop reasons + genetics (Nature Genetics)
5. Fu et al. (2022) — HINT/TOP benchmark (Patterns)
6. Gao et al. (2026) — CTO benchmark (Nature Health)
7. TrialBench (2025) — Multi-task clinical trial benchmark (Sci Data)
8. Laviolle et al. (2025) — Negative trial publication trends (CPT)

**Should-cite (positioning):**
9. Ubels et al. (2020) — RAINFOREST: value of failed trial data (Bioinformatics)
10. Harrer et al. (2019) — AI for clinical trial design (Trends Pharm Sci)
11. Zhang et al. (2025) — MEXA-CTP (SIAM SDM)
12. Shi & Du (2024) — Finer-grained trial results (Sci Data)
13. CT-ADE (2025) — Adverse events benchmark (Sci Data)
14. CliniFact (2025) — Clinical claim verification (Sci Data)

---

## 7. Technical Feasibility

### 7.1 ETL Pipeline Design

```
Stage 1: AACT Bulk Load (~2 hours)
  ├── Download pipe-delimited files (studies, interventions, outcomes, ...)
  ├── Load into local SQLite
  ├── Filter: drug/biologic trials, Phase I-IV
  └── Extract: status, why_stopped, dates, sponsors

Stage 2: Failure Classification (~1 day)
  ├── Terminated/Withdrawn: Apply Open Targets NLP model (17 categories)
  ├── Completed with results: Parse outcome_analyses for p-values
  ├── Map to our 30+ category taxonomy
  └── Assign confidence tiers

Stage 3: Drug-Target Mapping (~1 day)
  ├── Intervention name → DrugBank ID (fuzzy matching + manual curation)
  ├── DrugBank ID → SMILES, targets (UniProt)
  ├── Cross-reference with ChEMBL drug_mechanism table
  └── Link to NegBioDB DTI data where available

Stage 4: Quantitative Outcome Extraction (~2 days)
  ├── AACT outcome_analyses table: p-values, CIs, effect sizes
  ├── Shi & Du 2024 dataset: structured p-values for 11K trials
  ├── Merge and deduplicate
  └── Quality checks

Stage 5: ML Export + Splits (~1 day)
  ├── Flatten to Parquet (intervention features, condition features, trial features)
  ├── Apply 6 split strategies
  ├── Generate positive labels from FDA-approved drug-indication pairs
  └── Leakage verification
```

### 7.2 Drug Name → Molecular Identity Mapping

This is the hardest technical challenge. ClinicalTrials.gov uses free-text intervention names, not standardized IDs.

**Approach:**
1. **DrugBank name mapping** — DrugBank has 15K+ drugs with names, synonyms, brand names → SMILES + targets
2. **ChEMBL drug_dictionary** — Maps drug names to ChEMBL IDs → targets
3. **PubChem name resolution** — PubChem compound name search → CID → SMILES
4. **NER/entity linking** — For complex names (e.g., "BMS-986158"), use regex patterns + manual curation
5. **CDEK mapping** — 22K APIs already disambiguated from ClinicalTrials.gov intervention names

**Estimated coverage:**
- Small molecule drugs: ~70-80% automatable via DrugBank + ChEMBL
- Biologics/antibodies: ~50-60% via DrugBank + UniProt
- Combination therapies: ~30-40% (requires decomposition)
- Non-drug interventions: N/A (excluded from molecular analysis)

### 7.3 Effort Estimates

| Component | Estimated Time | Dependencies |
|-----------|---------------|-------------|
| Schema design + migrations | 2 days | None |
| AACT ETL pipeline | 3 days | Schema |
| Open Targets NLP integration | 2 days | AACT ETL |
| Drug name → molecular mapping | 5 days | AACT ETL |
| Quantitative outcome extraction | 3 days | AACT ETL |
| ML export + splits | 2 days | All ETL |
| LLM dataset construction | 3 days | All ETL |
| Tests | 3 days | Parallel |
| ML baseline training + evaluation | 5 days | All ETL + splits |
| Iteration / debugging buffer | 5-7 days | Parallel |
| **Total** | **~23 working days + 10-12 days buffer = 6-8 weeks** | — |

### 7.4 Implementation Blockers & Design Decisions (Must Resolve Before Coding)

**B-1: Open Targets NLP Model Accessibility.** The model is hosted on a GCS bucket (`gs://ot-team/olesya/Stop reasons/bert_stop_reasons_revised`) that may not be publicly accessible. The GitHub repo (`LesyaR/stopReasons`) has no `requirements.txt` or documented TensorFlow version. **Action:** Test GCS access immediately. If blocked, retrain a BERT classifier from the 3,747 HuggingFace labels (budget 3-5 days, not 2). Also note: the Open Targets labels are **multi-label** (a trial can have both "Safety" and "Negative" reasons).

**B-2: Multi-Label vs. Single-Label Taxonomy.** The schema defines `failure_category` as a single `TEXT NOT NULL` column, but real trials often fail for multiple reasons (e.g., futility + safety signal). Open Targets labels are also multi-label. **Decision needed:** (a) Add a junction table `result_failure_categories(result_id, failure_category, confidence)` for multi-label, keeping `primary_failure_category` as a convenience column; or (b) force single-label with tie-breaking rules. This cascades into CT-M2 (multi-label classification vs. multiclass) and CT-L1 (MCQ format: single-answer vs. "select all that apply").

**B-3: "Completed with Negative Results" Detection Algorithm.** The document estimates "200K+" in the executive summary but no detection algorithm is specified. Proposal: `SELECT * FROM outcome_analyses WHERE outcome_type = 'Primary' AND p_value > 0.05` plus terminated/withdrawn trials = ~45K max candidates (20-40K after curation). Consider using CTO's 125K binary outcome labels (MIT) as a pre-filter. The "200K+" estimate should be treated as an upper bound; the executive summary already uses the more conservative "20K-40K curated."

**B-4: Positive Labels for ML Benchmark.** CT-M1 needs "success" labels. Options: (a) CTO binary outcome labels (MIT, 125K trials); (b) AACT `overall_status = 'Completed'` + primary endpoint p < 0.05; (c) ChEMBL `drug_indication` table for FDA-approved drug-indication pairs. **Recommendation:** CTO labels for CT-M1; ChEMBL approvals for CT-M3 phase transition. Define explicitly before splitting data.

**B-5: AACT Table Manifest.** AACT has 46 tables. Required tables for v1: `studies`, `interventions`, `conditions`, `design_outcomes`, `outcomes`, `outcome_measurements`, `outcome_analyses`, `browse_interventions`, `browse_conditions`, `design_groups`, `design_group_interventions`, `sponsors`, `documents`. `reported_events` (6.4M rows) deferred to v2 unless needed for safety failure classification.

**B-6: Drug Name Resolution Cascade.** Exact algorithm: (1) Exact match in ChEMBL `molecule_synonyms` → ChEMBL ID → SMILES + targets. (2) Exact match in PubChem compound name → CID → SMILES. (3) Fuzzy match (Jaro-Winkler > 0.85). (4) Manual curation list for top-100 unmapped drugs. (5) Mark remainder as `unmapped`. For biologics: add `canonical_sequence TEXT` column to `interventions` table. Run a pilot on 1,000 random AACT intervention names to empirically measure coverage before committing full effort.

### 7.5 Infrastructure

- **Database:** SQLite (same as DTI; estimated 1-3 GB)
- **Storage:** Parquet + CSV exports
- **Compute:** Local Mac for ETL; Cayuga HPC for ML baselines (reuse DTI infrastructure)
- **Cost:** $0 (all free data sources + existing infrastructure)

---

## 8. Execution Plan

### 8.1 Phase 1: Core Database (Weeks 1-3)

**Week 1: Scaffolding + AACT ETL**
- [ ] Schema DDL + migrations
- [ ] AACT bulk download + filtering pipeline
- [ ] Open Targets NLP model integration for failure classification
- [ ] Basic failure taxonomy (8 top-level + subcategories)

**Week 2: Drug-Target Mapping + Outcomes**
- [ ] DrugBank + ChEMBL drug name resolution pipeline
- [ ] Intervention → SMILES + UniProt target mapping
- [ ] Quantitative outcome extraction from AACT + Shi & Du 2024
- [ ] Confidence tier assignment

**Week 3: Export + Quality**
- [ ] ML export pipeline (Parquet + CSV)
- [ ] 6 split strategies implementation
- [ ] Positive data integration (FDA approvals)
- [ ] Data quality checks + leakage verification
- [ ] Tests (target: 100+)

### 8.2 Phase 2: Benchmarks (Weeks 4-5)

**Week 4: ML Benchmark**
- [ ] CT-M1: Trial outcome prediction baselines (HINT, MEXA-CTP)
- [ ] CT-M2: Failure reason classification baselines
- [ ] CT-M3: Phase transition prediction baselines

**Week 5: LLM Benchmark**
- [ ] CT-L1: Failure reason MCQ dataset (2,000)
- [ ] CT-L2: Trial report extraction dataset (200)
- [ ] CT-L3: Failure reasoning pilot (50)
- [ ] CT-L4: Phase transition judgment dataset (500)
- [ ] ~~CT-L5: Preclinical-clinical bridge pilot~~ — *deferred to future work (see Section 5.2)*

### 8.3 Phase 3: Paper + Release (Weeks 6-8)

- [ ] Croissant metadata + Datasheet
- [ ] HuggingFace + Zenodo upload
- [ ] Paper writing
- [ ] Integration with NegBioDB DTI paper (if timing allows)

### 8.4 Timeline vs NeurIPS Sprint

This work is **independent** of the NeurIPS DTI paper (due ~May 15, 2026). Two options:

**Option A: Separate paper** — Submit Clinical Trial Failure domain as standalone paper to ICLR 2027 D&B or Scientific Data
- Pro: No time pressure, can be thorough
- Con: Delayed impact

**Option B: Integrated section** — Add a "Domain Expansion: Clinical Trial Failure" section to the NeurIPS DTI paper as proof of extensibility
- Pro: Strengthens the "extensible architecture" claim
- Con: Tight timeline, may dilute DTI focus

**Recommendation: Option A** — focus the NeurIPS paper on DTI, use Clinical Trial Failure as a follow-up publication demonstrating extensibility.

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Drug name → molecular mapping low coverage | High | High | Start with DrugBank-mapped drugs only; iterate |
| CTO/TrialBench adds failure taxonomy before us | Low-Medium | High | Move fast; our molecular linkage is unique regardless |
| AACT data quality issues | Medium | Medium | Validate against ClinicalTrials.gov API; cross-check with publications |
| NLP failure classification accuracy | Medium | Medium | Use Open Targets model (validated); add manual curation for gold tier |
| "Completed with negative results" hard to identify | High | Medium | Start with terminated (explicit) + results-posted (quantitative); defer ambiguous |
| Scope creep with 30+ failure categories | Medium | Medium | Start with 8 top-level categories; add subcategories iteratively |
| OpenTrials cautionary tale (project died) | Low | High | Keep scope focused; release early; build on NegBioDB community |

### 9.1 Ethics & Responsible Use

**Key considerations for paper and dataset release:**
- **Patient privacy:** ClinicalTrials.gov and AACT contain only aggregate/summary-level data; no individual patient data is included. However, small trials (N<5) may allow re-identification — exclude or flag these.
- **Sponsor reputation:** Failure records link to named sponsors. Frame failures as scientific contributions (advancing knowledge), not corporate shortcomings. Avoid rankings that could be misused for stock manipulation.
- **Drug stigmatization:** A drug that failed for one indication may succeed for another. Dataset and paper must clearly communicate that failure is context-specific (indication, phase, population, endpoint).
- **IRB/ethics review:** No IRB required as all data is publicly available and aggregate-level. Note this explicitly in the paper's ethics statement.
- **Dual use:** The dataset could theoretically inform decisions to avoid developing certain drug classes. This is a feature (reducing wasted resources), not a bug, but should be discussed in the paper's broader impact section.

---

## 10. References

### Databases & Datasets
1. AACT: https://aact.ctti-clinicaltrials.org/
2. CTO Benchmark: https://github.com/sunlabuiuc/CTO | Nature Health 2026
3. TrialBench: https://github.com/ML2Health/ML2ClinicalTrials | Sci Data 2025
4. Open Targets Stop Reasons: https://huggingface.co/datasets/opentargets/clinical_trial_reason_to_stop
5. ClinicalRisk: https://dl.acm.org/doi/10.1145/3583780.3615113 | CIKM 2023
6. TrialPanorama: https://arxiv.org/abs/2505.16097
7. Shi & Du 2024: https://www.nature.com/articles/s41597-023-02869-7
8. CT-ADE: https://www.nature.com/articles/s41597-025-04718-1
9. CliniFact: https://www.nature.com/articles/s41597-025-04417-x
10. CDEK: https://academic.oup.com/database/article/doi/10.1093/database/baz087/5549735
11. ClinSR.org: https://www.nature.com/articles/s41467-025-64552-2
12. openFDA CRLs: https://open.fda.gov/apis/transparency/completeresponseletters/
13. CTKG: https://www.nature.com/articles/s41598-022-08454-z
14. InertDB: https://link.springer.com/article/10.1186/s13321-025-00999-1
15. HCDT 2.0: https://www.nature.com/articles/s41597-025-04981-2

### Key Papers
16. Wong CH, Siah KW, Lo AW. Estimation of clinical trial success rates. Biostatistics. 2019;20(2):273-286.
17. Showell MG et al. Time to publication for results of clinical trials. Cochrane Database Syst Rev. 2024.
18. Minikel EV et al. Refining the impact of genetic evidence on clinical success. Nature. 2024;629:624-629.
19. Razuvayevskaya O et al. Genetic factors associated with reasons for clinical trial stoppage. Nature Genetics. 2024;56:1862-1867.
20. Fu T et al. HINT: Hierarchical Interaction Network for Trial Outcome Prediction. Patterns. 2022;3(4):100445.
21. Gao C et al. CTO: A large-scale database for clinical trial outcomes. Nature Health. 2026.
22. Laviolle B et al. Trends of Publication of Negative Trials Over Time. CPT. 2025.
23. Ubels J et al. RAINFOREST: a random forest approach to predict treatment benefit in failed trials. Bioinformatics. 2020;36(Suppl_2):i601-i609.
24. Harrer S et al. Artificial Intelligence for Clinical Trial Design. Trends Pharmacol Sci. 2019;40(8):577-591.
25. Zhang Y et al. MEXA-CTP: Mode Experts Cross-Attention. SIAM SDM 2025.
26. Aliper A et al. Prediction of clinical trials outcomes based on target choice and clinical trial design with multi-modal AI. Clinical Pharmacology & Therapeutics. 2023;114(5):972-980.
27. Sun D, Gao W, Hu H, Zhou S. Why 90% of clinical drug development fails and how to improve it? Acta Pharmaceutica Sinica B. 2022;12(7):3049-3062.
28. Cummings JL, Morstorf T, Zhong K. Alzheimer's disease drug-development pipeline: few candidates, frequent failures. Alzheimer's Research & Therapy. 2014;6(4):37.
29. Nilsonne G et al. Results reporting for clinical trials led by medical universities and university hospitals in the Nordic countries. Journal of Clinical Epidemiology. 2025;178:111654.
30. Yerunkar P et al. Exploring scalable assessment methods for terminated trials. medRxiv. 2025. (preprint)

### ML Models & Tools
31. HINT/TOP: https://github.com/futianfan/clinical-trial-outcome-prediction
32. MEXA-CTP: https://github.com/murai-lab/MEXA-CTP
33. PyTrial: https://github.com/RyanWangZf/PyTrial
34. TrialGPT: https://github.com/ncbi-nlp/TrialGPT
35. Open Targets NLP: https://github.com/LesyaR/stopReasons
36. pytrials: https://pypi.org/project/pytrials/
37. trialbench: https://pypi.org/project/trialbench/
