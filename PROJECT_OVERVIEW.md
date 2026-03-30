# NegBioDB: Negative Results Database for Drug-Target Interactions

> Biology-first, Science-extensible negative results database and dual ML+LLM benchmark

## Project Vision

Approximately 90% of scientific experiments produce null or inconclusive results, yet the vast majority remain unpublished. This systematic gap fundamentally distorts AI/ML model training and evaluation.

**Goal:** Starting with Drug-Target Interactions (DTI), systematically collect and structure experimentally confirmed negative results, and build benchmarks for AI/ML training and evaluation.

## Why This Matters

1. **Publication Bias**: 85% of published papers report only positive results (as of 2007)
2. **AI Model Bias**: Models trained without negative data produce excessive false positives
3. **Economic Waste**: Duplicated experiments, failed drug discovery pipelines (billions of dollars)
4. **Proven Impact**: Models trained with negative data are more accurate (Organic Letters 2023, bioRxiv 2024)

## Scope & Strategy

```
Biology-first, Science-extensible Architecture
┌─────────────────────────────────────┐
│  Common Layer                        │
│  - Hypothesis structure              │
│  - Experimental metadata             │
│  - Outcome classification            │
│  - Confidence / Statistical power    │
│  - Author annotation                 │
└──────────────┬──────────────────────┘
               │
    ┌──────────┼──────────────┐
    ▼          ▼              ▼
┌────────┐ ┌────────┐  ┌──────────┐
│Biology │ │Chem    │  │Materials │  ← Phase 2+
│(DTI)   │ │Domain  │  │Domain    │
└────────┘ └────────┘  └──────────┘
```

**Expansion Path:** DTI → Clinical Trial Failure → Gene Function → Chemistry → Materials Science

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Biology-first | Most severe problem, highest commercial value, largest AI evaluation gap |
| Starting Domain | Drug-Target Interaction | Data accessibility + existing infrastructure (ChEMBL) + pharma demand |
| Architecture | Extensible (common + domain layers) | Future expansion to Chemistry, Materials |

## Key Constraints

| Constraint | Detail |
|------------|--------|
| **License** | CC BY-SA 4.0 for NegBioDB (compatible with ChEMBL CC BY-SA 3.0) |
| **HCDT 2.0** | CC BY-NC-ND — cannot integrate directly; independently recreate from underlying sources |

## DTI Domain Implementation Progress (as of 2026-03-13)

| Step | Component | Status |
|------|-----------|--------|
| 1 | Schema & scaffolding | ✅ Complete |
| 2a | Data download (4 sources) | ✅ Complete |
| 2b | ETL: DAVIS, ChEMBL, PubChem, BindingDB | ✅ Complete |
| 3 | ML export & splits (6 strategies) | ✅ Complete |
| 4 | ML baseline models + SLURM harness | ✅ Complete |
| 5 | ML evaluation metrics (7 metrics, 329 tests) | ✅ Complete |
| 6a | ML baseline experiments (18/18 runs on Cayuga) | ✅ Complete |
| 6b | LLM benchmark infrastructure (L1–L4 datasets, prompts, eval, SLURM) | ✅ Complete |
| 6c | LLM benchmark execution (81/81 complete) | ✅ Complete |
| 7 | Paper writing & submission | Planned |

**DB:** [Database statistics pending publication]

### Key ML Results (18/18 complete)
- **Exp 1:** Degree-matched negatives inflate LogAUC — [results pending publication]
- **Split effect:** Cold-target splits reveal metric discrepancies — [results pending publication]
- **Exp 4:** DDB vs. random comparison — [results pending publication]

### Key LLM Results (81/81 complete)
- **L4:** [Results pending publication]

---

## Clinical Trial Failure Domain (NegBioDB-CT)

The second domain extends NegBioDB to clinical trial failures, capturing why drugs fail in human trials.

### Architecture

```
Data Sources                    Pipeline                      Database
┌──────────┐    ┌────────────────────────────┐    ┌─────────────────────┐
│ AACT     │───→│ etl_aact.py (13 tables)    │───→│ clinical_trials     │
│ CTO      │───→│ etl_classify.py (3-tier)   │───→│ trial_failure_results│
│ Open Tgt │───→│ drug_resolver.py (4-step)  │───→│ interventions       │
│ Shi & Du │───→│ etl_outcomes.py (enrich)   │───→│ conditions          │
└──────────┘    └────────────────────────────┘    └─────────────────────┘
```

**5 modules:** AACT ETL → Failure Classification → Drug Resolution → Outcome Enrichment → DB Layer

### Database State (as of 2026-03-18)

| Metric | Value |
|--------|-------|
| Clinical trials | 216,987 |
| Failure results | 132,925 |
| Interventions | 176,741 |
| Conditions | 55,915 |
| Intervention-condition pairs | 102,850 |

**Tier distribution:** [Results pending publication]

**Category distribution:** [Results pending publication]

**Drug resolution:** [Results pending publication]

### Data Sources

| Source | License | Records | Purpose |
|--------|---------|---------|---------|
| AACT (ClinicalTrials.gov) | Public domain | 216,987 trials | Trial metadata, outcomes |
| CTO (Clinical Trial Outcome) | MIT | 20,627 records | Binary success/failure labels |
| Open Targets | Apache 2.0 | 32,782 targets | Drug-target mappings |
| Shi & Du 2024 | CC BY 4.0 | 119K efficacy + 803K safety rows | P-values, SAE data |

### Key Design Decisions

- **Failure taxonomy:** 8 categories (safety > efficacy > PK > enrollment > strategic > regulatory > design > other)
- **3-tier detection:** Tier 1 NLP on `why_stopped` (bronze) → Tier 2 p-value analysis (silver/gold) → Tier 3 CTO labels (copper)
- **Drug resolution:** ChEMBL exact → PubChem API → fuzzy (JaroWinkler > 0.90) → manual CSV overrides
- **Tier upgrades:** Bronze + p-value → Silver, Silver + Phase III + PubMed → Gold

### Benchmark Design

**ML Benchmark** (3 tasks × 3 models × 6 splits): See [research/14](research/14_ct_ml_benchmark_design.md)
- CT-M1: Drug-condition failure prediction (binary)
- CT-M2: Failure category classification (7-way)
- CT-M3: Phase transition prediction (deferred)

**LLM Benchmark** (4 levels × 5 models): See [research/15](research/15_ct_llm_benchmark_design.md)
- CT-L1: Failure category MCQ (5-way, 1,500 records)
- CT-L2: Failure report extraction (500 records)
- CT-L3: Failure reasoning (200 records)
- CT-L4: Trial existence discrimination (500 records)

### Implementation Progress (as of 2026-03-18)

| Step | Component | Status |
|------|-----------|--------|
| CT-1 | Schema & scaffolding (2 migrations) | ✅ Complete |
| CT-2 | Data loading (4 sources) | ✅ Complete |
| CT-3 | Enrichment & resolution | ✅ Complete |
| CT-4 | Analysis & benchmark design | ✅ Complete |
| CT-5 | ML export & splits | Planned |
| CT-6 | ML baseline experiments | Planned |
| CT-7 | LLM benchmark execution | Planned |

---


## Timeline
- Project initiated: 2026-03-02
- CT domain initiated: 2026-03-17
- Last updated: 2026-03-18
