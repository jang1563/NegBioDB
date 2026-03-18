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
| **Budget** | $0 pre-publication (free data sources, free LLM tiers, free infrastructure) |
| **License** | CC BY-SA 4.0 for NegBioDB (compatible with ChEMBL CC BY-SA 3.0) |
| **HCDT 2.0** | CC BY-NC-ND — cannot integrate directly; independently recreate from underlying sources |
| **LLM Pipeline** | Mistral 7B (ollama) + Gemini 2.5 Flash free tier |

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

**DB:** 30.5M negative_results · 919K compounds · 3.7K targets · 13.22 GB

### Key ML Results (18/18 complete)
- **Exp 1:** Degree-matched negatives inflate LogAUC by +0.112 avg → benchmark inflation confirmed
- **Split effect:** Cold-target LogAUC drops to 0.15–0.33 while AUROC stays 0.76–0.89 → AUROC misleading
- **Exp 4:** DDB ≈ random (≤0.010 diff) → degree balancing alone is not harder

### Key LLM Results (81/81 complete)
- **L4:** All models near random (MCC ≤ 0.18) → LLMs cannot distinguish tested from untested pairs

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

**Tier distribution:** gold 23,570 (17.7%) / silver 28,505 (21.4%) / bronze 60,223 (45.3%) / copper 20,627 (15.5%)

**Category distribution:** efficacy 42.6% / enrollment 23.0% / other 21.0% / strategic 7.1% / safety 3.9% / design 1.8% / regulatory 0.7% / PK 0.0%

**Drug resolution:** 36,361/176,741 interventions (20.6%) have ChEMBL IDs; 27,534 (15.6%) have SMILES; 66,393 intervention-target mappings

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

## Project Documents

| Document | Description |
|----------|-------------|
| [research/01_dti_negative_data_landscape.md](research/01_dti_negative_data_landscape.md) | Current DTI negative data sources landscape |
| [research/02_benchmark_analysis.md](research/02_benchmark_analysis.md) | Analysis of existing DTI benchmarks and their negative data handling |
| [research/03_data_collection_methodology.md](research/03_data_collection_methodology.md) | Data collection, curation, and structuring methodologies |
| [research/04_publication_commercial_strategy.md](research/04_publication_commercial_strategy.md) | Academic publication and commercialization strategy |
| [research/05_technical_deep_dive.md](research/05_technical_deep_dive.md) | Data access APIs, license analysis, deduplication, baselines, metrics |
| [research/06_paper_narrative.md](research/06_paper_narrative.md) | Paper title/abstract, competitive positioning |
| [research/07a_llm_benchmark_landscape_survey.md](research/07a_llm_benchmark_landscape_survey.md) | Survey of existing bio/chem LLM benchmarks and evaluation methods |
| [research/07b_llm_benchmark_design.md](research/07b_llm_benchmark_design.md) | LLM benchmark: 6 tasks, evaluation methods, dual-track architecture |
| [research/08_expert_review_and_feasibility.md](research/08_expert_review_and_feasibility.md) | Expert review responses, feasibility analysis, revised scope |
| [research/09_schema_and_ml_export_design.md](research/09_schema_and_ml_export_design.md) | SQLite schema DDL, ML export patterns, Croissant metadata, Datasheet for Datasets |
| [research/10_expert_panel_review.md](research/10_expert_panel_review.md) | 6-expert panel review: reviewer, data eng, ML, domain, SW arch, PM |
| [research/11_full_plan_review.md](research/11_full_plan_review.md) | Pre-implementation audit: 16 issues found, feasibility ratings, execution adjustments |
| [research/12_review_findings_summary.md](research/12_review_findings_summary.md) | Schema/pipeline implementation review: 9 issues (3 critical, 3 high, 2 moderate, 1 low) |
| [research/13_clinical_trial_failure_domain.md](research/13_clinical_trial_failure_domain.md) | CT domain design: failure taxonomy, 3-tier detection, pipeline architecture |
| [research/14_ct_ml_benchmark_design.md](research/14_ct_ml_benchmark_design.md) | CT ML benchmark: 3 tasks, 6 splits, 3 models, 3 experiments |
| [research/15_ct_llm_benchmark_design.md](research/15_ct_llm_benchmark_design.md) | CT LLM benchmark: 4 levels, 5 models, contamination analysis |

## Timeline
- Project initiated: 2026-03-02
- CT domain initiated: 2026-03-17
- Last updated: 2026-03-18
