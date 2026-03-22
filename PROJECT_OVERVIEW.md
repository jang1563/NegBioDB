# NegBioDB: Negative Results Database + Dual ML/LLM Benchmark

> Project Bible v3.4 | Last updated: 2026-03-22

## Project Vision

Approximately 90% of scientific experiments produce null or inconclusive results, yet the vast majority remain unpublished. This systematic gap fundamentally distorts AI/ML model training and evaluation.

**Goal:** Systematically collect and structure experimentally confirmed negative results across three biomedical domains, and build dual ML + LLM benchmarks to quantify the impact.

## Why This Matters

1. **Publication Bias**: 85% of published papers report only positive results (as of 2007)
2. **AI Model Bias**: Models trained without negative data produce excessive false positives
3. **Economic Waste**: Duplicated experiments, failed drug discovery pipelines (billions of dollars)
4. **Proven Impact**: Models trained with negative data are more accurate (Organic Letters 2023, bioRxiv 2024)
5. **No Competitor**: As of March 2026, no comparable database or benchmark exists

## Scope & Architecture

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
│  DTI   │ │  CT    │  │   PPI    │
│(Drug-  │ │(Trial  │  │(Protein- │
│Target) │ │Failure)│  │Protein)  │
└────────┘ └────────┘  └──────────┘
  ✅ Done    ✅ Done    ✅ Done
```

## Key Constraints

| Constraint | Detail |
|------------|--------|
| **Budget** | $0 pre-publication (free data sources, free/low-cost LLM tiers) |
| **License** | CC BY-SA 4.0 for NegBioDB (compatible with ChEMBL CC BY-SA 3.0) |
| **HPC** | Cornell Cayuga cluster (SLURM, A40 GPUs) |
| **Total Tests** | ~800 passing across all domains |
| **Timeline** | Project started 2026-03-02 (~20 days of development) |

---

## Domain 1: Drug-Target Interaction (DTI) -- COMPLETE

### Pipeline

```
Data Sources                    Pipeline                      Database
┌──────────┐    ┌────────────────────────────┐    ┌─────────────────────┐
│ ChEMBL   │───→│ etl_chembl.py              │───→│ compounds           │
│ DAVIS    │───→│ etl_davis.py               │───→│ targets             │
│ PubChem  │───→│ etl_pubchem.py             │───→│ negative_results    │
│ BindingDB│───→│ etl_bindingdb.py           │───→│ compound_target_pairs│
└──────────┘    └────────────────────────────┘    └─────────────────────┘
```

### Database

| Metric | Value |
|--------|-------|
| Negative results | 30.5M |
| Compounds | 919K |
| Targets | 3.7K |
| DB size | 13.22 GB |
| Tests | 329 passing |

### ML Benchmark (18/18 complete)

**3 models** (GraphDTA, DrugBAN, MLPBaseline) x **3 splits** (random, cold_drug, cold_target) x **2 negatives** (NegBioDB, random)

| Key Finding | Detail |
|-------------|--------|
| Negative source inflation | Degree-matched inflates LogAUC by +0.112 avg |
| Cold-target catastrophe | LogAUC drops to 0.15-0.33 (AUROC misleadingly stays 0.76-0.89) |
| DDB = random | Degree-balanced split ≤0.010 diff from random |
| NegBioDB trivially separable | Random split AUROC ~1.0 (strong signal in curated negatives) |

### LLM Benchmark (81/81 complete)

**5 models** (Llama-70B, Qwen-32B, GPT-4o-mini, Gemini-2.5-Flash, Haiku-4.5) x **4 levels** x **4 configs**

| Level | Task | Best Model | Score |
|-------|------|-----------|-------|
| L1 | MCQ (4-way) | Gemini | Acc 0.72 |
| L2 | Extraction | Qwen | field_f1 0.72 |
| L3 | Reasoning | Haiku | 4.66/5.0 |
| L4 | Discrimination | All near random | MCC ≤ 0.18 |

**Key finding:** LLMs cannot distinguish tested from untested DTI pairs (L4 MCC ≤ 0.18).

---

## Domain 2: Clinical Trial Failure (CT) -- COMPLETE

### Pipeline

```
Data Sources                    Pipeline                      Database
┌──────────┐    ┌────────────────────────────┐    ┌─────────────────────┐
│ AACT     │───→│ etl_aact.py (13 tables)    │───→│ clinical_trials     │
│ CTO      │───→│ etl_classify.py (3-tier)   │───→│ trial_failure_results│
│ Open Tgt │───→│ drug_resolver.py (4-step)  │───→│ interventions       │
│ Shi & Du │───→│ etl_outcomes.py (enrich)   │───→│ conditions          │
└──────────┘    └────────────────────────────┘    └─────────────────────┘
```

**6 modules:** AACT ETL → Failure Classification → Drug Resolution → Outcome Enrichment → DB Layer → ML Export

### Database

| Metric | Value |
|--------|-------|
| Clinical trials | 216,987 |
| Failure results | 132,925 |
| Interventions | 176,741 |
| Conditions | 55,915 |
| Intervention-condition pairs | 102,850 |
| DB size | ~500 MB |
| Tests | 200 passing (7 CT-specific test modules) |

**Tier distribution:** gold 23,570 (17.7%) / silver 28,505 (21.4%) / bronze 60,223 (45.3%) / copper 20,627 (15.5%)

**Category distribution:** efficacy 42.6% / enrollment 23.0% / other 21.0% / strategic 7.1% / safety 3.9% / design 1.8% / regulatory 0.7% / PK 0.0%

**Drug resolution:** 36,361/176,741 (20.6%) ChEMBL IDs; 27,534 SMILES; 66,393 target mappings

### Data Sources

| Source | License | Records | Purpose |
|--------|---------|---------|---------|
| AACT (ClinicalTrials.gov) | Public domain | 216,987 trials | Trial metadata, outcomes |
| CTO (Clinical Trial Outcome) | MIT | 20,627 records | Binary success/failure labels |
| Open Targets | Apache 2.0 | 32,782 targets | Drug-target mappings |
| Shi & Du 2024 | CC BY 4.0 | 119K + 803K rows | P-values, SAE data |

### Key Design Decisions

- **Failure taxonomy:** 8 categories with precedence (safety > efficacy > PK > enrollment > strategic > regulatory > design > other)
- **3-tier detection:** Tier 1 NLP on `why_stopped` (bronze) → Tier 2 p > 0.05 (silver/gold) → Tier 3 CTO labels (copper)
- **Drug resolution:** ChEMBL exact → PubChem API → fuzzy (JaroWinkler > 0.90) → manual CSV
- **Conflict removal:** CTO success pairs intersected with failure pairs; overlaps removed from both sides

### ML Export

| Dataset | Rows | Description |
|---------|------|-------------|
| `negbiodb_ct_pairs.parquet` | 102,850 | All failure pairs, 6 splits, all tiers |
| `negbiodb_ct_m1_balanced.parquet` | 11,222 | Binary (5,611 pos + 5,611 neg, silver+gold) |
| `negbiodb_ct_m1_realistic.parquet` | 36,957 | Binary (1:~6 ratio) |
| `negbiodb_ct_m1_smiles_only.parquet` | 3,878 | Binary (SMILES-resolved only) |
| `negbiodb_ct_m2.parquet` | 112,298 | 7-way category (non-copper) |

**6 splits:** random, cold_drug, cold_condition, temporal (≤2017/2018-2019/2020+), scaffold (Murcko), degree_balanced

**Integrity:** cold leakage = 0, M1 conflict-free verified

### ML Benchmark Results (108/108 complete)

**3 models** (XGBoost, MLP, GNN) x **6 splits** x **3 seeds** + control experiments

**CT-M1 (Binary Failure Prediction):**

| Split | Best Model | AUROC | Notes |
|-------|-----------|-------|-------|
| Random (NegBioDB) | XGBoost | 1.000 | Trivially separable |
| Random (uniform_random) | XGBoost | 0.905 | |
| Random (degree_matched) | XGBoost | 0.844 | Hardest control |
| Cold drug | XGBoost | 1.000 | |
| Cold condition | XGBoost | 1.000 | |
| Temporal | All | NaN | Single-class val set (expected) |

**CT-M2 (7-Way Category Classification):**

| Split | Best Model | macro-F1 | Notes |
|-------|-----------|----------|-------|
| Random | XGBoost | 0.510 | |
| Degree-balanced | XGBoost | 0.521 | |
| Cold drug | XGBoost | 0.414 | |
| Cold condition | XGBoost | 0.338 | |
| Scaffold | XGBoost | 0.193 | Hardest |
| Temporal | XGBoost | 0.193 | Hardest |

**Key CT-M findings:**
- NegBioDB negatives trivially solvable (AUROC ~1.0) — positive/negative separation is obvious
- Degree-matched hardest control (AUROC 0.76-0.84)
- Exp CT-1 inflation: -0.156 to -0.242 (XGBoost → GNN)
- M2: XGBoost dominates; scaffold/temporal hardest (mF1 ~0.19)

### LLM Benchmark Results (80/80 complete)

**5 models** x **4 levels** x **4 configs** (zero-shot + 3-shot x 3 fewshot sets)

**CT-L1 (5-Way MCQ, 1,500 records):**

| Model | 3-shot Acc | zero-shot Acc | MCC |
|-------|-----------|--------------|-----|
| Gemini-2.5-Flash | 0.667±0.014 | 0.681 | 0.597±0.015 |
| Haiku-4.5 | 0.662±0.012 | 0.660 | 0.592±0.014 |
| Qwen-32B | 0.648±0.017 | 0.654 | 0.572±0.024 |
| Llama-70B | 0.634±0.022 | 0.631 | 0.560±0.026 |
| GPT-4o-mini | 0.625±0.011 | 0.641 | 0.546±0.012 |

**CT-L2 (Extraction, 500 records):**

| Model | 3-shot field_f1 | 3-shot cat_acc | schema |
|-------|----------------|---------------|--------|
| Qwen-32B | 0.808±0.162 | 0.709±0.095 | 1.000 |
| Llama-70B | 0.768±0.161 | 0.752±0.064 | 1.000 |
| Gemini-2.5-Flash | 0.746±0.162 | 0.742±0.068 | 1.000 |
| GPT-4o-mini | 0.734±0.185 | 0.715±0.089 | 0.917 |
| Haiku-4.5 | 0.476±0.099 | 0.738±0.055 | 1.000 |

**CT-L3 (Reasoning, 200 records) -- Ceiling effect:**

| Model | 3-shot overall |
|-------|---------------|
| Haiku-4.5 | 4.960±0.007 |
| Qwen-32B | 4.968±0.007 |
| Llama-70B | 4.826±0.007 |
| GPT-4o-mini | 4.743±0.058 |
| Gemini-2.5-Flash | 4.462±0.050 |

> GPT-4o-mini as judge gives 4.4-5.0/5.0 (too lenient). Only completeness dimension discriminates.

**CT-L4 (Discrimination, 500 records):**

| Model | 3-shot Acc | 3-shot MCC |
|-------|-----------|-----------|
| Gemini-2.5-Flash | 0.777±0.011 | 0.563±0.018 |
| Haiku-4.5 | 0.739±0.019 | 0.502±0.014 |
| Llama-70B | 0.739±0.023 | 0.504±0.036 |
| GPT-4o-mini | 0.738±0.008 | 0.485±0.007 |
| Qwen-32B | 0.724±0.017 | 0.484±0.018 |

**Key CT-LLM findings:**
- CT L4 shows **meaningful discrimination** (MCC 0.48-0.56) — unlike DTI L4 (MCC ≤ 0.18)
- Gemini best overall (L1 + L4)
- L3 ceiling effect renders judge scores non-discriminative
- All models achieve 100% evidence citation rate (heuristic threshold too easy)

### CT Code Quality Audit (2026-03-21)

| Area | Lines | Grade | Critical Bugs |
|------|-------|-------|---------------|
| Pipeline (5 modules) | ~3,165 | A | 0 |
| ML Export (6 modules) | ~2,800 | A+ | 0 |
| LLM Benchmark (9 modules) | ~1,814 | B+ | 0 |
| Tests (7 modules) | ~3,400 | A | 0 gaps |
| SLURM (9 scripts) | ~1,200 | A- | 0 |

**0 critical bugs, 0 data leakage, 4 LOW design observations.**

---

## Domain 3: Protein-Protein Interaction (PPI) -- COMPLETE

### Pipeline

```
Data Sources                    Pipeline                      Database
┌──────────┐    ┌────────────────────────────┐    ┌─────────────────────┐
│ IntAct   │───→│ etl_intact.py (PSI-MI TAB) │───→│ proteins            │
│ HuRI     │───→│ etl_huri.py (Y2H screen)   │───→│ negative_results    │
│ hu.MAP   │───→│ etl_humap.py (ML-derived)  │───→│ protein_protein_pairs│
│ STRING   │───→│ etl_string.py (zero-score) │───→│ ppi_split_*         │
└──────────┘    └────────────────────────────┘    └─────────────────────┘
```

### Database

| Metric | Value |
|--------|-------|
| Proteins | 18,412 |
| Negative results | 2,229,670 |
| Aggregated pairs | 2,220,786 |
| Multi-source overlaps | 8,800 |
| Positive pairs (HuRI) | 61,728 (578 conflicts removed) |
| DB size | 849 MB |
| Tests | 386 passing (176 pipeline + 109 ML + 101 LLM) |

**Tier distribution:** gold 500,069 (HuRI + IntAct) / silver 1,229,601 (hu.MAP + IntAct) / bronze 500,000 (STRING)

### Data Sources

| Source | License | Records | Purpose |
|--------|---------|---------|---------|
| IntAct | CC BY 4.0 | 779 pairs | Curated non-interactions (gold 69 / silver 710) |
| HuRI | CC BY 4.0 | 500,000 pairs | Y2H systematic screen negatives (gold) |
| hu.MAP 3.0 | MIT | 1,228,891 pairs | ML-derived from ComplexPortal (silver) |
| STRING v12.0 | CC BY 4.0 | 500,000 pairs | Zero-score pairs (bronze) |

### ML Benchmark Results (54/54 complete)

**3 models** (SiameseCNN, PIPR, MLPFeatures) x **4 splits** x **3 seeds** + controls

| Model | Random AUROC | Cold Protein | Cold Both | DDB |
|-------|-------------|-------------|----------|-----|
| SiameseCNN | 0.963±0.000 | 0.873±0.002 | 0.585±0.040 | 0.962±0.001 |
| PIPR | 0.964±0.001 | 0.859±0.008 | **0.409±0.077** | 0.964±0.000 |
| MLPFeatures | 0.962±0.001 | 0.931±0.001 | **0.950±0.021** | 0.961±0.000 |

**Key PPI-ML findings:**
- **PIPR cold_both catastrophic** (AUROC 0.409 — below random!) — cross-attention fails on unseen protein pairs
- **MLPFeatures cold_both robust** (0.950) — hand-crafted features (degree, subcellular loc) generalize
- **Negative source effect is MODEL-DEPENDENT:** sequence models +6-9% inflation (same as DTI); MLPFeatures reversed (-5% to -19%, NegBioDB harder)
- DDB ≈ random (same as DTI)

### LLM Benchmark Results (80/80 complete)

**Design:** research/16_ppi_llm_benchmark_design.md (1,005 lines)

**Modules:** `llm_prompts.py`, `llm_eval.py`, `llm_dataset.py` (101 PPI LLM tests passing)

**PPI-L1 (4-Way MCQ, 1,200 records):**

| Model | 3-shot Acc | 3-shot MCC | zero-shot Acc |
|-------|-----------|-----------|--------------|
| Haiku-4.5 | 0.999±0.001 | 0.999±0.001 | 0.750 |
| GPT-4o-mini | 1.000±0.001 | 0.999±0.001 | 0.749 |
| Gemini-2.5-Flash | 1.000±0.001 | 0.999±0.001 | 0.750 |
| Llama-70B | 1.000 | 1.000 | 0.750 |
| Qwen-32B | 0.826±0.069 | 0.803±0.069 | 0.750 |

**PPI-L2 (Extraction, 500 records):**

| Model | 3-shot entity_f1 | schema_compliance | zero-shot entity_f1 |
|-------|------------------|------------------|---------------------|
| Haiku-4.5 | 1.000 | 1.000 | 1.000 |
| Llama-70B | 1.000 | 1.000 | 1.000 |
| GPT-4o-mini | 0.999 | 1.000 | 0.999 |
| Qwen-32B | 0.998 | 1.000 | 0.999 |
| Gemini-2.5-Flash | 1.000 | 1.000 | 0.952 |

**PPI-L3 (Reasoning, 200 records, Gemini-2.5-Flash judge):**

| Model | zero-shot Overall | 3-shot Overall |
|-------|------------------|---------------|
| Haiku-4.5 | **4.68** | 3.97 |
| Gemini-2.5-Flash | 4.65 | 3.48 |
| Qwen-32B | 4.45 | 3.73 |
| GPT-4o-mini | 4.36 | 3.41 |
| Llama-70B | 4.28 | 1.00 (catastrophic) |

**PPI-L4 (Discrimination, 500 records):**

| Model | zero-shot Acc | zero-shot MCC | 3-shot MCC | Contamination Gap |
|-------|-------------|--------------|-----------|-------------------|
| Llama-70B | 0.703 | 0.441 | 0.371±0.056 | **0.590** |
| GPT-4o-mini | 0.699 | 0.430 | 0.352±0.039 | **0.517** |
| Qwen-32B | 0.645 | 0.366 | 0.369±0.009 | **0.527** |
| Gemini-2.5-Flash | 0.647 | 0.358 | 0.382±0.004 | **0.456** |
| Haiku-4.5 | 0.608 | 0.334 | 0.390±0.020 | **0.401** |

**Contamination vs Protein Popularity (all 5 models → true contamination):**

| Model | Avg Gap High-Degree | Avg Gap Low-Degree | Verdict |
|-------|--------------------|--------------------|---------|
| Haiku-4.5 | 0.580 | 0.434 | True contamination |
| Gemini-2.5-Flash | 0.498 | 0.532 | True contamination (stronger for obscure) |
| GPT-4o-mini | 0.555 | 0.330 | True contamination |
| Llama-70B | 0.532 | 0.406 | True contamination |
| Qwen-32B | 0.541 | 0.378 | True contamination |

**Key PPI-LLM findings:**
- **L1 3-shot is trivially solvable** (~1.0 accuracy) — evidence descriptions too informative with examples
- **L2 near-perfect** (entity_f1 ~1.0) — structured extraction is easy for all models
- **L3 NO ceiling effect** (unlike CT-L3). Overall 1.0-4.68 — genuine discrimination. Structural reasoning hardest. Zero-shot > 3-shot.
- **L3 Haiku zero-shot best** (4.68), Gemini 2nd (4.65). Llama 3-shot catastrophe (all 1.0).
- **L4 moderate discrimination** (MCC 0.33-0.44) — between DTI (≤0.18) and CT (~0.5)
- **L4 MASSIVE contamination** — All 5 models gap > 0.40 (threshold: 0.15). True contamination confirmed via popularity analysis.
- **100% evidence citation rate** — LLMs hallucinate evidence for untested pairs (same as DTI/CT)

---

## Cross-Domain Comparison

### Database Scale

| Domain | Negative Results | Entities | DB Size |
|--------|-----------------|----------|---------|
| DTI | 30.5M | 919K compounds, 3.7K targets | 13.22 GB |
| CT | 132,925 | 176K interventions, 56K conditions | ~500 MB |
| PPI | 2.23M | 18.4K proteins | 849 MB |
| **Total** | **~32.9M** | | **~14.6 GB** |

### ML Benchmark Scale

| Domain | Tasks | Models | Splits | Seeds | Total Runs |
|--------|-------|--------|--------|-------|-----------|
| DTI | 1 (M1) | 3 | 3+2 | 1 | 18 |
| CT | 2 (M1+M2) | 3 | 6 | 3 | 108 |
| PPI | 1 (M1) | 3 | 4+2 | 3 | 54 |
| **Total** | | | | | **180** |

### LLM Benchmark Scale

| Domain | Levels | Models | Configs | Total Runs | L4 MCC Range | Contamination |
|--------|--------|--------|---------|-----------|-------------|---------------|
| DTI | 4 | 5+1 | 4 | 81 | ≤ 0.18 (random) | Not detected |
| CT | 4 | 5 | 4 | 80 | 0.48-0.56 (meaningful) | Not detected |
| PPI | 4 | 5 | 4 | 80 | 0.33-0.44 (moderate) | **MASSIVE (0.40-0.59)** |
| **Total** | | | | **241** | | |

### Unified Cross-Domain Summary

| Metric | DTI | CT | PPI |
|--------|-----|-----|-----|
| **Database** | | | |
| Negative results | 30.5M | 132,925 | 2.23M |
| Key entities | 919K compounds, 3.7K targets | 176K interventions | 18.4K proteins |
| DB size | 13.22 GB | ~500 MB | 849 MB |
| Data sources | 4 (ChEMBL, PubChem, BindingDB, DAVIS) | 4 (AACT, CTO, OpenTargets, Shi&Du) | 4 (IntAct, HuRI, hu.MAP, STRING) |
| **ML Benchmark** | | | |
| Total runs | 18 | 108 | 54 |
| Random split AUROC | ~1.0 | 1.0 | 0.96 |
| Hardest split | cold_target (0.15-0.33 LogAUC) | scaffold/temporal (mF1 0.19) | cold_both PIPR (0.41 AUROC) |
| Neg source inflation | +0.112 LogAUC (degree_matched) | -0.156 to -0.242 AUROC | +6-9% seq models, -5-19% MLPFeatures |
| DDB vs random | ≤0.010 diff | ≤0.011 diff | ≤0.003 diff |
| **LLM Benchmark** | | | |
| Total runs | 81 | 80 | 80 |
| L1 (MCQ) best | Llama 0.991 (4-class) | Gemini 0.681 (5-class) | Llama 1.000 (4-class) |
| L2 (extraction) best | Qwen field_f1 0.72 | Qwen field_f1 0.81 | Haiku entity_f1 1.00 |
| L3 (reasoning) best | Gemini 4.66/5.0 | Haiku 5.00/5.0 (ceiling!) | Haiku 4.68/5.0 |
| L3 ceiling effect? | No (1.96-4.66 range) | **Yes** (4.4-5.0, GPT judge) | No (1.0-4.68 range) |
| L4 (discrimination) best MCC | Llama 0.184 | Gemini 0.563 | Llama 0.441 |
| L4 contamination | Not detected | Not detected | **Gap 0.40-0.59** |
| Evidence hallucination | 100% | 100% | 100% |

### Key Cross-Domain Insights

1. **L4 discrimination gradient:** DTI (opaque, MCC ≤ 0.18) < PPI (memorized, MCC ~0.4) < CT (partially reasoned, MCC ~0.5). DTI data is locked in specialized databases invisible to LLMs; PPI databases (IntAct, STRING) are in training corpora; clinical trial data on ClinicalTrials.gov is publicly accessible.

2. **PPI contamination is the smoking gun:** All 5 models show contamination gap > 0.40 (threshold: 0.15). Contamination vs popularity analysis confirms this is **true memorization**, not protein popularity bias — the gap persists for both high- and low-degree proteins. Gemini uniquely shows stronger contamination for obscure proteins (pure memorization signal).

3. **NegBioDB negatives are trivially separable by ML:** AUROC ~1.0 across all 3 domains on random splits. This is by design — curated negatives carry distinct signatures vs positives. The value of NegBioDB emerges in cold/temporal splits where models must generalize.

4. **L3 judge quality varies by domain:** DTI uses Gemini Flash-Lite (good discrimination, 1.96-4.66 range). CT uses GPT-4o-mini (ceiling effect, 4.4-5.0). PPI uses Gemini 2.5 Flash (good discrimination, 1.0-4.68 range). Judge choice critically affects L3 utility.

5. **Evidence hallucination is universal:** All models across all domains cite evidence for untested pairs at 100% rate. LLMs cannot distinguish "I know this was tested" from "I can generate plausible-sounding evidence."

---

## Implementation Progress

| Step | DTI | CT | PPI |
|------|-----|-----|-----|
| Infrastructure | ✅ | ✅ | ✅ |
| Data loading | ✅ | ✅ | ✅ |
| Enrichment/Resolution | ✅ | ✅ | ✅ |
| Benchmark design | ✅ | ✅ | ✅ |
| ML export & splits | ✅ | ✅ | ✅ |
| ML experiments | ✅ 18/18 | ✅ 108/108 | ✅ 54/54 |
| LLM benchmark | ✅ 81/81 | ✅ 80/80 | ✅ 80/80 |
| L3 judge | ✅ | ✅ 20/20 | ✅ 20/20 |
| Contamination analysis | N/A | N/A | ✅ |
| Paper writing | Planned | Planned | Planned |

---

## Project Documents

| Document | Description |
|----------|-------------|
| [research/01](research/01_dti_negative_data_landscape.md) | DTI negative data sources landscape |
| [research/02](research/02_benchmark_analysis.md) | Existing DTI benchmarks analysis |
| [research/03](research/03_data_collection_methodology.md) | Data collection methodologies |
| [research/04](research/04_publication_commercial_strategy.md) | Publication and commercialization strategy |
| [research/05](research/05_technical_deep_dive.md) | APIs, licenses, deduplication, baselines |
| [research/06](research/06_paper_narrative.md) | Paper title/abstract, positioning |
| [research/07a](research/07a_llm_benchmark_landscape_survey.md) | Bio/chem LLM benchmark survey |
| [research/07b](research/07b_llm_benchmark_design.md) | LLM benchmark: 6 tasks, dual-track |
| [research/08](research/08_expert_review_and_feasibility.md) | Expert review, feasibility analysis |
| [research/09](research/09_schema_and_ml_export_design.md) | Schema DDL, ML export, Croissant |
| [research/10](research/10_expert_panel_review.md) | 6-expert panel review |
| [research/11](research/11_full_plan_review.md) | Pre-implementation audit |
| [research/12](research/12_review_findings_summary.md) | Schema/pipeline review findings |
| [research/13](research/13_clinical_trial_failure_domain.md) | CT domain design |
| [research/14](research/14_ct_ml_benchmark_design.md) | CT ML benchmark design |
| [research/15](research/15_ct_llm_benchmark_design.md) | CT LLM benchmark design |
| [research/16](research/16_ppi_llm_benchmark_design.md) | PPI LLM benchmark design |
| [research/17a](research/17_ct_expert_panel_review.md) | CT domain 6-expert panel review |
| [research/17b](research/17_ppi_expert_panel_review.md) | PPI domain 6-expert panel review |

## Timeline

- 2026-03-02: Project initiated
- 2026-03-13: DTI domain complete (ML + LLM)
- 2026-03-17: CT domain initiated
- 2026-03-18: PPI domain initiated
- 2026-03-20: CT domain complete (ML 108/108 + LLM 80/80)
- 2026-03-21: PPI ML complete (54/54), LLM jobs submitted
- 2026-03-22: PPI LLM complete (80/80), L3 judged, contamination analysis done. All 3 domains complete.
- Next: Paper writing
