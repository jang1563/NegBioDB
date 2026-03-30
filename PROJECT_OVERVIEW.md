# NegBioDB: Negative Results Database & Dual ML/LLM Benchmark

> Biology-first, science-extensible negative results database and dual ML+LLM benchmark

*Last updated: 2026-03-30*

---

## Project Vision

Approximately 90% of scientific experiments produce null or inconclusive results, yet the vast majority remain unpublished. This systematic gap fundamentally distorts AI/ML model training and evaluation.

**Goal:** Systematically collect and structure experimentally confirmed negative results across biomedical domains, and build benchmarks that quantify the impact of publication bias on AI/ML models.

## Why This Matters

1. **Publication Bias**: 85% of published papers report only positive results
2. **AI Model Bias**: Models trained without negative data produce excessive false positives
3. **Economic Waste**: Duplicated experiments, failed drug discovery pipelines (billions of dollars)
4. **Proven Impact**: Models trained with negative data are more accurate (Organic Letters 2023, bioRxiv 2024)

---

## Architecture

```
Four Biomedical Domains
┌────────────────────────────────────────────────────────────┐
│                      NegBioDB                               │
│  DTI          CT            PPI           GE               │
│  (30.5M neg)  (133K neg)    (2.2M neg)    (28.8M neg)      │
│  ChEMBL+      AACT+         IntAct+       DepMap           │
│  PubChem+     CTO+          HuRI+         CRISPR+RNAi      │
│  BindingDB+   OpenTargets+  hu.MAP+                        │
│  DAVIS        Shi&Du        STRING                         │
└────────────────────────────────────────────────────────────┘
         │                │
  ┌──────┴──────┐   ┌─────┴──────┐
  │ ML Benchmark │   │LLM Benchmark│
  │ 3 models ×   │   │ 5 models ×  │
  │ 5 splits ×   │   │ 4 levels ×  │
  │ 2 neg types  │   │ 4 configs   │
  └─────────────┘   └────────────┘
```

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| License | CC BY-SA 4.0 | Compatible with ChEMBL CC BY-SA 3.0 (viral clause) |
| Storage | SQLite per domain | Portable, zero-infrastructure, reproducible |
| Export | Parquet with split columns | Standard ML format; lazy-loading friendly |
| ML metrics | LogAUC + 6 others | LogAUC[0.001,0.1] measures early enrichment, not just AUROC |
| LLM evaluation | 4 levels (L1–L4) | Progressive difficulty: MCQ → extraction → reasoning → discrimination |

---

## Domain Status Summary (as of 2026-03-30)

| Domain | DB Size | Negatives | ML Runs | LLM Runs | Status |
|--------|---------|-----------|---------|----------|--------|
| **DTI** | ~21 GB | 30,459,583 | 24/24 ✅ | 81/81 ✅ | Complete |
| **CT** | ~500 MB | 132,925 | 108/108 ✅ | 80/80 ✅ | Complete |
| **PPI** | 849 MB | 2,229,670 | 54/54 ✅ | 80/80 ✅ | Complete |
| **GE** | ~16 GB | 28,759,256 | Seed 42 ✅ | 4/5 models ✅ | Near complete |

---

## DTI Domain (Drug-Target Interaction)

Four sources: ChEMBL v36, PubChem BioAssay, BindingDB, DAVIS

### Database
- **30,459,583** negative results
- Source tiers: gold 818,611 / silver 198 / bronze 28,845,632
- 5 split strategies: random / cold_compound / cold_target / scaffold / temporal

### Key Results
- **ML:** Degree-matched negatives inflate LogAUC by +0.112 on average. Cold-target splits catastrophic (LogAUC 0.15–0.33) while AUROC stays deceptively high (0.76–0.89).
- **LLM L4:** All models near-random (MCC ≤ 0.18). DTI binding decisions are too nuanced for LLMs without domain context.
- **LLM L1:** Gemini achieves perfect accuracy (1.000) on 3-shot MCQ — artifact of format recognition.

---

## CT Domain (Clinical Trial Failure)

Four sources: AACT (ClinicalTrials.gov), CTO, Open Targets, Shi & Du 2024

### Database
- **132,925** failure results from 216,987 trials
- Tiers: gold 23,570 / silver 28,505 / bronze 60,223 / copper 20,627
- 8 failure categories: safety > efficacy > enrollment > strategic > regulatory > design > other
- Drug resolution: 4-step pipeline (ChEMBL exact → PubChem API → fuzzy JaroWinkler → manual CSV)

### Benchmark Design
- **ML:** CT-M1 binary failure prediction; CT-M2 7-way failure category (most challenging)
- **LLM:** L1 5-way MCQ (1,500 items), L2 failure report extraction (500), L3 reasoning (200), L4 discrimination (500)

### Key Results
- **CT-M1:** NegBioDB negatives trivially separable (AUROC=1.0). Control negatives reveal real difficulty (0.76–0.84).
- **CT-M2:** XGBoost best (macro-F1=0.51). Scaffold/temporal splits hardest (0.19).
- **LLM L4:** Gemini MCC=0.56 — highest across all domains. Meaningful discrimination possible for trial failure.
- **LLM L3:** Ceiling effect — GPT-4o-mini judge too lenient (4.4–5.0/5.0).

---

## PPI Domain (Protein-Protein Interaction)

Four sources: IntAct, HuRI, hu.MAP 3.0, STRING v12.0

### Database
- **2,229,670** negative results; 61,728 positive pairs (HuRI Y2H)
- 18,412 proteins; 4 split strategies: random / cold_protein / cold_both / degree_balanced

### Key Results
- **ML:** MLPFeatures (hand-crafted) dominates cold splits (AUROC 0.95 cold_both); PIPR collapses to 0.41 (below random).
- **LLM L1:** 3-shot near-perfect (0.997–1.000) is an artifact of example format leakage.
- **LLM L3:** zero-shot >> 3-shot (4.3–4.7 vs 3.1–3.7); gold reasoning examples degrade structural reasoning.
- **LLM L4:** MCC 0.33–0.44 with confirmed temporal contamination (pre-2015 acc ~0.6–0.8, post-2020 acc ~0.07–0.25).

---

## GE Domain (Gene Essentiality / DepMap)

Two sources: DepMap CRISPR (Chronos scores) and RNAi (DEMETER2)

### Database
- **28,759,256** negative results (genes with no essentiality signal)
- Final tiers: Gold 753,878 / Silver 18,608,686 / Bronze 9,396,692
- 19,554 genes × 2,132 cell lines; 22,549,910 aggregated pairs
- 5 split strategies: random / cold_gene / cold_cell_line / cold_both / degree_balanced

### Benchmark Design
- **ML:** XGBoost and MLPFeatures on gene expression + lineage features (gene-cell pair prediction)
- **LLM:** L1 4-way essentiality MCQ (1,200 items), L2 essentiality data extraction (500), L3 reasoning (200), L4 discrimination (475)

### Key Results (partial — Llama pending)
- **LLM L3:** zero-shot >> 3-shot (overall mean 4.5 vs 2.5) — same pattern as PPI.
- **LLM L4:** Expected intermediate MCC (DepMap is widely studied; likely contamination present).
- **ML:** Seed 42 complete; final aggregated results pending seeds 43/44.

---

## Dual Benchmark Framework

### LLM Benchmark Levels

| Level | Task | Difficulty | Automation |
|-------|------|-----------|------------|
| L1 | Multiple-choice classification | Easy | Fully automated (exact match) |
| L2 | Structured field extraction | Medium | Automated (JSON schema check + field F1) |
| L3 | Free-text reasoning quality | Hard | LLM-as-judge (Gemini 2.5-Flash, 4 rubric dimensions) |
| L4 | Real vs synthetic discrimination | Hard | Automated (MCC on binary decision) |

### LLM Models Evaluated

| Model | Provider | Type |
|-------|----------|------|
| Claude Haiku 4.5 | Anthropic API | Small API model |
| Gemini 2.5-Flash | Google API | Small API model |
| GPT-4o-mini | OpenAI API | Small API model |
| Qwen2.5-7B-Instruct | HuggingFace / vLLM | Open-weight local |
| Llama-3.1-8B-Instruct | HuggingFace / vLLM | Open-weight local |

### Cross-Domain LLM L4 Summary

```
DTI (≤0.18) < PPI (0.33–0.44) < CT (0.48–0.56)
                     ↑
         Increasing task complexity
         and LLM accessible signal
```

---

## Timeline

| Milestone | Date |
|-----------|------|
| Project initiated | 2026-03-02 |
| DTI domain complete (ML + LLM) | 2026-03-13 |
| CT domain initiated | 2026-03-17 |
| CT domain complete (ML + LLM) | 2026-03-20 |
| PPI domain complete (ML + LLM) | 2026-03-23 |
| GE domain ETL + ML export | 2026-03-23 |
| GE LLM (4/5 models) | 2026-03-24 |
| Public release (GitHub + HuggingFace) | 2026-03-30 |
