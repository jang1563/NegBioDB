# Expert Panel Review — 6-Perspective Analysis

> Comprehensive review of NegBioDB project from 6 expert viewpoints (2026-03-02)
> Findings incorporated into ROADMAP v6, research/08 §15-17, research/05, research/06, research/09

---

## Overview

Six simulated expert reviewers assessed the full NegBioDB project (10 documents) for NeurIPS 2026 D&B Track readiness, extensibility, and execution feasibility.

**Overall NeurIPS acceptance probability: 60-70%**

---

## 1. NeurIPS D&B Reviewer

### Acceptance Probability: 60-70%

**Expected Scores (1-6 scale):**

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Novelty & Significance | 5/6 | First curated negative DTI benchmark + dual ML/LLM |
| Quality of Construction | 4/6 | Confidence tiers + standardization solid; 10K entries is modest |
| Potential for Broad Impact | 5/6 | Every DTI paper needs negatives. LLM community reach |
| Documentation & Accessibility | 4-5/6 | Croissant + Datasheet + Python API. HuggingFace = 5 |
| Experimental Evaluation | 3-4/6 | 4 must-have experiments sufficient but thin without should-haves |
| Clarity of Presentation | 4/6 | Problem-first narrative strong; 9-page dual-track compression is challenge |

**Key Weaknesses:**

| ID | Weakness | Severity | Mitigation |
|----|----------|----------|------------|
| W1 | Solo author — credibility concern | High | Dockerfile, full reproducibility, detailed supplement |
| W2 | "Just aggregation" attack | High | Exp 1+4 as opening hook proves benchmarks are broken |
| W3 | LLM benchmark small (L2: 100 abstracts) | Medium | ChemBench also 2,700 MCQ; frame as annotation pipeline demo |
| W4 | 9 pages for dual-track content | Medium-High | Appendix for schema/metrics/details |
| W5 | Exp 1 "10-20% inflation" unverified | High | Week 6 Go/No-Go decision framework |
| W6 | Croissant desk rejection risk | Critical | Template in research/09; validate with mlcroissant Week 7 |

**Action:** Co-author search (+10-15% acceptance rate). Exp 7 (target class coverage) for strong figure.

---

## 2. Data Engineering Architect

### Pipeline Design: Solid with Critical Memory Issue

**Strengths:** FTP bulk decision correct. SQLite for portability. chembl_downloader setup.

**Issues:**

| ID | Issue | Severity | Solution |
|----|-------|----------|----------|
| DE1 | PubChem 12GB OOM risk | Critical | Streaming: chunksize=100K or polars lazy. Code in research/09 §6 |
| DE2 | SID→CID mapping join cost | High | SQLite temp table with indexed join |
| DE3 | compound_target_pairs GROUP BY slow | Medium | Batch INSERT + temp indexes |
| DE4 | RDKit standardization speed | Medium | Multiprocessing + InChIKey cache |
| DE5 | Parquet target_sequence inefficient | Low | Separate file or dictionary encoding |

**Extensibility:** Common/Domain layer separation excellent for future domains.

**Action:** Makefile pipeline with dependency graph (Week 1 priority).

---

## 3. ML Benchmark Expert

### Experimental Design: Strong Core, Critical Gap in Positive Data

**Strengths:** LogAUC primary metric, DDB split, 7-metric reporting, cold splits.

**Critical Issues:**

| ID | Issue | Severity | Resolution |
|----|-------|----------|------------|
| ML1 | Positive data source undefined | **Critical** | ChEMBL pChEMBL ≥ 6, shared targets. See ROADMAP §Positive Data Protocol |
| ML2 | Random negative not precisely defined | **Critical** | Uniform + degree-matched controls. See ROADMAP §Random Negative Control |
| ML3 | Class imbalance strategy undefined | High | Report balanced (1:1) + realistic (1:10) |
| ML4 | n=3 runs statistically weak | Medium | Accept for sprint; add Bootstrap CI |
| ML5 | Baseline hyperparameters undefined | Medium | Use original paper defaults; document in appendix |
| ML6 | DDB implementation unverified | Medium | Degree distribution visualization after implementation |
| LM1 | Exp 9 modality fairness (SMILES vs NL) | High | Acknowledge in paper; analyze as feature |
| LM2 | L4 Gemini training cutoff unknown | Medium | Pre/post-2024 performance gap as proxy |
| LM3 | 3-shot example selection bias | Medium | 3 different few-shot sets; report variance |

**Action:** A1 (positive protocol) and A2 (random negative) are P0 — both now in ROADMAP v6.

---

## 4. Medicinal Chemistry / Bioinformatics Domain Expert

### Scientific Validity: Sound with Important Caveats

**Strengths:** 10 uM threshold standard, PAINS flagging (not removal), BAO ontology.

**Issues:**

| ID | Issue | Severity | Resolution |
|----|-------|----------|------------|
| DC1 | "Inactive" is assay-context-dependent | High | Prominent caveat in paper + inactivity_caveat field |
| DC2 | Target family bias (kinase/GPCR heavy) | High | Exp 7 quantifies this; state as known limitation |
| DC3 | Activity cliff at threshold boundary | Medium | Store actual values; borderline zone excluded |
| DC4 | Selectivity panel data underused | Medium | Prioritize MLPCN selectivity assays |
| DC5 | ChEMBL v36 not v35 | Medium | Updated in research/05 and ROADMAP |
| DC6 | Gene vs protein level target | Low | Filter target_type = SINGLE PROTEIN |

**Extensibility Ranking:**
1. Gene Function (DepMap/CRISPR) — **best next domain** (DepMap public, targets table reusable)
2. Clinical Trial Failure (ClinicalTrials.gov) — easy data access
3. Chemistry (failed reactions) — medium difficulty
4. Materials Science — lowest priority

**Action:** Nature MI 2025 + Science 2025 editorial citations added to research/06.

---

## 5. Software Architect

### Code Structure: Needs Scaffolding (No Code Exists Yet)

**Strengths:** Schema DDL complete, export patterns clear, TDC-compatible API designed.

**Critical Issues:**

| ID | Issue | Severity | Resolution |
|----|-------|----------|------------|
| SA1 | No project code structure | Critical | Scaffolding defined in research/08 §17 |
| SA2 | No dependency management | High | pyproject.toml with pinned deps |
| SA3 | No test strategy | High | 3 minimum test files (standardize, dedup, export) |
| SA4 | Configuration hardcoding risk | Medium | config.yaml for all parameters |
| SA5 | No logging/monitoring | Medium | tqdm + Python logging |
| SA6 | No Docker | High (for review) | Dockerfile in Week 8-9 |

**Recommended Structure:** `src/negbiodb/` package with `standardize/`, `extract/`, `curate/`, `benchmark/`, `export/` modules. See research/08 §17 for full layout.

**Action:** A3 (scaffolding) + A4 (pyproject.toml) are Week 1 P0 priorities.

---

## 6. Project Manager

### Timeline: Feasible with ~22-Day Buffer

**Critical Path:** Download → Standardize → Dedup → Split → ML Baselines → Exp 1 → Paper = ~35 days
**Available:** ~77 days (11 weeks) → **~42 days buffer** if critical path holds.

**Top Risks:**

| ID | Risk | Probability | Impact | Mitigation |
|----|------|------------|--------|------------|
| PM1 | GPU unavailability | High | Critical | Kaggle free (30h/wk); Colab Pro fallback |
| PM2 | L2 annotation bottleneck | High | High | LLM first-pass saves 40%; dedicated Week 3-4 slot |
| PM3 | ML baseline environment setup | Medium | High | PyTDC example code as fallback |
| PM4 | Exp 1 weak results | Medium | Critical | Week 6 Go/No-Go framework (3 scenarios) |
| PM5 | 9-page content overflow | High | Medium | Figure plan pre-confirmed; appendix heavy |
| PM6 | Croissant validation failure | Low | Critical | mlcroissant Week 7; HuggingFace auto-gen fallback |

**Go/No-Go Framework:** See ROADMAP v6 §Week 6 Go/No-Go Decision Framework.

---

## Consolidated Action Items

### P0: Immediate (Week 1)

| # | Action | Status |
|---|--------|--------|
| A1 | Positive data protocol defined | ✅ ROADMAP v6 |
| A2 | Random negative control design | ✅ ROADMAP v6 |
| A3 | Project code scaffolding | ✅ research/08 §17 |
| A4 | pyproject.toml + dependency management | ✅ research/08 §17 |
| A5 | PubChem streaming processing | ✅ research/09 §6 |
| A6 | ChEMBL v36 confirmation | ✅ research/05 updated |

### P1: Core Strengthening (Week 1-4)

| # | Action | Status |
|---|--------|--------|
| B1 | GPU strategy (Kaggle free) | ✅ ROADMAP v6 finding #16 |
| B2 | Makefile pipeline | Defined; implement Week 1 |
| B3 | Exp 7 (target class coverage) | Add to sprint; analysis only |
| B4 | Dockerfile | Week 8-9 |
| B5 | Nature MI 2025 + Science 2025 citations | ⚠️ Added to research/06 but **[UNVERIFIED]** — must confirm in Week 1 |
| B6 | 9-page paper structure | ✅ research/06 §9 |
| B7 | Test code (3 minimum) | Implement Week 2-3 |
| B8 | Class imbalance (1:1 + 1:10) | ✅ ROADMAP v6 |
| B9 | Co-author search | User decision pending |

### P2: If Time Permits

| # | Action |
|---|--------|
| C1 | Borderline zone analysis (pChEMBL 4.5-5.5) |
| C2 | Multiple few-shot set validation |
| C3 | Bootstrap CI for 3-run experiments |
| C4 | Selectivity panel data prioritization |
| C5 | Parquet target_sequence optimization |

---

## Documents Updated

| Document | Changes |
|----------|---------|
| ROADMAP.md | v5→v6: Positive Data Protocol, Random Negative Control, Go/No-Go framework, expanded risks, project scaffolding tasks, 9 pages, GPU strategy, ChEMBL v36 |
| research/05 | ChEMBL v35→v36, positive data extraction SQL added |
| research/06 | 9-page structure plan, NeurIPS 2025/2026 requirements updated, key citations added |
| research/08 | §15 (Positive Data Protocol), §16 (Random Negative Control), §17 (Project Structure), expanded summary table |
| research/09 | §6 (PubChem Streaming Pattern with code) |
| PROJECT_OVERVIEW.md | Added research/10, updated ROADMAP version reference |

---

## Sources

- NeurIPS 2025 D&B Call: https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks
- NeurIPS 2025 Raising the Bar: https://blog.neurips.cc/2025/03/10/neurips-datasets-benchmarks-raising-the-bar-for-dataset-submissions/
- Croissant specification: https://docs.mlcommons.org/croissant/docs/croissant-spec.html
- DDB (BMC Biology 2025): https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-025-02231-w
- LIT-PCBA audit (2025): https://arxiv.org/html/2507.21404v1
- Nature MI 2025 (assumed negative validation): [DOI to be confirmed]
- Science 2025 editorial (negative results in AI): [DOI to be confirmed]
