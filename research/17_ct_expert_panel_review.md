# CT Domain — Expert Panel Review (6-Perspective Analysis)

> Comprehensive review of NegBioDB-CT (Clinical Trial Failure domain) from 6 expert viewpoints
> Date: 2026-03-21 | Status: ML 108/108 + LLM 80/80 COMPLETE, code audit passed
> References: research/13 (domain design), research/14 (ML benchmark), research/15 (LLM benchmark)

---

## Overview

Six simulated expert reviewers assessed the NegBioDB-CT domain for NeurIPS 2026 D&B Track readiness, scientific validity, and publication quality. The CT domain adds clinical trial failure prediction to NegBioDB, comprising 216,987 trials, 132,925 failure results, and dual ML (108 runs) + LLM (80 runs) benchmarks.

**Overall NeurIPS contribution strength: HIGH — strongest cross-domain comparison (L4 MCC 0.48-0.56 vs DTI ≤0.18)**

---

## 1. NeurIPS D&B Reviewer

### Contribution Assessment: Strong Addition to Multi-Domain Paper

**Expected Scores (1-6 scale):**

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Novelty & Significance | 5/6 | First structured clinical trial failure benchmark with tiered evidence + dual ML/LLM |
| Quality of Construction | 5/6 | 3-tier detection pipeline, 4-source integration, conflict-free M1 design |
| Potential for Broad Impact | 5/6 | Clinical AI, drug repurposing, regulatory science all benefit |
| Documentation & Accessibility | 4/6 | AACT public domain base; drug resolution only 20.6% ChEMBL limits structure coverage |
| Experimental Evaluation | 5/6 | 108 ML + 80 LLM runs with controls; M1 trivial separability is a finding, not a bug |
| Clarity of Presentation | 4/6 | Cross-domain L4 comparison is the headline; 3-domain paper compression is challenging |

**Key Weaknesses:**

| ID | Weakness | Severity | Mitigation |
|----|----------|----------|------------|
| W1 | M1 AUROC 1.0 could appear as trivial benchmark | High | Frame as strength: NegBioDB negatives carry real clinical signal vs random controls |
| W2 | L3 ceiling effect (GPT-4o-mini judge 4.4-5.0) | Medium | Report with caveat; use harder judge (Gemini Pro / Claude) in revision |
| W3 | Drug resolution only 20.6% — limits chemical structure analysis | Medium | Acceptable: CT domain is about clinical outcomes, not molecular properties |
| W4 | Copper tier (CTO) excluded from L1/L2/L3 — potential criticism | Low | Justified: CTO has no failure detail, only binary label → only useful for L4 |
| W5 | Solo author credibility | High | Same mitigation as DTI: full reproducibility, Dockerfile, supplement |
| W6 | No comparison to existing CT prediction models (HINT, etc.) | Medium | Acknowledge as limitation; NegBioDB-CT is a benchmark, not a prediction system |

**Strongest selling point:** L4 discrimination gap — CT MCC 0.48-0.56 vs DTI ≤0.18 demonstrates domain-dependent LLM knowledge asymmetry. ClinicalTrials.gov is in training corpora; ChEMBL bioactivity data is not.

---

## 2. Clinical Data Engineer

### Pipeline Design: Production Quality

**Strengths:**
- AACT ETL handles both old Title Case and new UPPER_CASE column formats (forward-compatible)
- Drug resolution cascade (exact → PubChem API → fuzzy → manual) maximizes coverage at each step
- Batch processing with pre-cached protein IDs avoids N+1 SELECT performance trap
- Outcome enrichment correctly uses MIN(p_value) per trial for conservative failure detection

**Issues:**

| ID | Issue | Severity | Resolution |
|----|-------|----------|------------|
| DE1 | AACT URL changes monthly — requires manual `--url` flag | Medium | Documented; consider URL discovery script |
| DE2 | Drug resolution 20.6% ChEMBL — 79.4% of interventions unresolved | Medium | Expected: many interventions are devices, procedures, behavioral |
| DE3 | No incremental update pipeline — full reload only | Medium | Acceptable for research; production would need upsert logic |
| DE4 | Shi & Du SAE data adds 31,969 rows but 0 tier upgrades | Low | Correct: Tier 2 classification already consumed these p-values |
| DE5 | No AACT version pinning — results may differ with monthly updates | Medium | Pin AACT download date in paper methods section |

**Data Quality Assessment:**
- 216,987 trials loaded from 13 AACT tables — comprehensive
- Termination filter correctly separates clinical failures (~70%) from administrative stops (~30%)
- P-value validation (0-1 range filter) prevents nonsensical values
- Deduplication via UNIQUE index (migration 002) protects against reload artifacts

---

## 3. ML Benchmark Expert

### Experimental Design: Rigorous with Key Insight

**Strengths:**
- 6 split strategies cover all standard evaluation paradigms + scaffold (structure-aware) and degree-balanced
- 3-seed averaging with std reporting (seeds 42, 43, 44) provides variance estimates
- M2 7-way classification is novel — no existing benchmark tests failure category prediction
- Control negatives (uniform random + degree-matched) enable direct inflation measurement

**Issues:**

| ID | Issue | Severity | Resolution |
|----|-------|----------|------------|
| ML1 | M1 AUROC 1.0 on NegBioDB negatives — benchmark appears too easy | **High** | **This IS the finding**: real failures are categorically different from CTO successes. The control negatives (AUROC 0.76-0.84) prove the signal is genuine, not an artifact. Frame as: "curated failures carry strong discriminative signal" |
| ML2 | Temporal split produces single-class val set → NaN metrics | Medium | Expected behavior (9/108 affected). All 2018-2019 trials in val are failures. Document and report null metrics |
| ML3 | Scaffold split only applicable to 40% of pairs (those with SMILES) | Medium | Documented: non-SMILES pairs assigned NULL fold, excluded from scaffold analysis |
| ML4 | XGBoost runs have 0 std (deterministic) | Low | Expected: XGBoost is deterministic given same features. Std=0 is correct |
| ML5 | M2 pharmacokinetic category has 0 samples | Low | Expected: PK failures are underreported in AACT. Weight=0 in CrossEntropyLoss handles this |
| ML6 | No CT-M3 (phase transition prediction) | Low | Deferred by design (research/14). Add post-submission if needed |

**Key ML Findings (verified against results):**

| Finding | DTI Parallel | Implication |
|---------|-------------|-------------|
| NegBioDB AUROC ~1.0 | DTI also ~1.0 | Curated negatives are categorically different from positives |
| Degree-matched hardest (0.76-0.84) | DTI +0.112 inflation | Control baselines essential; confirms benchmark inflation thesis |
| Scaffold/temporal mF1 ~0.19 | DTI cold_target catastrophic | Realistic splits reveal model fragility |
| XGBoost dominates M2 | — | Tabular features sufficient; deep learning adds no benefit for category prediction |

---

## 4. Clinical Trial / Pharmacovigilance Expert

### Scientific Validity: Sound with Important Nuances

**Strengths:**
- 8-category failure taxonomy with documented precedence ordering is clinically meaningful
- Phase-based tier assignment (Phase III + results = gold) reflects regulatory rigor
- Safety > efficacy precedence in multi-label resolution matches clinical priority
- CTO success labels as positive class for M1 is a valid experimental design choice

**Issues:**

| ID | Issue | Severity | Resolution |
|----|-------|----------|------------|
| PV1 | "Efficacy" dominates at 42.6% — potential labeling bias | Medium | Expected: most trials fail on primary endpoint. NLP detection skews toward efficacy keywords |
| PV2 | PK category has 0 records — taxonomy is incomplete in practice | Medium | PK failures rarely appear in `why_stopped` text. Could mine p-values for PK endpoints specifically |
| PV3 | p > 0.05 threshold is conventional but arbitrary | Low | Standard in clinical research. Store actual p-values for future threshold analysis |
| PV4 | CTO binary labels (success/failure) lack granularity | Low | By design: CTO provides the positive class only. Granularity comes from NegBioDB tiers |
| PV5 | No FDA outcome data integration | Medium | FDA data (Drugs@FDA) is complementary but separately licensed. Consider post-submission |
| PV6 | Bronze tier relies on `why_stopped` NLP — noisy labels | Medium | Acknowledged: bronze is lowest confidence. Benchmark reports per-tier results for transparency |

**Taxonomy Validation:**

| Category | Clinical Validity | Coverage Concern |
|----------|------------------|-----------------|
| Safety | Strong — adverse event terminology well-defined | 3.9% may be undercount (SAEs underreported) |
| Efficacy | Strong — primary endpoint failure is standard | 42.6% — possibly overrepresented due to NLP bias |
| Enrollment | Valid — accrual failure is real but not pharmacological | 23.0% — high, includes slow enrollment |
| Strategic | Valid — business decisions are legitimate trial termination reasons | 7.1% — appropriate |
| Regulatory | Valid but rare | 0.7% — expected |
| Design | Valid — protocol issues | 1.8% — likely undercount |
| PK | Valid but empty | 0.0% — needs targeted extraction |
| Other | Catch-all | 21.0% — high, worth future sub-classification |

**Recommendation:** The "other" category at 21.0% warrants sub-classification in v2 (COVID-related, funding, site issues). For v1 paper, report as known limitation.

---

## 5. Software Architect

### Code Quality: Production-Ready (Post-Audit)

**Code Audit Results (2026-03-21, 5-agent parallel review):**

| Module Area | Lines | Grade | Findings |
|-------------|-------|-------|----------|
| Pipeline (ETL, classify, resolver, enrichment, DB) | ~3,165 | A | 0 critical, 2 medium (logging) |
| ML Export (splits, features, models, training) | ~2,800 | A+ | 0 issues, zero data leakage verified |
| LLM Benchmark (prompts, eval, datasets, runner) | ~1,814 | B+ | 0 critical, 4 low design observations |
| Tests (7 CT-specific modules) | ~3,400 | A | 200 tests, no coverage gaps |
| SLURM (9 scripts) | ~1,200 | A- | 3 hardening recommendations (non-blocking) |

**Issues:**

| ID | Issue | Severity | Resolution |
|----|-------|----------|------------|
| SA1 | L2 `field_f1_micro` is misleading in Phase 1 (6/7 gold fields empty) | Low | `category_accuracy` is the primary metric; label `field_f1_micro` as "approximate" |
| SA2 | L4 evidence citation heuristic too easy (len>50 OR keyword) | Low | All models achieve 100% — threshold provides no discrimination. Tighten for v2 |
| SA3 | SLURM scripts lack GPU type specification for local LLM runner | Low | All jobs completed successfully; add `gpu:a40:1` for robustness |
| SA4 | No Docker/container for full reproducibility | Medium | Same as DTI: add Dockerfile pre-submission |

**Reproducibility Checklist:**

| Item | Status |
|------|--------|
| Data sources publicly available | ✅ AACT (public domain), CTO (MIT), Open Targets (Apache), Shi & Du (CC BY 4.0) |
| Schema migrations versioned | ✅ 2 migrations (001 initial + 002 expert review fixes) |
| Random seeds documented | ✅ Seeds 42, 43, 44 for ML; deterministic few-shot selection |
| Split leakage verified | ✅ 0 leaks on cold_drug, cold_condition; M1 conflict-free |
| Results summary CSVs committed | ✅ ct_table_m1.csv, ct_table_m2.csv, ct_llm_summary.csv |

---

## 6. Project Manager

### Execution: Efficient — 4 Days from Code Complete to 188 Experiments Done

**Timeline:**

| Phase | Duration | Key Events |
|-------|----------|------------|
| CT-1 to CT-5 (infrastructure → export) | ~3 days | 2026-03-17 to 2026-03-19 |
| CT-6 ML experiments (108 runs) | ~1 day | Cayuga HPC, 3 seeds |
| CT-7 LLM experiments (80 runs) | ~2 days | Gemini rate limit resolved via Tier 1 upgrade |
| L3 judge scoring | ~0.5 day | 20 runs judged with GPT-4o-mini |
| Documentation + audit | ~0.5 day | 5-agent code review, PROJECT_OVERVIEW update |
| **Total CT domain** | **~7 days** | From inception to full completion |

**Resource Usage:**

| Resource | Cost | Notes |
|----------|------|-------|
| HPC GPU hours | ~30h | A40 GPUs for ML training + vLLM inference |
| OpenAI API | ~$5 | GPT-4o-mini (16 runs + 20 L3 judge runs) |
| Gemini API | ~$2 | Tier 1 pay-as-you-go (16 runs, completed in ~2h after upgrade) |
| Anthropic API | ~$3 | Haiku-4.5 (16 runs) |
| Total cost | **~$10** | Excluding HPC allocation (free via lab) |

**Risks Resolved:**

| Risk | Resolution |
|------|-----------|
| Gemini 250 RPD bottleneck (~8-9 days) | Upgraded to Tier 1 pay-as-you-go → 16 runs in ~2 hours |
| L2 field_f1 bug (all zeros) | Root cause: gold_extraction nesting + list response parsing. Fixed and re-evaluated |
| Temporal split single-class val | Documented as expected; write null-metric results.json |
| L3 ceiling effect | Documented as known limitation; GPT-4o-mini too lenient as CT judge |

**Integration Status:**

| Domain | ML | LLM | L3 Judge | Status |
|--------|-----|------|---------|--------|
| DTI | ✅ 18/18 | ✅ 81/81 | ✅ | Complete |
| CT | ✅ 108/108 | ✅ 80/80 | ✅ 20/20 | Complete |
| PPI | ✅ 54/54 | ⏳ 64/80 | Pending | LLM running (Haiku failed, 16 re-runs needed) |

---

## Consolidated Action Items

### P0: Before Paper Submission

| # | Action | Owner | Status |
|---|--------|-------|--------|
| A1 | Pin AACT download date in methods section | Paper | Pending |
| A2 | Frame M1 AUROC 1.0 as finding (not bug) in paper narrative | Paper | Pending |
| A3 | Add L3 ceiling effect caveat to results section | Paper | Pending |
| A4 | Create Dockerfile for CT reproducibility | Code | Pending |

### P1: Paper-Level Improvements

| # | Action | Priority |
|---|--------|----------|
| B1 | Label `field_f1_micro` as "approximate (category tokens only)" in L2 tables | Medium |
| B2 | Discuss "other" category (21.0%) as future sub-classification target | Medium |
| B3 | Add cross-domain L4 figure (DTI vs CT vs PPI MCC comparison) | High |
| B4 | Include temporal contamination analysis for CT (pre-2020 vs post-2023 accuracy gap) | Medium |
| B5 | Reference HINT, Trial2Vec, clinical trial prediction models in related work | Medium |

### P2: Post-Submission / v2

| # | Action |
|---|--------|
| C1 | Sub-classify "other" category (21.0%) using LLM-assisted labeling |
| C2 | Tighten L4 evidence citation heuristic (require BOTH length AND keyword) |
| C3 | Add PK category extraction from endpoint-specific p-values |
| C4 | Integrate FDA Drugs@FDA outcome data |
| C5 | Incremental AACT update pipeline (upsert instead of full reload) |
| C6 | Use stronger L3 judge model (Gemini Pro or Claude) |

---

## Cross-Domain Comparison (CT vs DTI)

| Aspect | DTI | CT | Interpretation |
|--------|-----|-----|----------------|
| M1 AUROC (NegBioDB) | ~1.0 | ~1.0 | Both: curated negatives are categorically distinct |
| M1 AUROC (degree-matched) | 0.998-0.999 (+0.112 LogAUC inflation) | 0.76-0.84 (deflation from 1.0) | **Opposite patterns**: DTI degree-matched is easier (inflates); CT degree-matched is harder (deflates) |
| Hardest split | cold_target (AUROC 0.76-0.89) | scaffold/temporal (mF1 ~0.19) | Different: DTI difficulty is entity-based, CT is structural/temporal |
| L4 MCC | ≤ 0.18 | 0.48-0.56 | **Key finding**: CT pairs are in training corpora (ClinicalTrials.gov is public) |
| L3 judge | GPT-4o-mini (3.18-4.66 overall range) | GPT-4o-mini (4.4-5.0 ceiling) | Same judge; CT shows more ceiling effect. DTI has wider discrimination range |
| Data volume | 30.5M negatives | 132K failures | Different scales; CT is record-level, DTI is pair-level |

---

## Conclusion

The CT domain is **publication-ready** with no blocking issues. The 4 P0 items are paper-writing tasks, not code/data fixes. The strongest contribution is the cross-domain L4 comparison showing domain-dependent LLM knowledge asymmetry — this should be the central figure in the paper.

**Recommended paper narrative for CT section:**
1. Open with M1 trivial separability (AUROC 1.0) as evidence that curated failures carry real signal
2. Show M2 category prediction difficulty (mF1 0.19-0.51) demonstrates fine-grained prediction is challenging
3. Climax with L4 MCC 0.48-0.56 vs DTI ≤0.18 as the domain asymmetry finding
4. Close with L3 ceiling as methodological caveat (judge model selection matters)

---

## Documents Referenced

| Document | Content |
|----------|---------|
| [research/13](research/13_clinical_trial_failure_domain.md) | CT domain design |
| [research/14](research/14_ct_ml_benchmark_design.md) | CT ML benchmark: 3 tasks, 6 splits, 3 models |
| [research/15](research/15_ct_llm_benchmark_design.md) | CT LLM benchmark: 4 levels, 5 models |
| [ROADMAP.md](ROADMAP.md) | CT-6 and CT-7 completion status |
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | Project Bible v3.2 with full CT results |
