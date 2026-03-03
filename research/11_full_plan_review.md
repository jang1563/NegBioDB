# Full Plan Review — Pre-Implementation Audit

> Comprehensive review of all 12 research documents + ROADMAP + PROJECT_OVERVIEW (2026-03-02)
> Conducted before any code implementation begins.

---

## Overview

All 12 research documents, ROADMAP v6, and PROJECT_OVERVIEW were read in full and cross-referenced for consistency, feasibility, and completeness.

**Overall Assessment: Plan direction and strategy are strong. Critical gaps are in execution readiness, not in design.**

---

## A. Strengths

| # | Item | Assessment |
|---|------|-----------|
| 1 | **Gap identification** | "Assumed negative" problem is clear and well-documented. No direct competitor exists |
| 2 | **License analysis** | CC BY-SA 3.0→4.0 compatibility, HCDT 2.0/InertDB workaround strategies are thorough |
| 3 | **Data volume** | NOT the bottleneck — ChEMBL ~527K + PubChem ~61M + BindingDB ~30K |
| 4 | **$0 cost strategy** | LLM free tier + Kaggle GPU + SQLite = realistic |
| 5 | **Dual-track innovation** | First ML+LLM benchmark for negative DTI — novel and timely |
| 6 | **Confidence tier system** | Gold/Silver/Bronze/Copper + assay context = scientifically sound |
| 7 | **Go/No-Go framework** | Week 6 checkpoint + 3 scenarios = excellent risk management |
| 8 | **Schema design** | SQLite DDL + 3-level structure (results → pairs → splits) is clear |
| 9 | **Croissant/Datasheet** | Templates ready, NeurIPS mandatory requirements addressed |
| 10 | **Paper narrative** | Problem-first framing ("benchmarks are broken") is compelling |

---

## B. Issues Found

### CRITICAL (must fix before implementation)

#### B1. research/08 §3 — Exp 1 run count stale

| Location | Current | Correct |
|----------|---------|---------|
| research/08 line 79 | `Exp 1 (vs random negatives): 3 (random neg)` | `Exp 1: 6 additional runs (uniform random + degree-matched random)` |

ROADMAP §Random Negative Control Design expanded Exp 1 from 3 to 9 runs (3 models × 3 negative conditions), but research/08 §3 table was not updated.

**Run count clarification (Interpretation A = 18 total):**

```
Baselines (NegBioDB negatives, 3 splits):     3 models × 3 splits = 9
Exp 1 additional (2 random conditions):        3 models × 2 = 6   (NegBioDB condition = baseline random)
Exp 4 additional (DDB split only):             3 models × 1 = 3   (random split = baseline)
Total must-have ML: 18 runs ✓
```

The NegBioDB-negative random-split runs are shared between baselines and Exp 1/Exp 4, keeping the total at 18.

**Status:** Fixed in this update (research/08 §3 table + explanatory note).

---

#### B2. "Nature MI 2025" / "Science 2025 editorial" citations unverified

| Location | Status |
|----------|--------|
| research/06 line 350 | `[to be confirmed — validates assumed negative problem]` |
| research/06 line 351 | `[to be confirmed — call for negative data inclusion]` |
| ROADMAP Finding #18 | References Nature MI 2025 as "must cite" |

These are core supporting citations for the paper narrative. If they don't exist as described, the Related Work section needs alternative citations.

**Action:** Must verify in Week 1. If not found, substitute with:
- EviDTI (Nature Comms 2025) — already validated negative evidence approach
- DDB paper (BMC Biology 2025) — node degree bias independently shows benchmark distortion
- LIT-PCBA audit (2025) — data leakage proves benchmark fragility

**Status:** Added "[UNVERIFIED]" marker to all references. Week 1 verification task added to ROADMAP.

---

#### B3. NeurIPS 2026 D&B CFP not yet officially announced

Web search confirmed: **No official announcement as of 2026-03-02.**
- NeurIPS 2026 conference: December 6-12, Sydney
- Past pattern: CFP announced ~mid-March, deadline ~mid-to-late May
- 2023: May 26, 2024: May 29, 2025: May 15

**~May 15-29, 2026 estimate is reasonable** but not confirmed.

**Risk:** If deadline is earlier (early May), plan loses 1-2 weeks.
**Mitigation:** Monitor neurips.cc weekly. CFP expected within 2-4 weeks.

**Status:** Risk noted in ROADMAP. Weekly monitoring task added.

---

### HIGH (must resolve before affected phase)

#### B4. Hardware specifications undefined

No document specifies the user's machine specs. Key requirements:
- **Llama 3.3 70B local**: ~40GB RAM (or ~24GB quantized Q4)
- **PubChem streaming**: ~4-8GB RAM during processing
- **RDKit + pandas**: ~4-8GB RAM for standardization
- **Disk**: ~50GB+ for models + ~20GB for data sources

**Decision needed:** Can the user's machine run 70B models? If not:
- Option A: Use Llama 3.1 8B + Mistral 7B (both fit in 16GB RAM)
- Option B: Use quantized Llama 3.3 70B (Q4, ~24GB)
- Option C: Use Kaggle/Colab for LLM inference too

**Status:** Added as Week 1 decision point in ROADMAP.

---

#### B5. Shared target pool size unestimated

Positive Data Protocol requires positives from "shared targets only." The intersection of:
- NegBioDB target set (from PubChem + ChEMBL + BindingDB inactive extraction)
- ChEMBL pChEMBL ≥ 6 active target set

...is unknown. If the overlap is small (< 200 targets), the benchmark shrinks significantly.

**Status:** Added as Week 2 checkpoint: "Estimate shared target pool size. If < 200, expand NegBioDB target extraction."

---

#### B6. Borderline exclusion (pChEMBL 4.5-5.5) impact unestimated

Excluding pChEMBL 4.5-5.5 from both sides could remove a significant fraction of data, especially near the active threshold (pChEMBL 5.5-6.0).

**Status:** Added as Week 2 checkpoint: "Run pChEMBL distribution query; estimate data loss from borderline exclusion."

---

#### B7. PubChem bioactivities.tsv.gz actual column names unverified

research/09 §6 code assumes columns: `'AID', 'SID', 'CID', 'Activity_Outcome', 'Activity_Value', 'Activity_Name'`. Actual PubChem FTP column names may differ.

**Status:** Added as Week 1 task: "Verify PubChem TSV column names after download."

---

### MODERATE (track and address)

#### B8. Week 1 workload too heavy

Week 1 currently includes: scaffolding + pyproject.toml + Makefile + schema + standardization pipeline + target pipeline + dedup setup + all downloads + ChEMBL verification.

For a solo author, this is 60-80 hours of work in one week.

**Resolution:** Split Week 1 into scaffolding + download only. Move standardization/dedup to Week 2.

**Status:** ROADMAP Week 1-2 restructured.

---

#### B9. L2 annotation bottleneck (100 abstracts, ~25 hours)

100 abstracts × ~15 min human correction = ~25 hours in Weeks 3-4, concurrent with data extraction and ML coding.

**Mitigation options:**
- Reduce to 50 abstracts for NeurIPS MVP (still sufficient — CaseReportBench used 200 total)
- Spread across Weeks 3-5 instead of concentrating in Weeks 3-4

**Status:** Noted as risk. No ROADMAP change (user decision on scope).

---

#### B10. Makefile targets not defined

ROADMAP mentions "Makefile pipeline" but no specific targets are defined anywhere.

**Status:** Will be defined during Week 1 scaffolding implementation.

---

#### B11. Kaggle GPU type/compatibility unverified

Kaggle free tier provides T4 (16GB VRAM) or P100 (16GB). DrugBAN with bilinear attention may need batch size tuning for 16GB.

**Status:** Added as Week 5 pre-check: "Test DeepDTA on Kaggle before full baseline run."

---

#### B12. "10-20% inflation" claim is pre-experimental

research/08 §7 opening hook specifies "10-20%." This is aspirational, not measured.

**Status:** Go/No-Go framework at Week 6 handles this. Added note to research/08 §7: "10-20% is hypothesized; actual value from Exp 1."

---

### MINOR (reference only)

| # | Item |
|---|------|
| B13 | Murcko scaffold column not in initial DDL — needs migration 002 before scaffold split |
| B14 | `result_type = 'hypothesis_negative'` rarely applies to database-sourced data |
| B15 | DAVIS Kd → pKd conversion consistent with NegBioDB threshold (both use 10 uM = pKd/pChEMBL 5) |
| B16 | research/01 and research/02 have overlapping content on DAVIS, LIT-PCBA (consolidate in paper) |

---

## C. Cross-Document Inconsistencies

| # | Inconsistency | Location | Resolution |
|---|---------------|----------|------------|
| C1 | Exp 1 = "3 runs" (old) | research/08 §3 line 79 | → Update to "6 additional" with explanatory note |
| C2 | "54 GPU-hours" vs "36-72 hours" | ROADMAP #16 vs research/08 line 82 | → Use "36-72 hours" range; remove single-point estimate |
| C3 | Exp 4 = "6 runs" vs "DDB vs random, 3 models" | research/08 vs ROADMAP line 246 | → Clarify: 3 new DDB runs (random-split shared with baselines) |

---

## D. Feasibility Ratings

| Area | Grade | Rationale |
|------|-------|-----------|
| **Data acquisition** | **A** | Ample public sources, FTP bulk confirmed |
| **Schema/pipeline design** | **A-** | DDL complete, export patterns clear. Makefile content missing |
| **ML benchmark design** | **A-** | 7 splits, 7 metrics, DDB included. Positive protocol defined |
| **LLM benchmark design** | **A** | 6 tasks, world-first, automated eval + Judge separation |
| **Execution timeline** | **B+** | ~22-day buffer. Week 1 overloaded, solo author risk |
| **Paper structure** | **A-** | Problem-first narrative, 9p budget, 5 figures planned |
| **Documentation completeness** | **B+** | 12 comprehensive docs. Some stale data (B1), unverified citations (B2) |
| **Code readiness** | **F** | **Zero lines of code.** All design documents. Pure implementation from Week 1 |

---

## E. Recommended Execution Adjustments

### Week 1 (revised): Scaffolding + Download ONLY
1. Project scaffolding (directory structure, pyproject.toml, Makefile skeleton, config.yaml)
2. Data download (ChEMBL SQLite, PubChem FTP, BindingDB TSV, DAVIS via TDC)
3. **Verify PubChem TSV column names** (B7)
4. **Verify Nature MI 2025 / Science 2025 citations** (B2)
5. **Start NeurIPS 2026 CFP monitoring** (B3)
6. **Decide hardware/LLM model tier** (B4)
7. Apply SQLite initial schema (migration 001)

### Week 2 (revised): Standardization + Extraction Start
8. Compound standardization pipeline (RDKit)
9. Target standardization pipeline (UniProt)
10. Cross-DB deduplication framework
11. **Check shared target pool size** (B5)
12. **Check borderline exclusion impact** (B6)
13. Begin ChEMBL + PubChem extraction

### Week 3+ : Continue as planned (with run count fixes)

---

## F. Documents Updated by This Review

| Document | Changes |
|----------|---------|
| research/08 | §3 Exp 1 run count fixed; §7 "10-20%" marked as hypothesis |
| ROADMAP.md | Week 1-2 restructured; hardware/verification checkpoints added; GPU estimate updated |
| research/06 | "[UNVERIFIED]" markers on Nature MI 2025 and Science 2025 citations |
| PROJECT_OVERVIEW.md | Added research/11 |

---

## Sources

- NeurIPS 2026 organizer nominations: https://blog.neurips.cc/2026/01/07/neurips-2026-call-for-organizer-nominations/
- NeurIPS 2025 D&B retrospective: https://blog.neurips.cc/2025/12/05/neurips-datasets-benchmarks-track-from-art-to-science-in-ai-evaluations/
- NeurIPS 2025 D&B Call: https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks
- Llama 3.3 70B requirements: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
