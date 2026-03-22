# NegBioDB-PPI: Expert Panel Review

> 6-Expert Professional Review | 2026-03-21 | Domain: Protein-Protein Interaction
> Scope: ETL pipeline, ML benchmark, LLM benchmark, infrastructure, results

---

## Panel Composition

| Role | Focus Area |
|------|-----------|
| **R1: NeurIPS D&B Reviewer** | Publication readiness, novelty, experimental rigor |
| **R2: Data Engineering Architect** | Pipeline design, scalability, data integrity |
| **R3: ML Benchmark Expert** | Experimental design, metrics, statistical validity |
| **R4: PPI Domain Expert** | Biological validity, data source quality, interpretation |
| **R5: Software Architect** | Code quality, testing, maintainability |
| **R6: Project Manager** | Scope, risk, timeline, resource allocation |

---

## R1: NeurIPS D&B Reviewer

### Strengths
- **Three-domain scope is novel:** No existing benchmark systematically tests negative results across DTI, CT, and PPI. This breadth strengthens the contribution.
- **Contamination finding is publication-worthy:** PPI-L4 contamination gap (0.46-0.59) is the strongest evidence of LLM memorization vs reasoning across all three domains. This is a genuine contribution to the LLM evaluation literature.
- **Cross-domain gradient is compelling:** DTI (opaque, MCC ≤ 0.18) < PPI (memorized, MCC ~0.4) < CT (partially reasoned, MCC ~0.5) tells a coherent story about LLM knowledge sources.

### Issues

| ID | Issue | Severity | Recommendation |
|----|-------|----------|----------------|
| R1-1 | **L1 3-shot trivially solvable (~1.0 acc)** — evidence descriptions with examples make this a pattern-matching task, not a reasoning test. Undermines L1 as a benchmark level. | HIGH | Report as finding: "Evidence classification is solvable with examples, suggesting LLMs can pattern-match evidence types but not reason about them." Consider adding L1 without evidence descriptions (protein names only) as ablation. |
| R1-2 | **L2 near-perfect extraction (entity_f1 ~1.0)** — constructed evidence is too clean. Real-world extraction from abstracts would be harder and more scientifically interesting. CT-L2 field_f1 was 0.48-0.81, showing genuine difficulty. | HIGH | Frame as deliberate design: "Constructed evidence provides ceiling performance; real-world extraction remains untested." Acknowledge limitation in paper. Compare to CT-L2 to show domain contrast. |
| R1-3 | **L3 judge ceiling risk** — CT-L3 already showed GPT-4o-mini judge gives 4.4-5.0/5.0. PPI-L3 may repeat this problem with the same judge. | MEDIUM | Use stricter judge prompt with explicit examples of low scores. Consider Gemini Flash-Lite as judge (DTI-L3 showed more discrimination, range 1.96-4.66). |
| R1-4 | **16 Haiku runs missing** — incomplete 5th model weakens cross-domain comparison. All three domains should have the same 5 models for fair comparison. | MEDIUM | Must re-run after credit top-up. If infeasible, report 4-model results with explicit note. |
| R1-5 | **Contamination gap interpretation needs caution** — pre-2015 vs post-2020 also correlates with protein study completeness (well-studied proteins more likely to be tested earlier). Cannot fully separate memorization from reasoning about protein popularity. | MEDIUM | Add control analysis: compare accuracy on well-studied vs poorly-studied proteins within the same temporal group. If accuracy correlates with protein popularity, confound exists. |

### Publication Readiness: **7/10** (with caveats)
- L1 and L2 ceiling effects weaken 2 of 4 benchmark levels
- L4 contamination finding is the strongest contribution
- Needs 5th model (Haiku) and L3 judge completion

---

## R2: Data Engineering Architect

### Strengths
- **Reservoir sampling in HuRI and STRING** — elegant solution for 39.9M and >100M candidate spaces. Avoids OOM while maintaining uniform sampling.
- **Pre-caching optimization in HuRI ETL** — reduces protein ID lookups from millions of individual SELECTs to one batch query.
- **Canonical pair ordering enforced at DB level** — `CHECK (protein1_id < protein2_id)` prevents duplicate pairs in both directions.
- **Batch inserts (10K)** — consistent across all 4 ETL modules.

### Issues

| ID | Issue | Severity | Recommendation |
|----|-------|----------|----------------|
| R2-1 | **Race condition in `protein_mapper.py:get_or_insert_protein()`** — SELECT then INSERT without atomic UPSERT. Parallel ETL workers could create duplicate proteins. | HIGH | Replace with `INSERT OR IGNORE` + `SELECT` pattern, or use SQLite UPSERT. Not critical for single-worker execution, but blocks any future parallelization. |
| R2-2 | **Brittle file detection in `etl_string.py`** — `glob("*.uniprot_2_string*")` takes first match. Multiple STRING versions in same directory → non-deterministic behavior. | HIGH | Accept explicit file paths as parameters. Add warning if >1 match found. |
| R2-3 | **Hardcoded file dates in hu.MAP** — `ComplexPortal_reduced_20230309.*` baked into code. Data update requires code change. | MEDIUM | Accept version parameter or auto-detect from directory listing with user confirmation. |
| R2-4 | **DROP TABLE without transaction in `ppi_db.py`** — `DROP TABLE IF EXISTS _pdeg` outside transaction context. Process crash leaves temp table. | LOW | Wrap aggregation in explicit transaction. Not impactful since `_pdeg` is recreated on next run. |
| R2-5 | **No FK from `ppi_negative_results.pubmed_id` to `ppi_publication_abstracts.pmid`** — referential integrity not enforced at DB level. | LOW | Add FK constraint in future migration. Not critical since both tables are populated by controlled scripts. |

### Pipeline Quality: **A-**
- 4 independent ETL modules with clear tier assignment
- Idempotent design (DELETE-before-INSERT in dataset_versions)
- Minor race condition and file detection issues

---

## R3: ML Benchmark Expert

### Strengths
- **Degree leakage caught and fixed** — DB negatives had degree values, positives/controls had None. MLPFeatures AUROC=1.0 was the smoking gun. Fix: recompute degree from merged graph in `PPIDataset.__init__`. Exemplary bug detection.
- **4 split strategies** including Metis graph partitioning for cold_both — more sophisticated than DTI (3 splits) or CT (6 simpler splits).
- **Model-dependent negative source effect** is a genuine finding — sequence models show same inflation as DTI (+6-9%), but MLPFeatures shows reversed effect (-5% to -19%). This is novel.

### Issues

| ID | Issue | Severity | Recommendation |
|----|-------|----------|----------------|
| R3-1 | **Cold_both extreme imbalance** — Metis partitioning produces test set with only 1.7% positive (242/14,037). This is realistic for PPI but makes metrics unstable (MCC can swing with a few predictions). | HIGH | Report imbalance explicitly. Add bootstrapped confidence intervals for cold_both metrics. Consider reporting PR-AUC as primary metric for imbalanced splits instead of AUROC. |
| R3-2 | **MLPFeatures degree normalization arbitrary** — divides by 1000. If proteins have degree >> 1000, features are clipped; if << 1000, wasted dynamic range. | MEDIUM | Compute normalization from data statistics (mean/std or percentile). Not likely to change results significantly but is a methodological weakness if challenged by reviewers. |
| R3-3 | **No cross-seed significance tests** — results report mean±std across 3 seeds but no p-values or confidence intervals. Reviewers may question whether differences are significant. | MEDIUM | Add paired Wilcoxon signed-rank test for key comparisons (NegBioDB vs control negatives, cold_both vs random). Three seeds is minimum; consider 5 seeds for the paper. |
| R3-4 | **DDB staleness detection** — `collect_results.py` compares result mtime to parquet mtime to detect stale DDB results. Smart, but needs explicit warning when stale results are included. | LOW | Already handled with `--allow-stale-ddb` flag. Document in paper that DDB regeneration triggers re-training. |
| R3-5 | **L1 difficulty gradient shows direct_experimental=0% accuracy** — all models predict 0% on IntAct class in zero-shot, but 100% in 3-shot. This suggests the difficulty is entirely in label space recognition, not biological understanding. | MEDIUM | Investigate whether zero-shot 75% accuracy (3/4 classes) is because models default to categories B/C/D and never predict A. If so, L1 zero-shot measures label bias, not knowledge. Report per-class accuracy in paper. |

### Experimental Rigor: **B+**
- Degree leakage fix is exemplary
- Cold_both imbalance and 3-seed limitation are real concerns
- L1 ceiling diminishes that benchmark level

---

## R4: PPI Domain Expert

### Strengths
- **4 complementary data sources** with clear tier justification:
  - IntAct: Curated negative interactions from published experiments (gold/silver)
  - HuRI: Systematic Y2H screen — largest and most rigorous negative PPI source
  - hu.MAP: ML-derived from co-fractionation — orthogonal evidence type
  - STRING: Zero-score pairs — weakest evidence but from well-studied proteins
- **HuRI negatives are scientifically sound** — Y2H systematic screen captures true non-interactions (proteins were tested and found not to interact, not just untested).
- **MLPFeatures generalization finding is biologically meaningful** — hand-crafted features (subcellular location, degree, AA composition) capture orthogonal biological signals that sequence-only models miss.

### Issues

| ID | Issue | Severity | Recommendation |
|----|-------|----------|----------------|
| R4-1 | **Compartment extraction heuristic for L4** — keyword-based matching ("nucleus" in loc → nucleus) misclassifies complex annotations (e.g., "nuclear envelope" should be membrane-adjacent, not nuclear). | HIGH | Use UniProt controlled vocabulary (SL-xxxx terms) with systematic mapping. At minimum, validate heuristic against UniProt's official subcellular_location CV. |
| R4-2 | **hu.MAP negatives are ML predictions, not experimental** — trained on ComplexPortal positives, predicts non-complex pairs. This is one step removed from experimental evidence but ranked as Silver tier. | MEDIUM | Acknowledge in paper that hu.MAP Silver is "computationally inferred non-interactions" not "experimentally confirmed." Consider renaming to Bronze+ or documenting the distinction. The tier is defensible (ML trained on curated data) but should be explicitly discussed. |
| R4-3 | **STRING zero-score ≠ experimental negative** — zero combined score means "no evidence of interaction" not "confirmed non-interaction." Could include genuinely interacting pairs that haven't been studied. | MEDIUM | Already Bronze tier (appropriate). Emphasize in paper: "STRING negatives represent absence of evidence, not evidence of absence." This is the key difference from IntAct/HuRI negatives. |
| R4-4 | **L4 contamination confounded with protein study depth** — pre-2015 tested pairs involve well-studied proteins (more literature → more training data AND more experimental coverage). Cannot fully separate memorization from popularity bias. | HIGH | Add analysis: stratify L4 accuracy by protein degree/study depth within temporal groups. If high-degree proteins are more accurately classified regardless of year, it's popularity bias not contamination. Both interpretations are interesting but must be distinguished. |
| R4-5 | **L3 same-compartment balance** — 50/50 same/different-compartment is arbitrary. Real non-interactions are skewed toward different compartments (most untested pairs are cross-compartment). | LOW | Report as a design choice. Include compartment-stratified L3 judge scores to test whether LLMs give generic "different compartment" reasoning vs specific mechanistic arguments. |

### Domain Validity: **A-**
- Data sources are well-chosen and appropriately tiered
- hu.MAP/STRING tier labeling could be more explicit about evidence type
- Contamination-popularity confound needs explicit analysis

---

## R5: Software Architect

### Strengths
- **Excellent test coverage** — 386 tests (176 pipeline + 109 ML + 101 LLM). 3,045 test lines across 8 test modules.
- **Clean module separation** — ETL, DB, export, models, LLM prompts/eval/dataset as independent modules.
- **Resume support in LLM runner and judge** — predictions.jsonl append-only with completed_ids tracking. Robust against interruption.
- **NumPy 2.x compatibility** — `frombuffer`/`tolist` pattern instead of `from_numpy`/`.numpy()`.

### Issues

| ID | Issue | Severity | Recommendation |
|----|-------|----------|----------------|
| R5-1 | **Run name parsing fragile** — `collect_ppi_llm_results.py` uses `rsplit("_fs", 1)` to extract fewshot set. Model names containing "_fs" would break parsing. | MEDIUM | Use regex: `re.match(r'(ppi-l\d)_(.*?)_(zero-shot\|3-shot)_fs(\d)', name)`. More robust and self-documenting. |
| R5-2 | **No integration tests** — unit tests are excellent but no end-to-end pipeline test (ETL → split → export → train → evaluate). | MEDIUM | Add one integration test that runs a minimal pipeline (10 pairs, 1 epoch, 1 split) to catch interface mismatches. |
| R5-3 | **Mixed logging** — some modules use `logging`, others use `print()`. Inconsistent with CT domain (which uses logging throughout). | LOW | Standardize to `logging` module. Low priority since scripts work correctly. |
| R5-4 | **No schema validation in LLM runner** — dataset JSONL records not validated against expected schema before inference. Missing fields cause mid-run failures. | MEDIUM | Add upfront schema check: verify required fields (question_id, context_text/evidence_text, gold_answer) exist in first record. |
| R5-5 | **SLURM template validation missing** — `submit_ppi_llm_all.sh` doesn't check if SLURM templates exist before submission. Also doesn't capture submitted job IDs. | LOW | Add `[ -f "$SLURM_TMPL" ]` check. Capture `sbatch` output to job log. |

### Code Quality Grade: **A-**
- Professional-grade test coverage
- Clean modular architecture
- Minor robustness improvements needed

---

## R6: Project Manager

### Strengths
- **PPI completed in 3 days** (initiated 2026-03-18, ML done 2026-03-20, LLM submitted 2026-03-21). Extraordinary velocity leveraging DTI/CT patterns.
- **Parallel execution** — ML and LLM pipelines share no dependencies after export. Correct parallelization.
- **Budget achieved** — $0 infrastructure cost (Cayuga HPC, free/low-cost API tiers). Only Anthropic credit exhaustion caused issues.

### Issues

| ID | Issue | Severity | Recommendation |
|----|-------|----------|----------------|
| R6-1 | **Haiku credit exhaustion blocks completion** — 16/80 runs failed. Cannot complete cross-domain comparison without 5th model. Re-run cost: ~$5-10 for 16 API runs. | HIGH | Top up Anthropic credits immediately. Delete failed predictions.jsonl files (they contain ERROR entries that break resume logic), then re-submit. Estimated time: 2-4 hours after credit top-up. |
| R6-2 | **L3 judge not yet run** — 20 judge scoring runs needed (4 models × 4 configs + potential Haiku). Requires OpenAI API credits (GPT-4o-mini). | HIGH | Schedule L3 judge immediately after Haiku re-run. Budget: ~$2-5 for 20 runs × 200 records. |
| R6-3 | **Results collection not yet run** — `collect_ppi_llm_results.py` needs to be executed locally after transferring results from Cayuga. | MEDIUM | Transfer results: `scp -r cayuga:results/ppi_llm/ results/ppi_llm/`. Then run collector. Estimated time: 30 minutes. |
| R6-4 | **Paper writing not started** — all three domains now have results but no paper draft exists. This is the critical path item. | HIGH | Begin paper outline immediately. Structure: Intro (problem) → Database (3 domains) → Benchmark Design → Experiments (ML + LLM per domain) → Cross-Domain Analysis → Discussion. Target: 9 pages + appendix. |
| R6-5 | **Memory files growing beyond limits** — MEMORY.md is 209 lines (limit: 200). experiment_results.md continues growing. Risk of context truncation in future sessions. | MEDIUM | Refactor MEMORY.md: move detailed results to experiment_results.md, keep only current status and key findings in MEMORY.md. Target: MEMORY.md ≤ 150 lines. |

### Project Status: **On Track (with 3 blockers)**
1. Haiku re-run (2-4 hours after credit top-up)
2. L3 judge scoring (2-4 hours)
3. Results collection (30 minutes)

---

## Consolidated Findings

### By Severity

| Severity | Count | Key Items |
|----------|-------|-----------|
| **CRITICAL** | 0 | No show-stoppers |
| **HIGH** | 8 | R1-1 (L1 ceiling), R1-2 (L2 ceiling), R2-1 (race condition), R3-1 (cold_both imbalance), R4-1 (compartment heuristic), R4-4 (contamination confound), R6-1 (Haiku credits), R6-4 (paper not started) |
| **MEDIUM** | 12 | R1-3, R1-4, R1-5, R2-3, R3-2, R3-3, R3-5, R4-2, R4-3, R5-1, R5-2, R5-4 |
| **LOW** | 7 | R2-4, R2-5, R3-4, R4-5, R5-3, R5-5, R6-5 |

### By Action Required

**Must Fix Before Paper Submission:**
1. Re-run 16 Haiku jobs (R6-1)
2. Run L3 judge scoring (R6-2)
3. Run results collection (R6-3)
4. Add per-class L1 accuracy to results (R3-5)
5. Add contamination vs popularity analysis for L4 (R4-4, R1-5)

**Should Address in Paper:**
6. Frame L1/L2 ceilings as findings, not bugs (R1-1, R1-2)
7. Discuss hu.MAP/STRING evidence types explicitly (R4-2, R4-3)
8. Report cold_both imbalance with bootstrapped CIs (R3-1)
9. Add cross-seed significance tests (R3-3)

**Code Improvements (Not Blocking):**
10. Fix race condition in protein_mapper (R2-1)
11. Replace STRING file glob with explicit paths (R2-2)
12. Add regex-based run name parsing (R5-1)
13. Add integration test (R5-2)

---

## Overall Assessment

| Dimension | Grade | Notes |
|-----------|-------|-------|
| **Data Quality** | A | 4 complementary sources, appropriate tier assignment, canonical ordering enforced |
| **ML Benchmark** | A- | Degree leakage caught and fixed; cold_both imbalance documented; 3 seeds minimum |
| **LLM Benchmark** | B+ | L4 contamination finding is excellent; L1/L2 ceiling effects weaken 2/4 levels |
| **Code Quality** | A- | 386 tests, clean modules, resume support; minor robustness gaps |
| **Experimental Rigor** | B+ | Need significance tests, popularity confound analysis, 5th model |
| **Pipeline Integrity** | A | No data leakage, idempotent ETL, conflict removal verified |
| **Publication Readiness** | 7/10 | 3 blockers (Haiku, L3 judge, results collection); L1/L2 ceilings need framing |

**Bottom Line:** The PPI domain is well-engineered with a genuine scientific contribution (L4 contamination finding, model-dependent negative source effect). The L1/L2 ceiling effects should be framed as findings (evidence classification is easy; discrimination is hard) rather than benchmark failures. The primary risk is the contamination-popularity confound in L4, which needs explicit analysis before submission.

**Estimated effort to address HIGH items:** 1-2 sessions (code fixes + re-runs + analysis).
