# Expert Review Responses & Feasibility Analysis

> Concrete analysis, calculations, and decisions based on multi-expert review (2026-03-02)

---

## 1. Data Volume: 5K vs 10K vs 20K (P0)

### Raw Data Available (Verified)

| Source | Raw Records | After Quality Filter | Unique Compound-Target Pairs (est.) |
|--------|------------|---------------------|--------------------------------------|
| **ChEMBL (pchembl < 5)** | ~1.36M | ~527K (IC50/Ki/Kd/EC50, valid) | ~100-200K |
| **ChEMBL (activity_comment)** | ~763K (literature-curated) | ~300K (after dedup with above) | ~50-100K |
| **PubChem confirmatory inactive** | ~78.9M SID records | ~61M (with target annotations) | ~5-10M (est.) |
| **BindingDB (Kd/Ki > 10 uM)** | ~30K+ | ~25K (human targets) | ~20K |
| **DAVIS (pKd ≤ 5)** | ~27K | ~27K (complete matrix) | ~27K |

**Conclusion: Data volume is NOT the bottleneck.** After deduplication across sources, conservative estimate is **200K+ unique inactive compound-target pairs** from license-safe sources alone.

### Decision

| Target | Minimum | Stretch | Rationale |
|--------|---------|---------|-----------|
| **NeurIPS submission** | **10K curated** | **20K+** | 5K is insufficient for "large-scale" claim. 10K is achievable in 1-2 weeks of automated extraction + tier assignment. 20K+ with additional manual QC on Gold tier. |
| **Camera-ready** | 50K | 100K+ | Full extraction pipeline complete |

### Tier Distribution Target (10K minimum)

| Tier | Target Count | Source |
|------|-------------|--------|
| Gold | 500-1,000 | DAVIS matrix + multi-assay confirmed |
| Silver | 3,000-5,000 | ChEMBL dose-response + PubChem confirmatory |
| Bronze | 5,000-10,000 | PubChem primary screen inactive + ChEMBL pchembl < 5 |
| Copper | 1,000-5,000 | Text-mined (if time permits, otherwise Phase 1b) |

---

## 2. PubChem Extraction Strategy: FTP vs API (P1)

### FTP Bulk Download (Decided)

**Two files, ~3 GB total:**
1. `bioassays.tsv.gz` (52 MB) — all assay metadata (508K assays)
2. `bioactivities.tsv.gz` (2.98 GB, ~12 GB uncompressed, ~301M rows) — all results

**Processing pipeline:**
```
1. Download bioassays.tsv.gz → filter for confirmatory assays with target annotations
   → ~260,700 AIDs
2. Download bioactivities.tsv.gz → stream-filter for:
   - AID in confirmatory set
   - Activity_Outcome == "Inactive"
   → ~61M inactive records with targets
3. Map SID → CID → SMILES via Sid2CidSMILES.gz (122 MB)
4. Map targets to UniProt via Aid2GeneidAccessionUniProt.gz (7.2 MB)
```

**Time estimate:** Download: 1-2 hours. Processing: 2-4 hours. Total: < 1 day.

**Critical insight:** 96% of PubChem "confirmatory" assays are literature-extracted (ChEMBL/BindingDB). The ~4,500 MLPCN/MLSCN assays from NIH screening centers are the richest source of genuine HTS dose-response inactive data. These should be prioritized for Silver-tier entries.

### API Usage (Supplementary Only)
- For targeted lookups after bulk extraction
- For assay quality metadata not in TSV (Z-factor, dose-response curves)

---

## 3. Realistic Experiment Scope for 11 Weeks (P0+P1)

### Must-Have (NeurIPS Submission)

**ML Track — 18 training runs total:**

| Component | Models | Splits / Conditions | Runs | Notes |
|-----------|--------|---------------------|------|-------|
| Baselines | DeepDTA, GraphDTA, DrugBAN | Random, Cold-Compound, Cold-Target (NegBioDB negatives) | 9 | NegBioDB-neg random-split runs shared with Exp 1 & Exp 4 |
| Exp 1 additional | Same 3 models | Random split: uniform random neg + degree-matched random neg | 6 | NegBioDB condition = baseline random-split (no extra runs) |
| Exp 4 additional | Same 3 models | DDB split (NegBioDB negatives) | 3 | Random-split comparison = baseline (no extra runs) |
| **Total** | | | **18 runs** | |

> **Run count clarification:** Baselines include 3 models × 3 splits = 9 runs, all using NegBioDB negatives. The NegBioDB-negative random-split runs (3) are shared with Exp 1 (as NegBioDB condition) and Exp 4 (as random-split condition). This avoids double-counting while keeping the total at 18.

Estimated GPU time: 18 runs × 2-4 hours = 36-72 hours (3-4 days on single GPU)

**LLM Track — 18 evaluation runs total:**

| Component | Models | Tasks | Configs | Runs |
|-----------|--------|-------|---------|------|
| Baselines | Gemini Flash, Llama 3.3, Mistral 7B | L1, L2, L4 | zero-shot, 3-shot | 18 |
| Exp 9 (LLM vs ML) | Use L1 results vs M1 results | | | 0 (reuse) |
| Exp 10 (extraction) | Use L2 results | | | 0 (reuse) |
| **Total** | | | | **18 runs** |

All automated evaluation — no judge calls needed for must-haves.

**Excluded from must-have (defer to should-have/camera-ready):**
- XGBoost, RF, DTI-LM, EviDTI (ML)
- Phi-3.5, Qwen2.5, Gemini Flash-Lite (LLM)
- Cold-Both, Temporal, Scaffold, DDB splits (except for Exp 4)
- Tasks L3, L5, L6
- Experiments 2, 3, 5, 6, 7, 8, 11, 12

### Should-Have (If Time Permits in Sprint)

| Item | Extra Effort | Value |
|------|-------------|-------|
| RF + XGBoost baselines | +6 training runs (~12h) | Traditional ML comparison |
| L3 (Reasoning) + LLM-as-Judge | 200 examples, 3 models, judge = 6 days | Reasoning capability showcase |
| Exp 7 (target class coverage) | Analysis only, no training | Strong figure for paper |
| Exp 5 (cross-DB consistency) | Analysis only, no training | Quality validation argument |
| Cold-Both + Temporal splits | +6 training runs (~24h) | More splits for heatmap |

### Revised Week-by-Week Timeline

| Week | Task |
|------|------|
| 1 | Schema + download (ChEMBL SQLite 1h, PubChem FTP 2h, BindingDB TSV) |
| 1-2 | Standardization pipeline (RDKit compound, UniProt target) + dedup |
| 2-3 | Confidence tier assignment (automated) + quality spot-check |
| 3 | L1 dataset construction (2,000 MCQ — semi-automated from DB entries) |
| 3-4 | L2 dataset (100 abstracts — LLM first-pass + human correction, 3-4 days) |
| 4 | L4 dataset (500 pairs) + evaluation scripts |
| 4-5 | ML baselines training (18 runs, 3-4 days GPU) |
| 5-6 | LLM baselines evaluation (18 runs, automated) |
| 6-7 | Analysis + figures + should-have experiments if time |
| 7-8 | Croissant metadata + Datasheet |
| 8-10 | Paper writing (9 pages + appendix) |
| 10-11 | Polish + ArXiv preprint + submit |

---

## 4. LLM-as-Judge Rate Limit Analysis (P1)

### Gemini 2.5 Flash Free Tier: 250 RPD

**Must-have tasks (L1, L2, L4): ALL automated evaluation. No judge needed.** ✓

**Should-have (L3 Reasoning):**
- Test set: 170 examples
- 3 LLM models to evaluate
- 3 judge runs per evaluation (majority vote)
- Total: 170 × 3 × 3 = **1,530 calls = 6.1 days**
- ✓ Feasible within sprint

**All judge tasks (L3 + L5 + L6, 3 models):**
- L3: 170 × 3 × 3 = 1,530
- L5: 125 × 3 × 3 = 1,125
- L6: 250 × 3 × 3 = 2,250
- Total: **4,905 calls = 19.6 days with 3 models**
- ⚠️ Tight but possible if started early in sprint (running in background)

**All judge tasks (6 models):**
- Total: **9,810 calls = 39.2 days**
- ✗ NOT feasible for NeurIPS sprint

### Decision
- Must-have: L1, L2, L4 only (all automated). No judge dependency.
- Should-have: Add L3 with 3 models, 6 days of judge calls. Start judging at Week 7.
- Nice-to-have: L5, L6 deferred to camera-ready.
- Flagship models (6 total): deferred to post-submission.

### Mitigation: Local Judge Fallback
- If Gemini rate-limited: Use Llama 3.3 70B via ollama as judge (no rate limit)
- Trade-off: Slightly lower judge quality (Prometheus 2 showed open-source judges are ~0.6-0.7 Pearson vs GPT-4)
- Report both in paper: "Primary judge: Gemini Flash; Validation: Llama 3.3 local"

---

## 5. L4 Anti-Contamination Strategy (P1)

### The Problem
Task L4 asks LLMs to distinguish "experimentally tested negative" from "untested" pairs.
If a compound-target pair's test result is in LLM training data, this becomes a **memorization test**, not a reasoning test.

### Strengthened Strategy

**5a. Temporal partitioning (primary defense):**
- L4 test set: compound-target pairs from assays deposited **after January 2024**
- Most LLMs (Gemini, Llama 3.3, Mistral 7B) have training cutoffs in 2023-2024
- PubChem `bioactivities.tsv.gz` includes deposition dates → filter directly
- ChEMBL v36 (Oct 2025) includes records from 2024-2025 publications

**5b. Low-profile pair selection:**
- For "untested" pairs: use well-known drugs + understudied targets (Tdark from IDG/Pharos)
- For "tested negative" pairs: use lesser-known compounds from recent assays
- Avoid pairs that would be easily Google-able

**5c. Contamination detection metric:**
- Compare model performance on pre-2023 pairs vs. post-2024 pairs
- If accuracy drops >15% on post-2024: memorization is significant → report and flag
- If accuracy is similar: reasoning is dominant → stronger result

**5d. Synthetic untested pairs:**
- Generate novel compound-target pairs by combining real compounds with targets from different therapeutic areas
- Example: approved CNS drug × rare kinase target = almost certainly untested
- Verify against NegBioDB and PubChem that no test data exists

**5e. Ablation: knowledge vs. reasoning:**
- Include "trick" pairs where the drug IS well-known and the target IS well-known, but the specific combination has never been tested
- Example: Metformin (diabetes, well-known) + CDK4/6 (oncology, well-known) = no published binding data
- If LLM says "untested" → correct reasoning. If LLM says "tested" → hallucination/guessing.

### Implementation
- 500 total L4 pairs: 250 tested (from NegBioDB) + 250 untested
- Tested: 125 pre-2023 + 125 post-2024 (for contamination detection)
- Untested: 125 well-known drug × Tdark target + 125 well-known drug × well-known target (trick pairs)

---

## 6. L2 Dataset: Reduction and Semi-Automation (P1)

### Current Plan: 200 abstracts, manual annotation
- Time: ~10-15 min/abstract × 200 = 33-50 hours = 1-1.5 weeks full-time
- Bottleneck: largest single-person task in the sprint

### Revised Plan: 100 abstracts, semi-automated

**Step 1: Abstract selection (2 hours)**
- Use PubMed search: `("did not inhibit" OR "no significant binding" OR "inactive" OR "IC50 >") AND (drug OR compound) AND (target OR receptor OR kinase)`
- Select 100 abstracts with visible negative DTI results
- Stratify: 40 explicit, 30 hedged, 30 implicit negatives

**Step 2: LLM first-pass extraction (4 hours)**
- Run Gemini Flash or Llama 3.3 on each abstract with L2 prompt
- Generate initial JSON extraction for each
- Automated: ~2-3 minutes per abstract

**Step 3: Human correction (15-20 hours)**
- Review each LLM extraction against the original abstract
- Correct errors, add missing entries, fix field values
- ~8-12 min/abstract (faster than from-scratch annotation because LLM provides scaffold)

**Step 4: Quality verification (3 hours)**
- Random sample 20 abstracts: independent re-annotation
- Compute inter-annotator agreement (self-consistency check)

**Total: ~25 hours over ~4 days**

### For NeurIPS Paper
- 100 abstracts is sufficient — CaseReportBench used 200 cases total, LIT-PCBA used 15 targets
- Clearly document the semi-automated annotation process as a contribution
- Report inter-annotator agreement as quality metric

---

## 7. Paper Narrative: Exp 1 + Exp 4 as Opening Hook (P0)

### Current framing (06_paper_narrative.md abstract):
"We introduce NegBioDB..." → database-first narrative

### Revised framing: Problem-first narrative

**Opening hook should be:**
> "We demonstrate that existing DTI benchmarks systematically overestimate model performance by 10-20% due to the use of assumed negatives (Experiment 1), and that this inflation is further compounded by node degree bias in negative sampling (Experiment 4). To address this, we introduce NegBioDB..."
>
> **Note:** The "10-20%" figure is hypothesized, not measured. Actual value will come from Exp 1 results (Week 6 Go/No-Go checkpoint).

### Paper structure recommendation:
1. **Section 1 (Intro):** The problem — show Exp 1 + Exp 4 preview results as motivation
2. **Section 2:** NegBioDB database design + curation methodology
3. **Section 3:** NegBioBench dual-track benchmark design
4. **Section 4:** Experiments and Results
   - 4.1: Exp 1 (performance inflation) — the main result
   - 4.2: Exp 4 (node degree bias)
   - 4.3: ML baseline comparison across splits
   - 4.4: LLM benchmark results (L1, L2, L4)
   - 4.5: Exp 9 (ML vs LLM comparison)
5. **Section 5:** Analysis and Discussion
6. **Section 6:** Conclusion + Limitations

**Key change:** The paper is NOT "here's a database" — it's "here's proof that current benchmarks are broken, and here's how to fix them." The database is the solution, not the contribution.

---

## 8. Cross-DB Conflict Resolution Examples (P2)

### Why This Matters
Reviewers will ask: "How do you handle conflicting results across databases?"
We need concrete examples, not just a pipeline description.

### Expected Conflict Types

**Type 1: Same compound-target, different activity labels**
- Example (hypothetical): Compound X vs EGFR
  - ChEMBL: IC50 = 8.5 uM (pchembl = 5.07) → "borderline active"
  - PubChem confirmatory: "Inactive" at 10 uM screen
  - Resolution: Record both; assign as "Conditional Negative" with assay context
  - This IS the 5-15% from Experiment 3

**Type 2: Different thresholds across sources**
- BindingDB: Kd = 15 uM → "weakly active" by some definitions
- HCDT threshold (>100 uM) → "not included as negative"
- Our threshold (>10 uM) → "included as Bronze-tier negative"
- Resolution: Record activity value; let users filter by their own threshold

**Type 3: Assay format discordance**
- Biochemical assay: IC50 > 100 uM (clearly inactive)
- Cell-based assay: EC50 = 5 uM (active)
- Resolution: Both are valid; compound is "conditionally negative" (biochemically inactive but cellularly active, possibly through off-target effect)
- Record as two separate entries with different assay_format values

### Implementation
- Run Experiment 5 (Cross-DB Consistency) early in pipeline
- Document top 20 most-conflicting compound-target pairs
- Use as examples in paper to demonstrate curation value over aggregation

---

## 9. Assay Format Stratified Analysis (P2)

### Addition to Experiment 3

**Current Exp 3 design:** Identify pairs with conflicting results across assays
**Enhanced Exp 3 design:** Stratify conflicts by assay format

| Stratification | Analysis |
|---------------|----------|
| Biochemical vs. Cell-based | How often does a compound inactive biochemically become active in cells? (permeability effect) |
| Primary vs. Confirmatory | False-negative rate estimation for primary single-point screens |
| Different target constructs | Full-length vs. catalytic domain → different binding pocket accessibility |
| Human vs. Rodent | Cross-species activity discordance rate |

### Addition to Benchmark Design
- All ML splits should report metrics stratified by `assay_format`
- Include in supplementary table: "Performance by assay format (biochemical vs. cell-based)"
- This addresses medicinal chemistry reviewer concern MC3

---

## 10. HCDT 2.0 Threshold Differentiation (P2)

### The Risk
HCDT 2.0 uses >100 uM threshold. If we use the same threshold on the same sources, the output is nearly identical → potential ND license argument.

### Solution: Multi-Threshold Analysis

| Threshold | Label | Use |
|-----------|-------|-----|
| >100 uM (Kd/Ki/IC50) | "Stringent Inactive" | HCDT-equivalent, Gold/Silver tier |
| >10 uM | "Standard Inactive" | Our primary threshold (community standard, pchembl < 5) |
| >1 uM | "Weak/Inactive" | Extended set, includes borderline cases |

**NegBioDB stores the actual activity VALUE, not just a binary label.**
- Users can apply ANY threshold they prefer
- Our default splits use >10 uM (standard in DTI literature)
- This is fundamentally different from HCDT 2.0's approach (binary label only)

### Key Differentiators from HCDT 2.0
1. We use **10 uM** as primary threshold (not 100 uM) — more inclusive
2. We store **quantitative values** (not just binary inactive)
3. We include **PubChem confirmatory data** (HCDT 2.0 does not)
4. We add **confidence tiers** (HCDT 2.0 has no quality stratification)
5. We provide **benchmark splits** (HCDT 2.0 has no ML-ready format)

This makes NegBioDB substantively different in approach, not just a derivative.

---

## 11. Inactive Definition Caveat (MC1)

### Required Caveat for Paper

> "NegBioDB defines 'inactive' as 'showing no significant activity within the tested assay context at the tested concentration range.' This does not imply absolute non-interaction. A compound inactive in a biochemical binding assay may interact with the target through allosteric mechanisms, in different cellular contexts, or at concentrations above those tested. All entries record the specific assay conditions and maximum concentration tested, enabling users to apply context-appropriate activity thresholds."

### Implementation
- Add `inactivity_caveat` field to metadata (text, standardized)
- Possible values: "inactive_at_tested_concentrations", "inactive_in_this_assay_format", "inactive_for_this_endpoint"
- Include this caveat prominently in Datasheet for Datasets

---

## 12. Data Leakage Prevention for ML Track (BM3)

### Overlap Analysis Plan

**Must include in paper:**

| Overlap Check | Method |
|---------------|--------|
| NegBioDB test compounds ∩ DAVIS train compounds | InChIKey comparison |
| NegBioDB test compounds ∩ TDC/KIBA compounds | InChIKey comparison |
| NegBioDB test compounds ∩ DUD-E actives/decoys | InChIKey comparison |
| NegBioDB test targets ∩ DAVIS targets | UniProt comparison |
| NegBioDB test targets ∩ LIT-PCBA targets | Target name comparison |

**Report in paper:** "X% of NegBioDB test set compounds appear in DAVIS training set"
**If high overlap:** Use cold-compound split as primary evaluation (test compounds guaranteed unseen)

### Additional Leakage Prevention
- For temporal split: verify no data leakage via publication date (not deposit date)
- For scaffold split: verify Murcko scaffolds don't bridge train/test
- Run sanity check: train model with empty features → if performance > random, there's leakage

---

## 13. DDB Split Implementation (BM2)

### Reference Implementation

DDB (Degree Distribution Balanced) from BMC Biology 2025:

```python
# Pseudo-code for DDB negative sampling
def ddb_sample_negatives(positive_pairs, all_compounds, all_targets, ratio=1):
    """
    Sample negatives that match the degree distribution of positives.
    Avoids the bias where well-connected nodes are over-represented.
    """
    # 1. Compute degree for each compound and target in positive set
    compound_degree = Counter(c for c, t in positive_pairs)
    target_degree = Counter(t for c, t in positive_pairs)

    # 2. For each positive pair (c, t), sample a negative pair (c', t')
    #    where degree(c') ≈ degree(c) and degree(t') ≈ degree(t)
    negatives = []
    for c, t in positive_pairs:
        c_deg = compound_degree[c]
        t_deg = target_degree[t]

        # Find compounds with similar degree
        candidates_c = [c2 for c2 in all_compounds
                       if abs(compound_degree.get(c2, 0) - c_deg) <= 1]
        candidates_t = [t2 for t2 in all_targets
                       if abs(target_degree.get(t2, 0) - t_deg) <= 1]

        # Sample negative pair not in positive set
        neg_c = random.choice(candidates_c)
        neg_t = random.choice(candidates_t)
        while (neg_c, neg_t) in positive_pairs:
            neg_c = random.choice(candidates_c)
            neg_t = random.choice(candidates_t)

        negatives.append((neg_c, neg_t))

    return negatives
```

**Reference code:** https://github.com/NYXFLOWER/GripNet (BMC Biology 2025 supplementary)

---

## 14. Co-Author Consideration (ST3)

### Options (User's Decision)
1. **Advisor/PI at institution** — strongest signal for NeurIPS reviewers
2. **Collaborator with domain expertise** (medicinal chemist, cheminformatician)
3. **Computational collaborator** who runs some experiments
4. **Solo author** — viable if reproducibility is impeccable

### If Solo Author: Mitigation
- GitHub repo with full commit history showing development
- Docker/conda environment for 100% reproducible experiments
- All raw data and scripts publicly available at submission
- Detailed supplementary with every experimental detail
- Emphasize "individual researcher" accessibility as a feature, not a limitation

---

## 15. Positive Data Protocol for ML Benchmarking (P0 — Expert Panel v6)

### The Gap

NegBioDB collects only inactive (negative) compound-target pairs. However, ML Task M1 (binary DTI prediction) requires both actives and inactives. No prior document defined the positive data source.

### Decision

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Positive source** | ChEMBL v36, pChEMBL ≥ 6 | Standard potency threshold (IC50/Ki/Kd/EC50 ≤ 1 uM). Quality-filtered. |
| **Target pool** | Shared targets only | Positives and negatives cover the same target set. No "positive-only" or "negative-only" targets in benchmark |
| **Validation set** | DAVIS actives (pKd ≥ 7) | Complete matrix = gold-standard ground truth |
| **Borderline exclusion** | pChEMBL 4.5–5.5 excluded from both | Clean class separation. Eliminates threshold artifacts |
| **Class ratios** | Balanced (1:1) + Realistic (1:10) | Both reported. Balanced for fair model comparison; Realistic for drug discovery simulation |

### Why 1 uM and Not 100 nM?

The 1 uM (pChEMBL 6) active threshold is standard in DTI benchmarks (TDC, WelQrate). Using 100 nM (pChEMBL 7) would reduce the positive pool substantially and bias toward potent inhibitors only. The gap between pChEMBL 6 (active) and pChEMBL 5 (inactive) provides a clear 10-fold separation.

---

## 16. Random Negative Control Design for Exp 1 (P0 — Expert Panel v6)

### The Gap

Exp 1 is the paper's primary result: "NegBioDB negatives vs. random negatives." But "random" was not precisely defined, allowing multiple interpretations that affect the result.

### Decision: Two Random Controls

| Control | Generation Method | What It Tests |
|---------|------------------|---------------|
| **Uniform random** | Sample untested (compound, target) pairs uniformly from the full cross-product | Standard practice (TDC default). Maximum expected performance inflation |
| **Degree-matched random** | Sample untested pairs matching NegBioDB's compound-degree and target-degree distribution | Isolates confirmation effect from degree bias. More rigorous control |

### Experimental Design

- 3 ML models × 3 negative conditions (NegBioDB, uniform random, degree-matched random) = 9 runs
- Same positive data, same random split, same seed
- Only the negative set changes
- Degree-matched random is generated via the DDB sampling algorithm (research/08 §13) applied in reverse

### Expected Results

- Uniform random: Highest apparent model performance (inflated by easy negatives)
- Degree-matched random: Moderate performance (some bias removed)
- NegBioDB: Lowest apparent performance (hardest negatives = most realistic)
- **The gap between uniform random and NegBioDB is the "inflation" number for the paper abstract**

---

## 17. Project Code Structure (P0 — Expert Panel v6)

### Recommended Directory Layout

```
negbiodb/
├── README.md
├── pyproject.toml              # Python 3.11+, rdkit, pandas, pyarrow, mlcroissant
├── Makefile                    # Pipeline orchestration with dependency tracking
├── config.yaml                 # All configurable parameters (thresholds, paths, URLs)
├── Dockerfile
│
├── src/negbiodb/               # Installable Python package
│   ├── __init__.py
│   ├── config.py               # Load config.yaml
│   ├── standardize/            # RDKit compound + UniProt target standardization
│   ├── extract/                # Per-source extraction (pubchem, chembl, bindingdb, davis)
│   ├── curate/                 # Dedup, confidence tiers, quality flags
│   ├── benchmark/              # Split generation, ML baselines, LLM evaluation
│   ├── export/                 # CSV/Parquet export, Croissant generation
│   └── api.py                  # User-facing Python API (TDC-compatible)
│
├── scripts/                    # CLI entry points (01_fetch_pubchem.py ... 09_export.py)
├── migrations/                 # Numbered SQL migration files
├── tests/                      # pytest: standardize, dedup, export edge cases
├── data/                       # .gitignore'd — generated data
├── exports/                    # .gitignore'd — generated exports
├── research/                   # Planning & analysis documents
├── paper/                      # LaTeX source
└── croissant/                  # metadata.json
```

### Critical Tests to Implement

1. `test_standardize.py`: Salt removal, stereoisomers, empty SMILES, tautomers, InChIKey generation
2. `test_dedup.py`: Cross-DB deduplication correctness, borderline zone exclusion
3. `test_export.py`: CSV/Parquet schema validation, split assignment completeness, Croissant validation

---

## Summary of All Decisions

| Item | Decision | Impact |
|------|----------|--------|
| Minimum dataset size | **10K** (up from 5K) | Strengthens "large-scale" claim |
| PubChem extraction | **FTP bulk download** (3 GB, < 1 day) | Eliminates API bottleneck |
| ChEMBL extraction | **SQLite via chembl_downloader** (4.6 GB, 1 hour) | Fastest setup |
| ML must-have scope | **3 models × 3 splits** (18 training runs, ~3 days GPU) | Feasible in sprint |
| LLM must-have scope | **3 models × 3 tasks × 2 configs** (18 eval runs, automated) | No rate limit issue |
| LLM judge scope | **L3 only with 3 models** (1,530 calls, 6 days) — should-have | Within 250 RPD limit |
| L2 dataset | **100 abstracts, semi-automated** (~25 hours, 4 days) | Halves annotation burden |
| L4 anti-contamination | **Post-2024 data + trick pairs + contamination detection** | Addresses memorization |
| Paper narrative | **Problem-first**: "benchmarks are broken" → NegBioDB as fix | More compelling hook |
| Inactive definition | **10 uM primary threshold** (not 100 uM like HCDT) + store actual values | Legal differentiation |
| Inactive caveat | Explicit "within tested assay context" language | Scientific rigor |
| Data leakage check | **Mandatory overlap analysis** with DAVIS/TDC/DUD-E | Prevents reviewer objection |
| Positive data source | **ChEMBL pChEMBL ≥ 6** (shared targets, borderline 4.5-5.5 excluded) | Enables ML binary classification |
| Class ratios | **1:1 (balanced) + 1:10 (realistic)** both reported | Fair comparison + real-world simulation |
| Random negative controls | **Uniform + degree-matched** (2 controls for Exp 1) | Rigorous experimental design |
| Project structure | **src/negbiodb/ + Makefile + pyproject.toml + Dockerfile** | Reproducibility + solo author credibility |
| GPU strategy | **Kaggle free tier** (30h/wk) primary; Colab Pro fallback | $0 budget maintained |
| ChEMBL version | **v36** (Sep 2025, 24.3M activities) | Latest available data |
| Go/No-Go framework | **Week 6 checkpoint**: Exp 1 result determines narrative | Risk mitigation for weak results |

---

## Sources

- PubChem FTP Bioassay: https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/Extras/
- ChEMBL downloader: https://pypi.org/project/chembl-downloader/
- ChEMBL FTP: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/
- DDB sampling: https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-025-02231-w
- BMC Biology 2025 code: https://github.com/NYXFLOWER/GripNet
- UniChem cross-reference: https://www.ebi.ac.uk/unichem/
- Compound-target pairs dataset: https://www.nature.com/articles/s41597-024-03582-9
- PubChem inactive pairs analysis: https://pmc.ncbi.nlm.nih.gov/articles/PMC4619454/
