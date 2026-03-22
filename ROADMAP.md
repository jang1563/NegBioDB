# NegBioDB — Execution Roadmap

> Last updated: 2026-03-22 (v18 — DTI complete, CT COMPLETE, PPI COMPLETE — all 3 domains done)

---

## Critical Findings (Updated March 2026)

1. **HCDT 2.0 License: CC BY-NC-ND 4.0** — Cannot redistribute derivatives. Must independently recreate from underlying sources (BindingDB, ChEMBL, GtoPdb, PubChem, TTD). Use 10 uM primary threshold (not 100 uM) to differentiate.
2. **InertDB License: CC BY-NC** — Cannot include in commercial track. Provide optional download script only.
3. **Submission requirements**: downloadable data, Croissant metadata, code available, Datasheet for Datasets.
4. **LIT-PCBA compromised** (2025 audit found data leakage) — Creates urgency for NegBioDB as replacement gold-standard.
5. **Recommended NegBioDB License: CC BY-SA 4.0** — Compatible with ChEMBL (CC BY-SA 3.0) via one-way upgrade.
6. **No direct competitor exists** as of March 2026.
7. **No LLM benchmark tests negative DTI tasks** — ChemBench, Mol-Instructions, MedQA, SciBench all lack negative result evaluation. NegBioBench LLM track is first-of-kind.
8. **LLM evaluation also free** — Gemini Flash free tier as LLM-as-Judge + ollama local models as baselines. Flagship models (GPT-4, Claude) added post-stabilization only.
9. **Data volume is NOT the bottleneck** — ChEMBL alone has ~527K quality inactive records (pchembl < 5, validated). PubChem has ~61M target-annotated confirmatory inactives. Estimated 200K+ unique compound-target pairs available. Minimum target raised to **10K curated entries** (from 5K).
10. **PubChem FTP bulk is far superior to API** — `bioactivities.tsv.gz` (3 GB) contains all 301M bioactivity rows. Processing: < 1 day. API approach would take weeks.
11. **LLM-as-Judge rate limit (250 RPD)** — Must-have tasks (L1, L2, L4) all use automated evaluation. Judge needed only for should-have L3 (1,530 calls = 6 days). All judge tasks with 3 models = 20 days. With 6 models = 39 days (NOT feasible for sprint).
12. **Paper narrative must be problem-first** — "Existing benchmarks are broken" (Exp 1 + Exp 4), not "Here's a database." Database is the solution, not the contribution.
13. **Positive data protocol required** — NegBioDB is negative-only. For ML benchmarking (M1), positive data must be sourced from ChEMBL (pChEMBL ≥ 6). Report two class ratios: balanced (1:1) and realistic (1:10). See §Positive Data Protocol below.
14. **Random negative baseline must be precisely defined** — Exp 1 compares NegBioDB negatives against random negatives. Random = uniform sampling from untested compound-target pairs (TDC standard). See §Random Negative Control Design.
15. **Paper format: 9 pages** + unlimited appendix. Croissant is **mandatory** (desk rejection if missing/invalid).
16. **GPU strategy: Kaggle free tier** (30 hrs/week) is sufficient for 18 ML baseline runs (~36-72 GPU-hours over 4 weeks). Fallback: Colab Pro ($10/month).
17. **ChEMBL v36** (Sep 2025, 24.3M activities) should be used, not v35. `chembl_downloader` fetches latest by default.
18. **[UNVERIFIED] Nature MI 2025 validates our premise** — Biologically driven negative subsampling paper independently shows "assumed negatives" distort DTI models. Must cite in Related Work alongside Science 2025 editorial on negative results in AI. **Both citations need verification in Week 1.** If not found, substitute: EviDTI (Nature Comms 2025), DDB paper (BMC Biology 2025), LIT-PCBA audit (2025).

---

## Cost Strategy

**Pre-publication cost target: $0.** All data sources are free. LLM pipeline uses free tiers and local models only.

### LLM Pipeline (Zero Cost)

```
Stage 1: Coarse Filtering ("Does this paper contain negative DTI results?")
  → Mistral 7B local via ollama (no rate limit, unlimited)
  → OR Gemini 2.5 Flash-Lite free tier (1,000 RPD)

Stage 2: Fine-grained Extraction (compound, target, conditions, outcome)
  → Gemini 2.5 Flash free tier (250 RPD, 250K TPM)
  → OR Llama 3.3 local via ollama (if RAM ≥ 32GB)

Stage 3: Validation
  → Human review on sampled outputs (no LLM cost)
```

**Throughput estimate:**
- 10K papers via Gemini Flash-Lite: ~10 days (free)
- 100K papers via local Mistral 7B: ~1-2 weeks (free, speed depends on hardware)

### Infrastructure (Free Tier)
- DB: SQLite local (MVP) → Supabase free (Phase 2)
- Web: Vercel free tier
- Storage: GitHub LFS / Zenodo (dataset DOI)
- CI/CD: GitHub Actions free

### Only Paid Cost: Publication
- OA APC: ~$2,500-3,000 (J. Cheminformatics or Nature Sci Data)
- Conference registration: ~$200-400
- D&B track: No publication fee (if accepted)

---

## Positive Data Protocol (P0 — Expert Panel Finding)

NegBioDB is a negative-only database. For ML benchmarking (Task M1: binary DTI prediction), **positive (active) data is required**. This section defines the protocol.

### Positive Data Source

```sql
-- Extract active DTIs from ChEMBL v36 SQLite
-- Threshold: pChEMBL ≥ 6 (IC50/Ki/Kd/EC50 ≤ 1 uM)
SELECT
  a.molregno, a.pchembl_value, a.standard_type,
  cs.canonical_smiles, cs.standard_inchi_key,
  cp.accession AS uniprot_id
FROM activities a
JOIN compound_structures cs ON a.molregno = cs.molregno
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
LEFT JOIN target_components tc ON td.tid = tc.tid
LEFT JOIN component_sequences cp ON tc.component_id = cp.component_id
WHERE a.pchembl_value >= 6
  AND a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
  AND a.data_validity_comment IS NULL
  AND td.target_type = 'SINGLE PROTEIN'
  AND cp.accession IS NOT NULL
```

### Positive-Negative Pairing

| Setting | Ratio | Purpose | Primary Use |
|---------|-------|---------|-------------|
| **Balanced** | 1:1 (active:inactive) | Fair model comparison | Exp 1, Exp 4, baselines |
| **Realistic** | 1:10 (active:inactive) | Real-world HTS simulation | Supplementary evaluation |

- Positives restricted to **shared targets** between ChEMBL actives and NegBioDB inactives (same target pool)
- Same compound standardization pipeline (RDKit) applied to positives
- DAVIS matrix known actives (pKd ≥ 7, Kd ≤ 100 nM) used as **Gold-standard validation set**

### Overlap Prevention

- Active and inactive compound-target pairs must not overlap (same pair cannot be both active and inactive)
- Borderline zone (pChEMBL 4.5–5.5) excluded from both positive and negative sets for clean separation
- Overlap analysis: report % of NegBioDB negatives where the same compound appears as active against a different target

---

## Random Negative Control Design (P0 — Expert Panel Finding)

Experiment 1 compares NegBioDB's experimentally confirmed negatives against **random negatives**. The random negative generation must be precisely defined.

### Control Conditions for Exp 1

| Control | Method | What it Tests |
|---------|--------|---------------|
| **Uniform random** | Sample untested compound-target pairs uniformly at random from the full cross-product space | Standard TDC approach; tests baseline inflation |
| **Degree-matched random** | Sample untested pairs matching the degree distribution of NegBioDB pairs | Isolates the effect of experimental confirmation vs. degree bias |

**All Exp 1 runs:**
- 3 ML models (DeepDTA, GraphDTA, DrugBAN)
- Random split only (for controlled comparison)
- Same positive data, same split seed
- Only the negative set changes: NegBioDB confirmed vs. uniform random vs. degree-matched random
- **Total: 3 models × 3 negative conditions = 9 runs** (was 3 runs; updated)
- **Note:** The 3 NegBioDB-negative random-split runs are shared with the baseline count (9 baselines include random split). Thus Exp 1 adds only **6 new runs** (uniform random + degree-matched random). Similarly, Exp 4 shares the random-split baseline and adds only **3 new DDB runs**. Overall: 9 baseline + 6 Exp 1 + 3 Exp 4 = **18 total**.
- **Exp 4 definition:** The DDB comparison uses a full-task degree-balanced split on the merged M1 balanced benchmark. Positives and negatives are reassigned together under the same split policy.

### Reporting

- Table: [Model × Negative Source × Metric] for LogAUC, AUPRC, MCC
- Expected: NegBioDB > degree-matched > uniform random for precision-oriented metrics
- If NegBioDB ≈ uniform random → narrative shifts to Exp 4 (DDB bias) as primary result

---

## Phase 1: Implementation Sprint (Weeks 0-11)

### Week 1: Scaffolding + Download + Schema ✅ COMPLETE

- [x] **Project scaffolding**: Create `src/negbiodb/`, `scripts/`, `tests/`, `migrations/`, `config.yaml`, `Makefile`, `pyproject.toml`
- [x] **Dependency management**: `pyproject.toml` with Python 3.11+, rdkit, pandas, pyarrow, mlcroissant, tqdm, scikit-learn
- [x] **Makefile skeleton**: Define target structure (full pipeline encoding in Week 2)
- [x] Finalize database schema (SQLite for MVP) — apply `migrations/001_initial_schema.sql`
- [x] Download all source data (see below — < 1 day total)
- [x] **Verify ChEMBL v36** (Sep 2025) downloaded, not v35
- [x] **[B7] Verify PubChem bioactivities.tsv.gz column names** after download
- [ ] **[B4] Hardware decision**: Test local RAM/GPU. If < 32GB RAM → use Llama 3.1 8B + Mistral 7B (not 70B). If ≥ 32GB → quantized Llama 3.3 70B (Q4). Document choice.
- [ ] **[B2] Verify citations**: Search for Nature MI 2025 negative subsampling paper + Science 2025 editorial. If not found → substitute with EviDTI, DDB paper, LIT-PCBA audit
- [ ] **[B3] Monitor submission deadlines**

### Week 2: Standardization + Extraction Start ✅ COMPLETE

- [x] Implement compound standardization pipeline (RDKit: salt removal, normalization, InChIKey)
- [x] Implement target standardization pipeline (UniProt accession as canonical ID)
- [x] Set up cross-DB deduplication (InChIKey[0:14] connectivity layer)
- [x] **Makefile pipeline**: Encode full data pipeline dependency graph as executable Makefile targets
- [ ] **[B5] Check shared target pool size**: Count intersection of NegBioDB targets ∩ ChEMBL pChEMBL ≥ 6 targets. If < 200 targets → expand NegBioDB target extraction
- [ ] **[B6] Check borderline exclusion impact**: Run pChEMBL distribution query on ChEMBL. Estimate data loss from excluding pChEMBL 4.5–5.5 zone

### Week 2-4: Data Extraction ✅ COMPLETE

**Result: 30.5M negative_results (>minimum target of 10K — far exceeded)**

**Data Sources (License-Safe Only):**

| Source | Available Volume | Method | License |
|--------|-----------------|--------|---------|
| PubChem BioAssay (confirmatory inactive) | **~61M** (target-annotated) | **FTP bulk: `bioactivities.tsv.gz` (3 GB)** + `bioassays.tsv.gz` (52 MB) | Public domain |
| ChEMBL pChEMBL < 5 (quality-filtered) | **~527K** records → ~100-200K unique pairs | **SQLite via `chembl_downloader`** (4.6 GB, 1h setup) | CC BY-SA 3.0 |
| ChEMBL activity_comment "Not Active" | **~763K** (literature-curated) | SQL query on same SQLite dump | CC BY-SA 3.0 |
| BindingDB (Kd/Ki > 10 uM) | **~30K+** | Bulk TSV download + filter | CC BY |
| DAVIS complete matrix (pKd ≤ 5) | **~27K** | TDC Python download | Public/academic |

**NOT bundled (license issues):**
- HCDT 2.0 (CC BY-NC-ND) — Use as validation reference only; we use 10 uM threshold (not 100 uM) to differentiate
- InertDB (CC BY-NC) — Optional download script for users

**PubChem FTP extraction pipeline (< 1 day):**
```
1. bioassays.tsv.gz → filter confirmatory AIDs with target annotations → ~260K AIDs
2. bioactivities.tsv.gz (stream) → filter AID ∈ confirmatory, Outcome=Inactive → ~61M records
3. Prioritize MLPCN/MLSCN assays (~4,500 AIDs, genuine HTS dose-response) for Silver tier
4. Map SID→CID via Sid2CidSMILES.gz, targets via Aid2GeneidAccessionUniProt.gz
```

- [x] Download PubChem FTP files (bioactivities.tsv.gz + bioassays.tsv.gz + mapping files)
- [x] Download ChEMBL v36 SQLite via chembl_downloader
- [x] Download BindingDB bulk TSV
- [x] Build PubChem FTP extraction script (**streaming with chunksize=100K** — 12GB uncompressed)
- [x] Build ChEMBL extraction SQL: inactive (activity_comment + pChEMBL < 5) **AND active (pChEMBL ≥ 6)** for positive data
- [x] Build BindingDB extraction script (filter Kd/Ki > 10 uM, human targets)
- [x] Integrate DAVIS matrix from TDC (both actives pKd ≥ 7 and inactives pKd ≤ 5)
- [x] Run compound/target standardization on all extracted data (multiprocessing for RDKit)
- [x] Run cross-DB deduplication + **overlap analysis** (vs DAVIS, TDC, DUD-E, LIT-PCBA)
- [x] Assign confidence tiers (gold/silver/bronze/copper — lowercase, matching DDL CHECK constraint)
- [x] **Extract ChEMBL positives**: 883K → 863K after 21K overlap removal (pChEMBL ≥ 6, shared targets only)
- [x] **Positive-negative pairing**: M1 balanced (1.73M, 1:1) + M1 realistic (9.49M, 1:10). Zero compound-target overlap verified.
- [x] **Borderline exclusion**: pChEMBL 4.5–5.5 removed from both pools
- [x] Spot-check top 100 most-duplicated compounds (manual QC checkpoint)
- [x] Run data leakage check: cold split leaks = 0, cross-source overlaps documented

### Week 3-5: Benchmark Construction (ML + LLM)

**ML Track:**
- [x] Implement 3 must-have splits (Random, Cold-Compound, Cold-Target) + DDB for Exp 4
- [x] Implement ML evaluation metrics: LogAUC[0.001,0.1], BEDROC, EF@1%, EF@5%, AUPRC, MCC, AUROC
- [x] (Should have) Add Cold-Both, Temporal, Scaffold splits (all 6 implemented)

**LLM Track:** ✅ INFRASTRUCTURE COMPLETE (2026-03-12)
- [x] Design prompt templates for L1, L2, L4 (priority tasks) → `llm_prompts.py`
- [x] Construct L1 dataset: 2,000 MCQ from NegBioDB entries → `build_l1_dataset.py`
- [x] Construct L2 dataset: 116 candidates (semi-automated) → `build_l2_dataset.py`
- [x] Construct L4 dataset: 500 tested/untested pairs → `build_l4_dataset.py`
- [x] Implement automated evaluation scripts → `llm_eval.py` (L1: accuracy/F1, L2: entity F1, L4: classification F1)
- [x] Build compound name cache → `compound_names.parquet` (144,633 names from ChEMBL)
- [x] Construct L3 dataset: 50 pilot reasoning examples → `build_l3_dataset.py`
- [x] LLM client (vLLM + Gemini) → `llm_client.py`
- [x] SLURM templates + batch submission → `run_llm_local.slurm`, `run_llm_gemini.slurm`, `submit_llm_all.sh`
- [x] Results aggregation → `collect_llm_results.py` (Table 2)
- [x] 54 new tests (29 eval + 25 dataset), 329 total pass
- [ ] **L2 gold annotation**: 15–20h human review needed for `l2_gold.jsonl`

**Shared:**
- [ ] Generate Croissant machine-readable metadata (mandatory for submission)
- [ ] **Validate Croissant** with `mlcroissant` library. Gate: `mlcroissant.Dataset('metadata.json')` runs without errors
- [ ] Write Datasheet for Datasets (Gebru et al. template)

### Week 5-7: Baseline Experiments (ML + LLM)

**ML Baselines:**

| Model | Type | Priority | Runs (3 splits) | Status |
|-------|------|----------|-----------------|--------|
| DeepDTA | Sequence CNN | Must have | 3 | ✅ Implemented |
| GraphDTA | Graph neural network | Must have | 3 | ✅ Implemented |
| DrugBAN | Bilinear attention | Must have | 3 | ✅ Implemented |
| Random Forest | Traditional ML | Should have | 3 | Planned |
| XGBoost | Traditional ML | Should have | 3 | Planned |
| DTI-LM | Language model-based | Nice to have | 3 | Planned |
| EviDTI | Evidential/uncertainty | Nice to have | 3 | Planned |

**Must-have ML: 9 baseline runs (3 models × 3 splits) + 6 Exp 1 (2 random conditions) + 3 Exp 4 (DDB split) = 18 total (~36-72 GPU-hours, 3-4 days)**

> **Status (2026-03-13):** All 18/18 ML baseline runs COMPLETE on Cayuga HPC. Results in `results/baselines/`. 3 timed-out DrugBAN jobs recovered via `eval_checkpoint.py`. Key findings: degree-matched negatives inflate LogAUC by +0.112 avg; cold-target LogAUC drops to 0.15–0.33; DDB ≈ random (≤0.010 diff).

**LLM Baselines (all free):**

| Model | Access | Priority |
|-------|--------|----------|
| Gemini 2.5 Flash | Free API (250 RPD) | Must have |
| Llama 3.3 70B | Ollama local | Must have |
| Mistral 7B | Ollama local | Must have |
| Phi-3.5 3.8B | Ollama local | Should have |
| Qwen2.5 7B | Ollama local | Should have |

**Must-have LLM: 3 models × 3 tasks (L1,L2,L4) × 2 configs (zero-shot, 3-shot) = 18 eval runs (all automated)**

**Flagship models (post-stabilization):**
- GPT-4/4.1, Claude Sonnet/Opus, Gemini Pro — added to leaderboard later

**Must-have experiments (minimum for paper):**
- [x] **Exp 1: NegBioDB vs. random negatives** ✅ COMPLETE — degree-matched avg +0.112 over negbiodb → benchmark inflation confirmed
- [x] **Exp 4: Node degree bias** ✅ COMPLETE — DDB ≈ random (≤0.010 diff) → degree balancing alone not harder
- [ ] **Exp 9: LLM vs. ML comparison** (L1 vs. M1 on matched test set — reuses baseline results; awaiting LLM runs)
- [ ] **Exp 10: LLM extraction quality** (L2 entity F1 — awaiting LLM runs)

**Should-have experiments (strengthen paper, no extra training):**
- [ ] Exp 5: Cross-database consistency (analysis only, no training)
- [ ] Exp 7: Target class coverage analysis (analysis only)
- [ ] Exp 11: Prompt strategy comparison (add CoT config to LLM baselines)
- [ ] L3 task + Exp 12: LLM-as-Judge reliability (1,530 judge calls = 6 days)

**Nice-to-have experiments (defer to camera-ready):**
- [ ] Exp 2: Confidence tier discrimination
- [ ] Exp 3: Assay context dependency (with assay format stratification)
- [ ] Exp 6: Temporal generalization
- [ ] Exp 8: LIT-PCBA recapitulation

### Week 8-10: Paper Writing

- [ ] Write benchmark paper (**9 pages** + unlimited appendix)
- [ ] Create key figures (see research/06_paper_narrative.md for figure plan)
- [ ] **Paper structure (9 pages)**: Intro (1.5) → DB Design (1.5) → Benchmark (1.5) → Experiments (3) → Discussion (1.5)
- [ ] **Appendix contents**: Full schema DDL, all metric tables, L2 annotation details, few-shot examples, Datasheet
- [ ] Python download script: `pip install negbiodb` or simple wget script
- [ ] Host dataset (HuggingFace primary + Zenodo DOI for archival)
- [ ] Author ethical statement
- [ ] **Dockerfile** for full pipeline reproducibility: Python 3.11, rdkit, torch, chembl_downloader, pyarrow, mlcroissant. Must reproduce full pipeline from raw data → final benchmark export

### Week 10-11: Review & Submit

- [ ] Internal review and polish
- [ ] Submit abstract (~May 1)
- [ ] Submit full paper (~May 15)
- [ ] Post ArXiv preprint (same day or before submission)

---

## Phase 1-CT: Clinical Trial Failure Domain

> Initiated: 2026-03-17 | Pipeline code + data loading complete, benchmark design complete

### Step CT-1: Infrastructure ✅ COMPLETE

- [x] CT schema design (2 migrations: 001 initial + 002 expert review fixes)
- [x] 5 pipeline modules: etl_aact, etl_classify, drug_resolver, etl_outcomes, ct_db
- [x] 138 tests passing
- [x] Data download scripts for all 4 sources

### Step CT-2: Data Loading ✅ COMPLETE

- [x] AACT ETL: 216,987 trials, 476K trial-interventions, 372K trial-conditions
- [x] Failure classification (3-tier): 132,925 results (bronze 60K / silver 28K / gold 23K / copper 20K)
- [x] Open Targets: 32,782 intervention-target mappings
- [x] Pair aggregation: 102,850 intervention-condition pairs

### Step CT-3: Enrichment & Resolution ✅ COMPLETE

- [x] Outcome enrichment: +66 AACT p-values, +31,969 Shi & Du SAE records
- [x] Drug resolution Steps 1-2: ChEMBL exact (18K) + PubChem API
- [x] Drug resolution Step 3: Fuzzy matching — 15,616 resolved
- [x] Drug resolution Step 4: Manual overrides — 291 resolved (88 entries used)
- [x] Pair aggregation refresh (post-resolution) — 102,850 pairs
- [x] Post-run coverage analysis — 36,361/176,741 (20.6%) ChEMBL, 27,534 SMILES, 66,393 targets

### Step CT-4: Analysis & Benchmark Design ✅ COMPLETE

- [x] Data quality analysis script (`scripts_ct/analyze_ct_data.py`) — 16 queries, JSON+MD output
- [x] Data quality report (`results/ct/ct_data_quality.md`)
- [x] ML benchmark design (`research/14_ct_ml_benchmark_design.md`)
  - 3 tasks: CT-M1 (binary), CT-M2 (7-way category), CT-M3 (phase transition, deferred)
  - 6 split strategies, 3 models (XGBoost, MLP, GNN+Tabular)
  - 3 experiments: negative source, generalization, temporal
- [x] LLM benchmark design (`research/15_ct_llm_benchmark_design.md`)
  - 4 levels: CT-L1 (5-way MCQ), CT-L2 (extraction), CT-L3 (reasoning), CT-L4 (discrimination)
  - 5 models, anti-contamination analysis

### Step CT-5: ML Export & Splits ✅ COMPLETE

- [x] CT export module (`src/negbiodb_ct/ct_export.py`) — 1,100 lines, 3 loaders + 6 splits + M1 builder + leakage report
- [x] CTO success trials extraction — 5,611 clean pairs (7,835 conflict pairs removed from both sides)
- [x] 6 split strategies (random, cold_drug, cold_condition, temporal, scaffold, degree_balanced)
- [x] CT-M1: balanced (11,222), realistic (36,957), smiles_only (3,878)
- [x] CT-M2: 112,298 results (non-copper, 7-way classification)
- [x] Leakage report: cold leakage=0, M1 conflict-free verified
- [x] 52 tests passing (`tests/test_ct_export.py`)

### Step CT-6: ML Baseline Experiments ✅ COMPLETE

- [x] Feature encoding (`ct_features.py`): drug/condition/trial features, 1044-dim M1, 1066-dim M2
- [x] Model definitions (`ct_models.py`): CT_MLP, CT_GNN_Tab
- [x] Training harness (`train_ct_baseline.py`): XGBoost/MLP/GNN, M1 binary + M2 7-way
- [x] Exp CT-1 negative generation (`prepare_ct_exp_data.py`)
- [x] Results aggregation (`collect_ct_results.py`) + inflation analysis
- [x] SLURM infrastructure: 36 jobs × 3 seeds = 108 total
- [x] 282 CT tests passing, 3-round review (16 bugs found and fixed)
- [x] 108/108 runs complete (6 M1-temporal = NaN due to single-class val set — expected)
- [x] Results collected: `ct_table_m1.csv`, `ct_table_m2.csv`, `ct_exp_ct1_inflation.md`

> **Key CT-6 findings:** NegBioDB negatives trivially solvable (AUROC~1.0); degree-matched hardest (0.76-0.84); Exp CT-1 inflation: -0.156 to -0.242 (XGBoost→GNN). M2: XGBoost best (mF1=0.51), scaffold/temporal splits hardest.

### Step CT-7: LLM Benchmark Execution ✅ 80/80 COMPLETE

- [x] CT-L1/L2/L3/L4 dataset construction scripts (`build_ct_l1/l2/l3/l4_dataset.py`)
- [x] CT prompt templates (`src/negbiodb_ct/llm_prompts.py`) + evaluation functions (`src/negbiodb_ct/llm_eval.py`)
- [x] SLURM infrastructure: `submit_ct_llm_all.sh`, local + OpenAI + Gemini + Anthropic runners
- [x] Results aggregation (`collect_ct_llm_results.py`)
- [x] 5 models × 4 tasks × 4 configs = 80 runs complete
- [x] L3 LLM-as-Judge: GPT-4o-mini — all 20 runs judged
- [x] L2 field_f1_micro bug fixed (gold_extraction nesting + list response parsing)
- [x] L3 ceiling effect identified: GPT-4o-mini judge gives 4.4-5.0/5.0 (too lenient)
- [x] Gemini rate limit resolved: Tier 1 pay-as-you-go (200 RPM, no RPD cap)

> **Key CT-7 findings (5/5 models):** Gemini best on L1 (0.68) and L4 (MCC 0.56). L1 accuracy 0.63-0.68, L2 field_f1 0.48-0.81, L4 MCC 0.48-0.56 — meaningful discrimination unlike DTI-L4 (near random). L3 judge ceiling: GPT-4o-mini gives 4.4-5.0/5.0.

---

## Phase 1-PPI: Protein-Protein Interaction Domain

> Initiated: 2026-03-18 | Phase C (ML benchmark) complete, training pending

### Step PPI-A: Infrastructure ✅ COMPLETE

- [x] PPI schema design (1 migration: 001 initial)
- [x] 4 ETL modules: etl_intact, etl_huri, etl_humap, etl_string
- [x] Protein mapper with UniProt validation
- [x] 176 tests passing

### Step PPI-B: Data Loading ✅ COMPLETE

- [x] IntAct: 779 curated non-interactions (gold 69 / silver 710)
- [x] HuRI: 500,000 Y2H systematic screen negatives (gold)
- [x] hu.MAP: 1,228,891 ML-derived negatives from ComplexPortal (silver)
- [x] STRING: 500,000 zero-score pairs (bronze)
- [x] Pair aggregation: 2,220,786 pairs (8,800 multi-source overlaps)
- [x] DB: 849 MB (`data/negbiodb_ppi.db`), 18,412 proteins

### Step PPI-C: ML Export & Benchmark ✅ COMPLETE (2026-03-21)

- [x] UniProt sequence fetch: 18,412 proteins annotated
- [x] Export module (`src/negbiodb_ppi/export.py`): 4 split strategies + negative export + positive merge
- [x] 4 splits: random, cold_protein, cold_both (Metis graph partition), degree_balanced
- [x] Positive collection: 61,728 HuRI positives (578 conflicts removed from both sides)
- [x] M1 balanced (123,456), realistic (679,008), + 2 Exp 1 controls + DDB
- [x] 3 models: SiameseCNN, PIPR, MLPFeatures (67-dim hand-crafted)
- [x] Training harness (`train_baseline.py`): 18 runs/seed (9 baseline + 6 Exp1 + 3 Exp4)
- [x] Results collection (`collect_results.py`) + inflation analysis
- [x] SLURM infrastructure: `train_ppi_baseline.slurm`, `submit_ppi_all.sh`
- [x] 285 PPI tests passing, 3-agent audit (1 cosmetic fix)
- [x] Data transfer to Cayuga + 54/54 jobs complete (3 seeds × 18 configs)
- [x] Degree leakage bug found + fixed (recompute degree from merged graph)
- [x] Results collected: `results/ppi/table1_aggregated.csv`

> **Key PPI-C findings:** PIPR cold_both catastrophic (AUROC 0.409±0.077, below random). MLPFeatures cold_both robust (0.950±0.021). Negative source effect MODEL-DEPENDENT: sequence models +6-9% inflation (same as DTI), MLPFeatures REVERSED -5% to -19% (NegBioDB harder). DDB ≈ random.

### Step PPI-D: LLM Benchmark ✅ 80/80 COMPLETE (2026-03-22)

- [x] PPI schema migration 002 (function descriptions, GO terms, PMID abstracts table)
- [x] UniProt function descriptions fetch (15,722/18,412 = 85.4% with function)
- [x] PubMed abstract fetch for IntAct PMIDs (65 unique PMIDs → L2 fallback to constructed evidence)
- [x] Design document (`research/16_ppi_llm_benchmark_design.md`, 1,005 lines)
- [x] PPI LLM modules (`llm_prompts.py`, `llm_eval.py`, `llm_dataset.py`) — 101 tests passing
- [x] 4 dataset builders: PPI-L1 (1,200), PPI-L2 (500), PPI-L3 (200), PPI-L4 (500)
- [x] Inference runner + L3 judge + results collector + SLURM scripts (4 templates + submit)
- [x] 80 jobs submitted (SLURM IDs 2708050-2708129)
- [x] 80/80 complete: all 5 models × 4 levels × 4 configs
- [x] L3 judge scoring (Gemini 2.5 Flash) — 20/20 runs judged
- [x] Results collection and analysis — `ppi_llm_summary.csv` + `.md`
- [x] Contamination vs popularity analysis — all 5 models show true contamination
- [x] Expert panel review fixes (protein_mapper race condition, STRING file glob, collector regex)

> **Key PPI-D findings (5 models):**
> - **L1 3-shot trivially solvable** (~1.0 acc for 4/5 models; Qwen ~0.83)
> - **L2 near-perfect extraction** (entity_f1 ~1.0 for all models)
> - **L3 NO ceiling effect** (1.0-4.68 range). Haiku zero-shot best (4.68). Structural reasoning hardest dimension.
> - **L4 moderate discrimination** (MCC 0.33-0.44, between DTI ≤0.18 and CT ~0.5)
> - **L4 MASSIVE contamination** — gap 0.40-0.59 (threshold 0.15). True contamination confirmed (not popularity confound).
> - **100% evidence hallucination** — all models cite evidence for untested pairs

---

## Phase 1b: Post-Submission Expansion (Months 3-6)

### Data Expansion (if not at 10K+ for submission)
- [ ] Complete PubChem BioAssay extraction (full confirmatory set)
- [ ] LLM text mining pipeline activation (PubMed abstracts)
- [ ] Supplementary materials table extraction (pilot)

### Benchmark Refinement
- [ ] Add remaining ML and LLM baseline models
- [ ] Complete all 12 validation experiments (8 ML + 4 LLM)
- [ ] Complete LLM tasks L5, L6 datasets
- [ ] Add flagship LLM evaluations (GPT-4, Claude)
- [ ] Build public leaderboard (simple GitHub-based, separate ML and LLM tracks)

### Perspective Paper (Parallel Track)
- [ ] Write "Publication Bias in DTI Prediction" perspective
- [ ] Target: Briefings in Bioinformatics or Drug Discovery Today
- [ ] Cite NegBioDB as the solution

---

## Phase 2: Community & Platform (Months 6-18)

### 2.1 Platform Development
- [ ] Web interface (search, browse, download)
- [ ] Python library: `pip install negbiodb`
- [ ] REST API with tiered access
- [ ] Community submission portal with controlled vocabularies
- [ ] Leaderboard system

### 2.2 Community Building
- [ ] GitHub repository with documentation and tutorials
- [ ] Partner with SGC and Target 2035/AIRCHECK for data access
- [ ] Engage with DREAM challenge community
- [ ] Tutorial at relevant workshop
- [ ] Researcher incentive design (citation credit, DOI per submission)

### 2.3 Publication Strategy

| Target | Type |
|--------|------|
| ArXiv preprint | Establish priority |
| Datasets & Benchmarks Track | Primary benchmark paper |
| Perspective paper | Publication bias in DTI |
| J. Cheminformatics / Nature Sci Data | Database descriptor |
| NAR Database Issue | Database recognition |

### 2.4 Funding Applications

| Month | Target | Amount |
|-------|--------|--------|
| 3-6 | NIH PAR-23-236 (R24) | Up to $350K/yr × 4yr |
| 6-9 | CZI Open Science | Varies |
| 12-18 | NSF IDSS / Cyberinfrastructure | Varies |

---

## Schema Design

### Common Layer

```
NegativeResult {
  id: UUID
  compound_id: InChIKey + ChEMBL ID + PubChem CID
  target_id: UniProt ID + ChEMBL Target ID

  // Core negative result
  result_type: ENUM [hard_negative, conditional_negative, methodological_negative,
                     hypothesis_negative, dose_time_negative]
  confidence_tier: ENUM [gold, silver, bronze, copper]

  // Quantitative evidence
  activity_value: FLOAT (IC50, Kd, Ki, EC50)
  activity_unit: STRING
  activity_type: STRING
  pchembl_value: FLOAT
  inactivity_threshold: FLOAT
  max_concentration_tested: FLOAT

  // Assay context (BAO-based)
  assay_type: BAO term
  assay_format: ENUM [biochemical, cell-based, in_vivo]
  assay_technology: STRING
  detection_method: STRING
  cell_line: STRING (if cell-based)
  organism: STRING

  // Quality metrics
  z_factor: FLOAT
  ssmd: FLOAT
  num_replicates: INT
  screen_type: ENUM [primary_single_point, confirmatory_dose_response,
                     counter_screen, orthogonal_assay]

  // Provenance
  source_db: STRING (PubChem, ChEMBL, literature, community)
  source_id: STRING (assay ID, paper DOI)
  extraction_method: ENUM [database_direct, text_mining, llm_extracted,
                           community_submitted]
  curator_validated: BOOLEAN

  // Target context (DTO-based)
  target_type: DTO term
  target_family: STRING (kinase, GPCR, ion_channel, etc.)
  target_development_level: ENUM [Tclin, Tchem, Tbio, Tdark]

  // Metadata
  created_at: TIMESTAMP
  updated_at: TIMESTAMP
  related_positive_results: [UUID] (links to known actives for same target)
}
```

### Biology/DTI Domain Layer

```
DTIContext {
  negative_result_id: UUID (FK)
  binding_site: STRING (orthosteric, allosteric, unknown)
  selectivity_data: BOOLEAN (part of selectivity panel?)
  species_tested: STRING
  counterpart_species_result: STRING (active in other species?)
  cell_permeability_issue: BOOLEAN
  compound_solubility: FLOAT
  compound_stability: STRING
}
```

---

## Benchmark Design (NegBioBench) — Dual ML + LLM Track

### Track A: Traditional ML Tasks

| Task | Input | Output | Primary Metric |
|------|-------|--------|----------------|
| **M1: DTI Binary Prediction** | (compound SMILES, target sequence) | Active / Inactive | LogAUC[0.001,0.1], AUPRC |
| **M2: Negative Confidence Prediction** | (SMILES, sequence, assay features) | gold/silver/bronze/copper | Weighted F1, MCC |
| **M3: Activity Value Regression** | (SMILES, sequence) | pIC50 / pKd | RMSE, R², Spearman ρ |

**ML Baselines:** DeepDTA, GraphDTA, DrugBAN, RF, XGBoost, DTI-LM, EviDTI

### Track B: LLM Tasks

| Task | Input | Output | Metric | Eval Method |
|------|-------|--------|--------|-------------|
| **L1: Negative DTI Classification** | Natural language description | Active/Inactive/Inconclusive/Conditional (MCQ) | Accuracy, F1, MCC | Automated |
| **L2: Negative Result Extraction** | Paper abstract | Structured JSON (compound, target, outcome) | Schema compliance, Entity F1, STED | Automated |
| **L3: Inactivity Reasoning** | Confirmed negative + context | Scientific explanation | 4-dim rubric (accuracy, reasoning, completeness, specificity) | LLM-as-Judge + human sample |
| **L4: Tested-vs-Untested Discrimination** | Compound-target pairs | Tested/Untested + evidence | Accuracy, F1, evidence quality | Automated + spot-check |
| **L5: Assay Context Reasoning** | Negative result + condition changes | Prediction + reasoning per scenario | Prediction accuracy, reasoning quality | LLM-as-Judge |
| **L6: Evidence Quality Assessment** | Negative result + metadata | Confidence tier + justification | Tier F1, justification quality | Automated + LLM-judge |

**LLM Baselines (Phase 1 — Free):** Gemini 2.5 Flash, Llama 3.3, Mistral 7B, Phi-3.5, Qwen2.5
**LLM Baselines (Phase 2 — Flagship):** GPT-4, Claude Sonnet/Opus, Gemini Pro
**LLM-as-Judge:** Gemini 2.5 Flash free tier (validated against human annotations)

### Track C: Cross-Track (Future)

| Task | Description |
|------|-------------|
| **C1: Ensemble Prediction** | Combine ML model scores + LLM reasoning — does LLM improve ML? |

### Splitting Strategies (7 total, for Track A)
1. Random (stratified 70/10/20)
2. Cold compound (Butina clustering on Murcko scaffolds)
3. Cold target (by UniProt accession)
4. Cold both (compound + target unseen)
5. Temporal (train < 2020, val 2020-2022, test > 2022)
6. Scaffold (Murcko scaffold cluster-based)
7. DDB — Degree Distribution Balanced (addresses node degree bias)

### Evaluation Metrics (Track A)

| Metric | Type | Role |
|--------|------|------|
| **LogAUC[0.001,0.1]** | Enrichment | **Primary ranking metric** |
| **BEDROC (α=20)** | Enrichment | Early enrichment |
| **EF@1%, EF@5%** | Enrichment | Top-ranked performance |
| **AUPRC** | Ranking | **Secondary ranking metric** |
| **MCC** | Classification | Balanced classification |
| **AUROC** | Ranking | Backward compatibility only (not for ranking) |

### LLM Evaluation Configuration
- **Full benchmark** (5 configs): zero-shot, 3-shot, 5-shot, CoT, CoT+3-shot
- **Must-have** (2 configs): zero-shot, 3-shot only (see research/08 §3)
- **Should-have** (add CoT): 3 configs total for Exp 11 (prompt strategy comparison)
- 3 runs per evaluation, report mean ± std
- Temperature = 0, prompts version-controlled
- Anti-contamination: temporal holdout + paraphrased variants + contamination detection

---

## Phase 3: Scale & Sustainability (Months 18-36)

### 3.1 Data Expansion
- [ ] Expand to 100K+ curated negative DTIs
- [ ] Full LLM-based literature mining pipeline (PubMed/PMC)
- [ ] Supplementary materials table extraction (Table Transformer)
- [ ] Integrate Target 2035 AIRCHECK data as it becomes available
- [ ] Begin Gene Function (KO/KD) negative data collection

### 3.2 Benchmark Evolution (NegBioBench v1.0)
- [ ] Track A expansion: multi-modal integration (protein structures, assay images)
- [ ] Track B expansion: additional tasks — Failure Diagnosis, Experimental Design Critique, Literature Contradiction Detection
- [ ] Track C: Cross-track ensemble evaluation (ML + LLM combined prediction)
- [ ] Specialized bio-LLM evaluations (LlaSMol, BioMedGPT, DrugChat)
- [ ] Regular leaderboard updates (both ML and LLM tracks)

### 3.3 Commercialization
- [ ] Launch tiered API access (free / pro / enterprise)
- [ ] Approach pharma companies for consortium membership
- [ ] Develop analytics dashboard
- [ ] Explore Insight-as-a-Service model

### 3.4 NAR Database Issue Application
- [ ] Contact NAR Executive Editor by July
- [ ] Demonstrate community adoption metrics
- [ ] Publish update paper following January

---

## Phase 4: Domain Expansion (Months 36+)

```
DTI (Phase 1 — COMPLETE)
  │
  ├── Clinical Trial Failure (Phase 1-CT — COMPLETE)
  │     └── 132,925 failure results, 108 ML + 80 LLM runs done
  │
  ├── Protein-Protein Interaction (Phase 1-PPI — COMPLETE)
  │     └── 2.2M negative pairs, 54 ML + 80 LLM runs done
  │
  ├── Gene Function (CRISPR KO/KD negatives)
  │     └── Leverage CRISPR screen data, DepMap
  │
  ├── Chemistry Domain Layer
  │     └── Failed reactions, yield = 0 data
  │
  └── Materials Science Domain Layer
        └── HTEM DB integration, failed synthesis conditions
```

---

## Key Milestones (Revised)

| Milestone | Target Date | Deliverable | Status |
|-----------|------------|-------------|--------|
| Schema v1.0 finalized | Week 2 (Mar 2026) | SQLite schema + standardization pipeline | ✅ Done |
| Data extraction complete | Week 3-4 (Mar 2026) | **30.5M** negative results (far exceeded 10K target) | ✅ Done |
| ML export & splits | Week 3 (Mar 2026) | 6 split strategies + M1 benchmark datasets | ✅ Done |
| ML evaluation metrics | Week 3 (Mar 2026) | 7 metrics, 329 tests | ✅ Done |
| ML baseline infrastructure | Week 4 (Mar 2026) | 3 models + SLURM harness | ✅ Done |
| ML baseline experiments | Week 5 (Mar 2026) | 18/18 runs complete, key findings confirmed | ✅ Done |
| LLM benchmark infrastructure | Week 5 (Mar 2026) | L1–L4 datasets, prompts, eval, SLURM templates | ✅ Done |
| LLM benchmark execution | Week 5-6 (Mar 2026) | 81/81 runs complete (9 models × 4 tasks + configs) | ✅ Done |
| **ArXiv preprint** | **Week 9** | **Priority establishment** |
| **Paper submission** | **Week 11** | **Benchmark paper** |
| Perspective paper submitted | Month 4-6 | Publication bias in DTI |
| Python library v0.1 | Month 8 | `pip install negbiodb` |
| Decision notification | Month 7 | Accept/reject notification |
| Web platform launch | Month 12 | Public access + leaderboard |
| Database descriptor paper | Month 8-12 | J. Cheminformatics or Nature Sci Data |
| NIH R24 funding | Month 12-18 | Multi-year sustainability |
| 100K+ entries | Month 24 | Scale milestone |
| NAR Database Issue | Month 24-30 | Gold standard DB recognition |

---

## Week 6 Go/No-Go Decision Framework

Exp 1 result determines the paper's primary narrative. Check at Week 6:

| Scenario | Exp 1 Result | Action |
|----------|-------------|--------|
| **A: Strong** | NegBioDB > Random by ≥10% (p < 0.05) | Narrative holds: "benchmarks are broken." Add should-have experiments |
| **B: Moderate** | NegBioDB > Random by 3-8% (p < 0.1) | Co-primary with Exp 4 (DDB bias). Reframe: "bias-free evaluation" |
| **C: Null** | NegBioDB ≈ Random | Major pivot: LLM track (L1, L2, L4) as primary contribution. "Database + novel evaluation tasks" framing |

### Milestone Checkpoints

| Week | Checkpoint | Go Criteria | No-Go Action |
|------|-----------|-------------|--------------|
| 2 | Data extraction | ≥10K unique inactive pairs | Add PubChem primary screen (Bronze) for volume |
| 4 | LLM datasets ready | L1 2K + L2 100 + L4 500 | Reduce L2 to 50 (minimum viable) |
| 5 | ML baseline first run | DeepDTA successfully trained | Switch to PyTDC example code |
| **6** | **Exp 1 result** | Significant difference | See scenario table above |
| 8 | Paper draft complete | All figures + tables | Drop all should-have experiments |
| 10 | Submission ready | Croissant valid + HuggingFace live | 1-2 day buffer available |

---

## Risk Assessment (Updated — Expert Panel v6)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Submission deadline tight (11 weeks) | Medium-High | High | Focus on core experiments; ArXiv as fallback |
| **Exp 1 results weak or null** | **Medium** | **Critical** | Week 6 Go/No-Go framework; Exp 4 as backup primary; LLM track pivot |
| **Solo author credibility** | **Medium** | **High** | Dockerfile + full reproducibility scripts + detailed supplement |
| **GPU unavailability** | **Medium** | **High** | Kaggle free (30h/wk) as primary; Colab Pro ($10) fallback |
| HCDT 2.0 license blocks integration | **Confirmed** | Medium | Already mitigated: independently extract from underlying sources |
| **Positive data protocol undefined** | **Resolved** | — | See §Positive Data Protocol (ChEMBL pChEMBL ≥ 6, shared targets) |
| Insufficient data quality | Medium | High | Strict QC pipeline + confidence tiers |
| Low community adoption | Medium | High | TDC-style easy access + workshop tutorials |
| Competitive entry before submission | Low | Medium | First-mover advantage + ArXiv priority |
| **Croissant validation failure** | Low | **Critical** | `mlcroissant` validation; HuggingFace auto-generation as fallback |
| **PubChem OOM during processing** | Medium | Medium | Streaming (chunksize=100K or polars lazy); resolved in pipeline design |
| Funding gap | Medium | High | Multiple funding sources + early commercial track |
| Schema over-engineering | Medium | Medium | Start minimal (SQLite), iterate based on user feedback |
| Pharma resistance to sharing | High | Medium | Start with public data; build trust first |
| ChEMBL CC BY-SA viral clause | Low | Medium | Use CC BY-SA 4.0 for NegBioDB; compatible |

---

## Document Index

| Document | Content |
|----------|---------|
| [research/01_dti_negative_data_landscape.md](research/01_dti_negative_data_landscape.md) | Survey of existing DTI negative data sources |
| [research/02_benchmark_analysis.md](research/02_benchmark_analysis.md) | Analysis of existing DTI benchmarks and their negative handling |
| [research/03_data_collection_methodology.md](research/03_data_collection_methodology.md) | Methodologies for collecting and curating negative data |
| [research/04_publication_commercial_strategy.md](research/04_publication_commercial_strategy.md) | Publication venues, funding, commercialization |
| [research/05_technical_deep_dive.md](research/05_technical_deep_dive.md) | Data access APIs, license analysis, dedup, ML baselines, metrics |
| [research/06_paper_narrative.md](research/06_paper_narrative.md) | Paper title/abstract, competitive positioning |
| [research/07a_llm_benchmark_landscape_survey.md](research/07a_llm_benchmark_landscape_survey.md) | Survey of existing bio/chem LLM benchmarks and evaluation methods |
| [research/07b_llm_benchmark_design.md](research/07b_llm_benchmark_design.md) | LLM benchmark tasks, evaluation methods, dual-track architecture |
| [research/08_expert_review_and_feasibility.md](research/08_expert_review_and_feasibility.md) | Expert review responses, feasibility analysis, concrete decisions |
| [research/09_schema_and_ml_export_design.md](research/09_schema_and_ml_export_design.md) | SQLite schema DDL, ML export patterns, Croissant metadata, Datasheet |
| [research/10_expert_panel_review.md](research/10_expert_panel_review.md) | 6-expert panel review: reviewer, data eng, ML, domain, SW arch, PM |
| [research/11_full_plan_review.md](research/11_full_plan_review.md) | Pre-implementation audit: 16 issues found, feasibility ratings, execution adjustments |
| [research/12_review_findings_summary.md](research/12_review_findings_summary.md) | Schema/pipeline implementation review: 9 issues (3 critical, 3 high, 2 moderate, 1 low) |
| [research/13_clinical_trial_failure_domain.md](research/13_clinical_trial_failure_domain.md) | CT domain design: failure taxonomy, 3-tier detection, pipeline architecture |
| [research/14_ct_ml_benchmark_design.md](research/14_ct_ml_benchmark_design.md) | CT ML benchmark: 3 tasks, 6 splits, 3 models, 3 experiments |
| [research/15_ct_llm_benchmark_design.md](research/15_ct_llm_benchmark_design.md) | CT LLM benchmark: 4 levels, 5 models, contamination analysis |
| [research/16_ppi_llm_benchmark_design.md](research/16_ppi_llm_benchmark_design.md) | PPI LLM benchmark: 4 levels, 5 models, contamination analysis |
| [research/17_ct_expert_panel_review.md](research/17_ct_expert_panel_review.md) | CT domain 6-expert panel review |
| [research/17_ppi_expert_panel_review.md](research/17_ppi_expert_panel_review.md) | PPI domain 6-expert panel review |
