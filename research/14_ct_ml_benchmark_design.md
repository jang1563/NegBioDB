# 14. CT Domain ML Benchmark Design

## 1. Overview

The NegBioDB-CT ML benchmark evaluates whether machine learning models can predict clinical trial failure outcomes. It mirrors the DTI domain's benchmark structure (3 models × 6 splits × 2 datasets) but adapts to the unique characteristics of clinical trial data: categorical outcomes, mixed feature types (molecular + clinical), and strong temporal structure.

### Database State (as of 2026-03-18)
- **132,925 failure results** across 28,135 unique trials
- **102,850 intervention-condition pairs**
- **Tier distribution:** gold 23,570 (17.7%) / silver 28,505 (21.4%) / bronze 60,223 (45.3%) / copper 20,627 (15.5%)
- **Drug resolution:** ~13% with ChEMBL ID (in progress, expected ~35-45% when complete)
- **Category distribution:** efficacy 42.6% / enrollment 23.0% / other 21.0% / strategic 7.1% / safety 3.9% / design 1.8% / regulatory 0.7% / PK 0.0%

---

## 2. Task Definitions

### CT-M1: Drug-Condition Failure Prediction (Binary)

**Question:** Given a drug and a disease indication, will the clinical trial fail?

| Aspect | Details |
|--------|---------|
| Input | Drug features + condition features + phase |
| Output | Binary: failure (0) vs success (1) |
| Negative class | NegBioDB-CT failure records (silver+gold, ≥52K pairs) |
| Positive class | CTO success trials (labels=1.0) matched to DB, OR DrugBank/Open Targets approved drug-condition pairs |
| Minimum tier | Silver (quantitative p-value evidence) |
| DTI analog | M1 (active/inactive classification) |

**Positive Class Acquisition (separate script required):**
1. Load CTO parquet → filter `labels=1.0` (success)
2. Match NCT IDs against `clinical_trials.source_trial_id`
3. Expand to (intervention_id, condition_id) pairs via junction tables
4. Validate: exclude pairs that also appear in failure results (contradictions)
5. Estimated yield: ~100K-200K success pairs (CTO has ~900K rows, ~50% success)

**Alternative positive source:** DrugBank/Open Targets approved drug-indication pairs. Advantage: high confidence. Disadvantage: only covers late-stage approved drugs.

**Experiment design:**
- **Balanced (1:1):** Equal success/failure pairs, sampled to match phase distribution
- **Realistic:** Natural success/failure ratio by phase (~50% Phase III, ~70% Phase II failure rate)

### CT-M2: Failure Category Classification (7-way)

**Question:** Given that a drug-condition trial failed, what is the failure category?

| Aspect | Details |
|--------|---------|
| Input | Drug features + condition features + trial features |
| Output | 7-class: efficacy / enrollment / safety / strategic / design / regulatory / other |
| Data source | All failure results EXCLUDING copper tier (copper = generic "other") |
| Effective data | 112,298 records (132,925 - 20,627 copper) |
| PK excluded | 0 records in database, biologically valid but undetectable from trial-level data |

**Class weights (inverse frequency):**

| Category | Count | Weight |
|----------|-------|--------|
| efficacy | 56,588 | 1.0x |
| enrollment | 30,567 | 1.9x |
| other | 7,277 (non-copper) | 7.8x |
| strategic | 9,400 | 6.0x |
| safety | 5,164 | 11.0x |
| design | 2,377 | 23.8x |
| regulatory | 925 | 61.2x |

Note: "other" after removing copper is 7,277 (27,904 - 20,627). Regulatory class (925) is very small — may need upsampling or merging for some models.

### CT-M3: Phase Transition Prediction

**Question:** Given a drug's result at phase N, will it advance to phase N+1?

| Aspect | Details |
|--------|---------|
| Input | Drug features + condition features + current phase + outcome data |
| Output | Binary: advances / does not advance |
| Data source | Trials with known phase progression history |
| Complexity | Requires linking same drug-condition pairs across phases |

This is the most clinically actionable task but requires complex data engineering (tracking the same drug-condition pair across multiple trials at different phases). **Deferred to post-M1/M2 implementation.**

---

## 3. Feature Engineering

### 3.1 Drug Features

| Feature | Source | Type | Availability |
|---------|--------|------|-------------|
| Morgan fingerprint (1024-bit) | `canonical_smiles` via RDKit | Binary vector | Only small molecules with SMILES (~12K/130K drugs) |
| Molecular properties (MW, LogP, HBD, HBA, PSA, RotBonds) | RDKit from SMILES | Continuous | Same as above |
| Molecular type | `molecular_type` column | One-hot (8 types) | ~18K interventions |
| Target count | `intervention_targets` JOIN count | Integer | ~18K with ChEMBL resolution |
| Drug degree | `intervention_condition_pairs.intervention_degree` | Integer | All drugs in pairs table |

**Biologics handling:** Monoclonal antibodies, peptides, and other biologics lack SMILES. They are represented by molecular_type one-hot + target count + drug degree only. This creates a heterogeneous feature space — models must handle missing SMILES gracefully (zero-pad fingerprints or separate encoding path).

### 3.2 Condition Features

| Feature | Source | Type |
|---------|--------|------|
| Therapeutic area | `conditions.therapeutic_area` | One-hot |
| Condition frequency | COUNT from `trial_conditions` | Log-scale integer |
| Condition degree | `intervention_condition_pairs.condition_degree` | Integer |

Note: `conditions.icd10_code` is sparsely populated — not reliable for encoding. `conditions.mesh_id` is similarly sparse. Therapeutic area one-hot is the most reliable condition feature.

### 3.3 Trial Features (CT-M2 only)

| Feature | Source | Type |
|---------|--------|------|
| Phase | `clinical_trials.trial_phase` | One-hot (8 levels) |
| Randomized | `clinical_trials.randomized` | Binary |
| Blinding | `clinical_trials.blinding` | One-hot |
| Control type | `clinical_trials.control_type` | One-hot |
| Enrollment | `clinical_trials.enrollment_actual` | Log-scale continuous |
| Sponsor type | `clinical_trials.sponsor_type` | One-hot |
| Has results | `clinical_trials.has_results` | Binary |

---

## 4. Split Strategies

### 4.1 Six Split Strategies

| Split | Strategy | Train | Val | Test | Purpose |
|-------|----------|-------|-----|------|---------|
| random | Random 70/10/20 | ~70% | ~10% | ~20% | Baseline |
| cold_drug | Group by drug identity | All pairs for unseen drugs | Same | Same | New molecule generalization |
| cold_condition | Group by condition | All pairs for unseen conditions | Same | Same | New disease generalization |
| temporal | Year-based cutoffs | ≤2017 | 2018-2019 | 2020+ | Prospective validation |
| scaffold | Murcko scaffold groups | Same scaffold → same fold | Same | Same | Chemical series generalization |
| degree_balanced | Stratified by (drug_degree, condition_degree) bins | Balanced sampling | Same | Same | Hub bias control |

### 4.2 Temporal Split Design

Based on actual data distribution (by start_date year):

| Partition | Years | Records | % | Notes |
|-----------|-------|---------|---|-------|
| Train | ≤ 2017 | ~52,100 | 39% | Stable historical data |
| Validation | 2018-2019 | ~16,700 | 13% | Pre-COVID |
| Test | 2020+ | ~30,400 | 23% | Includes COVID spike (2020: 12,600) |
| No date | N/A | ~33,700 | 25% | Excluded from temporal split |

**COVID-19 concern:** 2020 has 12,600 results (2x normal year), dominated by COVID trials. Options:
1. **Include as-is** — tests model robustness to distribution shift (recommended for main analysis)
2. **Exclude COVID condition** — sensitivity analysis, shows non-pandemic performance
3. **Report both** — main + COVID-excluded as supplementary

**Recommendation:** Option 3 (report both). The COVID spike is a real-world phenomenon that tests model generalizability.

### 4.3 Cold Split Identity

- **cold_drug:** Group by `interventions.intervention_id` (or `inchikey_connectivity` if available)
- **cold_condition:** Group by `conditions.condition_id` (or `conditions.mesh_id` if populated)
- **scaffold:** Murcko scaffold computed from SMILES via RDKit — only applicable to small molecules with SMILES

---

## 5. Model Architecture

### 5.1 Model Selection

| Model | Type | Input | DTI Analog | Rationale |
|-------|------|-------|------------|-----------|
| XGBoost | Gradient boosting | Tabular features | None (new) | Strong tabular baseline, handles mixed features |
| MLP | Neural network | Concatenated vectors | DeepDTA-like | Neural baseline for continuous features |
| GNN+Tab | Graph neural net + tabular | Drug graph + condition/trial vectors | GraphDTA-like | Leverages molecular structure |

**Why not DeepDTA/DrugBAN directly?**
DTI models take (SMILES, protein_sequence) as input. CT domain's "target" is a disease (condition), not a protein. While `intervention_targets` can bridge to proteins, this adds complexity and limits to drugs with known targets. Better to design CT-native architectures.

### 5.2 XGBoost Configuration

```python
params = {
    "objective": "binary:logistic",  # CT-M1
    # "objective": "multi:softmax", "num_class": 7,  # CT-M2
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 500,
    "early_stopping_rounds": 20,
    "eval_metric": "logloss",
    "scale_pos_weight": auto,  # class imbalance
}
```

### 5.3 MLP Configuration

```python
# Drug encoder: FP(1024) + MolProps(6) + MolType(8) + degree(1) = 1039 dim
# Condition encoder: TherapArea(~20) + freq(1) + degree(1) = ~22 dim
# Trial encoder (M2): Phase(8) + design(~10) + enrollment(1) = ~19 dim
# Concat → FC(512) → ReLU → FC(256) → ReLU → FC(1 or 7)
```

### 5.4 GNN+Tabular Configuration

```python
# Drug: SMILES → RDKit molecular graph → GCN (3 layers, 128 dim) → readout
# Condition: tabular features → FC(64)
# Concat → FC(256) → ReLU → FC(1 or 7)
# Only for drugs with valid SMILES (~12K/130K interventions)
```

---

## 6. Evaluation Metrics

### CT-M1 (Binary Classification)

| Metric | Description | Primary? |
|--------|-------------|----------|
| AUROC | Area under ROC curve | Yes |
| AUPRC | Area under PR curve (failure = positive) | Yes |
| MCC | Matthews correlation coefficient | Yes |
| LogAUC[0.001, 0.1] | Log-scale AUC (DTI compatibility) | No (supplementary) |
| Accuracy | Overall accuracy | No |
| F1 | Binary F1 score | No |

### CT-M2 (7-way Classification)

| Metric | Description | Primary? |
|--------|-------------|----------|
| Macro F1 | Unweighted average F1 across 7 classes | Yes |
| Weighted F1 | Frequency-weighted F1 | Yes |
| MCC | Multiclass MCC | Yes |
| Per-class accuracy | Breakdown by category | Supplementary |
| Confusion matrix | Full 7×7 confusion | Supplementary |

---

## 7. Experimental Design

### Exp CT-1: Negative Source Comparison (parallels DTI Exp 1)

Compare models trained on NegBioDB-CT curated failures vs random negative sampling:

| Negative Source | Description | Expected Bias |
|-----------------|-------------|---------------|
| NegBioDB-CT silver+gold | Curated failures with quantitative evidence | Low bias, high quality |
| Random untested | Drug-condition pairs never tested together | High bias (includes future successes) |
| Degree-matched untested | Sampled to match NegBioDB degree distribution | Intermediate |

**Hypothesis:** NegBioDB-CT negatives should give lower but more realistic performance than random negatives (same pattern as DTI where degree_matched inflated LogAUC by +0.112).

### Exp CT-2: Generalization Comparison

| Comparison | Splits | Question |
|------------|--------|----------|
| Random vs cold_drug | random, cold_drug | Can model predict for unseen drugs? |
| Random vs cold_condition | random, cold_condition | Can model predict for unseen diseases? |
| Random vs temporal | random, temporal | Can model predict future failures? |

**Hypothesis:** cold_drug performance will degrade most (like DTI cold_target was catastrophic). Temporal split tests real-world utility.

### Exp CT-3: Temporal Predictive Power (CT-unique)

Train on historical data (≤2017), predict future failures (2020+):
- Does temporal prediction work at all?
- How does COVID-19 distribution shift affect performance?
- Is there temporal degradation (2020 vs 2023 accuracy)?

---

## 8. Dataset Construction Pipeline

### 8.1 Export Module: `src/negbiodb_ct/ct_export.py`

Analogous to `src/negbiodb/export.py`, this module will:
1. Query `trial_failure_results` + `interventions` + `conditions` + `clinical_trials`
2. Compute features (fingerprints, properties, one-hot encodings)
3. Generate splits (6 strategies)
4. Export to parquet + CSV

### 8.2 Positive Class Script: `scripts_ct/extract_success_trials.py`

1. Load CTO parquet → filter success labels
2. Match against DB clinical_trials
3. Expand to (intervention, condition) pairs
4. Output: `exports/ct/ct_success_pairs.parquet`

### 8.3 Training Script: `scripts_ct/train_ct_baseline.py`

Adapted from `scripts/train_baseline.py`:
- Same training loop (early stopping, checkpoint, logging)
- Modified feature encoding (tabular + optional graph)
- 7 metrics for M1, confusion matrix for M2

---

## 9. Implementation Priority

| Priority | Task | Prerequisite |
|----------|------|-------------|
| P0 | Extract CTO success trials | Drug resolution complete |
| P1 | CT-M1 with XGBoost (random split) | P0 |
| P2 | CT-M1 with all 6 splits | P1 |
| P3 | CT-M2 with XGBoost | P1 |
| P4 | MLP and GNN models | P2 |
| P5 | Full Exp CT-1, CT-2, CT-3 | P4 |
| Deferred | CT-M3 (phase transition) | Multi-phase data engineering |

---

## 10. Expected Outcome

Based on DTI domain experience:
- **Random split** will show moderate performance (AUROC ~0.7-0.8)
- **Cold splits** will degrade significantly (especially cold_drug)
- **Temporal split** will be most informative for clinical utility
- **NegBioDB-CT vs random negatives** will show inflation similar to DTI (+0.05-0.15 AUROC)
- **CT-M2** (7-way) will be challenging due to class imbalance — regulatory class accuracy likely near random
