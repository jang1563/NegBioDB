# DTI Benchmark Analysis: How Negatives Are Handled

> In-depth analysis of existing DTI benchmarks and their negative data treatment (2026-03-02)

---

## Executive Summary

Across nearly every major DTI benchmark, "negatives" are either absent, assumed from untested pairs, randomly sampled from unknown interaction space, or generated through computational heuristics that introduce their own biases. Known interactions represent **less than 1%** of all possible drug-target pairs, meaning >99% of the combinatorial space is unlabeled — and most benchmarks treat this unlabeled space as negative.

**Core Problem: The "Untested = Negative" assumption is pervasive and systematically distorts model evaluation.**

---

## Benchmark-by-Benchmark Analysis

### 1. MoleculeNet (Wu et al., 2018)

**Nature:** Molecular property prediction benchmark (not DTI-specific)

| Dataset | Size | Negative Handling | Experimentally Confirmed? |
|---------|------|-------------------|--------------------------|
| PCBA | 438K compounds, 128 assays | HTS inactive | Yes (per assay) |
| HIV | 41K compounds | "Confirmed Inactive" label | Yes |
| BACE | 1,522 compounds | IC50 threshold | Yes |
| PDBbind | 3,040 complexes | **No negatives** | N/A |
| MUV | 93K compounds, 17 targets | Refined nearest-neighbor inactives | Yes (PubChem) |

**Key Criticisms:**
- Nature Communications 2023: "numerous flaws, making it difficult to draw conclusions from method comparisons"
- Contains invalid SMILES (uncharged tetravalent nitrogen)
- AUROC misleading with extreme class imbalance
- Pat Walters (2023): "highly flawed" — combining IC50 from different ChEMBL assays is problematic

---

### 2. TDC — Therapeutics Data Commons (Huang et al., 2021)

**DTI Datasets:**

| Dataset | Pairs | Drugs | Targets | Metric |
|---------|-------|-------|---------|--------|
| DAVIS | 27,621 | 72 | 442 | Kd |
| KIBA | ~118K | 2,111 | 229 | KIBA score |
| BindingDB_Kd | 52,284 | 10,665 | 1,413 | Kd |
| BindingDB_Ki | 375,032 | 174,662 | 3,070 | Ki |
| BindingDB_IC50 | 991,486 | 549,205 | 5,078 | IC50 |

**Negative Generation:**
- `neg_sample` utility: randomly samples unobserved (untested) pairs as negatives
- Assumption: most unknown pairs are true negatives (justified by sparsity, but...)
- **No experimentally confirmed negatives provided**

**Splitting Strategies:** Random, Cold drug, Cold target, Cold drug+target

**Key Criticisms:**
- BMC Biology 2025: "well-trained ML models tend to predict based solely on node degree"
- Scale-free network bias → well-studied drugs/targets predicted as interacting regardless of biology
- DAVIS = kinase only; KIBA also kinase-heavy
- ChEMBL source overlap across datasets → train/test contamination risk

---

### 3. DAVIS Dataset (Davis et al., 2011)

**The only major complete interaction matrix benchmark**

- 72 kinase inhibitors × 442 kinases ≈ 30,056 binding affinity values
- Nearly all pairs actually measured → **experimentally confirmed non-binding pairs exist**
- pKd ≥ 7 commonly used as "active" threshold
- Minimum measured Kd = 10 uM → "inactive" entries are still weak binders

**Limitation:** Kinase targets only. Does not represent the broader druggable proteome.

---

### 4. DUD-E (Mysinger et al., 2012)

- 102 targets, 22,886 actives, ~1.4M decoys (50 per active)
- Decoys: physicochemically property-matched but topologically dissimilar (from ZINC)
- **Decoys are NOT experimentally tested** — assumed inactive

**Severe Biases (PLOS ONE 2019, Chen et al.):**
- **Analog bias**: Actives cluster in chemical space → models learn chemical series, not binding
- **Decoy bias**: Property matching itself creates signal → decoys distinguishable from actives by selection criteria artifacts
- "High performance is not attributable to learning protein-ligand interactions but rather to analog and decoy bias"
- **All virtual screening programs showed dramatic performance drops when biases were removed**

---

### 5. LIT-PCBA (Tran-Nguyen et al., 2020)

**Explicitly designed to fix DUD-E's problems**

- 15 targets
- 7,844 confirmed actives / 407,381 **confirmed inactives** (dose-response)
- **Both actives AND inactives experimentally confirmed** ← key differentiator
- Property-matched + AVE (Asymmetric Validation Embedding) → bias minimization
- Hit rates ~1-2% (realistic)

**Recent Issue (2025):** Data leakage and redundancy discovered — even this carefully designed benchmark is not immune.

---

### 6. BioSNAP (Stanford SNAP)

| Metric | Value |
|--------|-------|
| Drugs | 4,510 |
| Targets | 2,181 |
| Positive DTI | 13,741 |
| Negative DTI | 13,741 (random sampled) |

- Negatives randomly sampled from unobserved DrugBank pairs
- **1:1 balance → unrealistic** (real hit rate 0.1-1%)
- All negatives are assumed (untested), not experimentally confirmed
- Severe node degree bias

---

### 7. DrugBank

- **Positive-only database.** Does not represent non-interactions
- Missing pair = "inactive" OR "never studied" — indistinguishable
- DrugBank, KEGG, NDF-RT overlap < 50% → all databases fundamentally incomplete

---

## Comparison Summary

| Benchmark | Size | Negative Type | Experimentally Confirmed? | Primary Bias |
|-----------|------|---------------|--------------------------|-------------|
| MoleculeNet/PCBA | 438K, 128 assays | Tested inactives | Yes (per assay) | Extreme imbalance |
| MoleculeNet/PDBbind | 3,040 | None | N/A | No non-binders |
| TDC/DAVIS | 27,621 | Threshold (Kd) | Partially | Kinase only |
| TDC/KIBA | ~118K | Threshold (KIBA) | Partially | Kinase-heavy |
| DUD-E | 22,886 + 1.4M decoys | Property-matched decoys | **No** | Analog + decoy bias |
| **LIT-PCBA** | 7,844 + 407K | **Dose-response confirmed** | **Yes** | 15 targets only |
| BioSNAP | 13,741 + 13,741 | Random from unobserved | **No** | Degree bias |
| DrugBank | ~14K DTI | None | N/A | Positive only |

---

## Key Critical Papers (2023-2025)

### Negative Sampling and Node Degree Bias
- **BMC Biology 2025**: ML models learn node degree, not biology, with random negative sampling. Proposed Degree Distribution Balanced (DDB) sampling.
  - URL: https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-025-02231-w

### General Benchmark Problems
- **Expert Opinion Drug Discovery 2024**: Analyzed 3,286 DTI papers. Experimental validation "relatively rare."
  - URL: https://www.tandfonline.com/doi/full/10.1080/17460441.2024.2430955
- **Practical Cheminformatics 2023** (Pat Walters): Most widely used ML benchmarks are "highly flawed."
  - URL: http://practicalcheminformatics.blogspot.com/2023/08/we-need-better-benchmarks-for-machine.html
- **Nature Communications 2025**: Evidential deep learning showed performance gap between assumed vs. validated negatives.
  - URL: https://www.nature.com/articles/s41467-025-62235-6

### The "Assumed Negative" Problem — Scale

- Known interactions < 1% of all possible drug-target pairs
- ~10,000 drugs × ~3,000 targets = ~30M pairs → known = hundreds of thousands → **99%+ unlabeled**
- "False positive" predictions are frequently actual undiscovered true interactions
- Mitigation attempts: Random, BN (Balanced Negative), DDB, RNIDTP, LIT-PCBA-style confirmation, GAN synthetic — each introduces its own biases

---

## Implications for NegBioDB Benchmark Design

1. **Use only experimentally confirmed negatives** (LIT-PCBA approach)
2. **Minimize "untested = negative" assumption** — distinguish via confidence tiers
3. **Address node degree bias** — DDB sampling or degree-aware evaluation
4. **Diverse target classes** — overcome kinase/GPCR skew
5. **Record assay context** — same compound can be active/inactive depending on assay
6. **Multiple splitting strategies** — random, cold drug, cold target, temporal
7. **Realistic hit rates** — not 1:1 balance, but real-world 0.1-1% active rates

---

## Sources

- MoleculeNet: https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/
- TDC: https://tdcommons.ai/multi_pred_tasks/dti/ | https://arxiv.org/abs/2102.09548
- DUD-E: https://pubs.acs.org/doi/10.1021/jm300687e
- DUD-E Bias: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220113
- LIT-PCBA: https://pubs.acs.org/doi/10.1021/acs.jcim.0c00155
- LIT-PCBA Leakage: https://arxiv.org/html/2507.21404v1
- BioSNAP: https://snap.stanford.edu/biodata/datasets/10002/10002-ChG-Miner.html
- Comprehensive DTI Review 2025: https://www.sciencedirect.com/science/article/pii/S2001037025005719
- Komet/LCIdb: https://pubs.acs.org/doi/10.1021/acs.jcim.4c00422
