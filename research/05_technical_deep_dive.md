# Technical Deep Dive: Data Access, Licensing, Deduplication & Evaluation

> Detailed technical specifications for NegBioDB implementation (2026-03-02)

---

## 1. Data Source Access — Exact Methods

### 1.1 PubChem BioAssay (Primary Negative Source)

**API:** PubChem PUG REST + PUG View

**Extraction Strategy: Confirmatory Dose-Response Inactives**

```
# Step 1: Find confirmatory bioassays (not primary screens)
GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/type/confirmatory/aids/JSON

# Step 2: Get inactive results for a specific assay (AID)
GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{AID}/CSV?sid=inactive

# Step 3: Get compound details for inactive SIDs
GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{CID}/property/InChIKey,CanonicalSMILES,MolecularFormula/JSON
```

**Key Parameters:**
- `activity_outcome=inactive` for filtering inactive results
- Confirmatory assays have `assay_type=confirmatory` (dose-response, not single-point)
- Rate limit: 5 requests/second (no API key needed)
- Bulk download via PubChem FTP: `ftp://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/`

**Filtering Priority:**
1. Confirmatory dose-response assays (highest value)
2. Counter-screens (orthogonal validation)
3. Primary screens with multi-concentration data
4. Primary single-point screens (lowest value — only at Bronze tier)

**Volume Estimate:** ~50K+ confirmatory inactive dose-response records; millions of primary screen inactives

**Output Format:** CSV/JSON with SID, CID, activity outcome, activity values, assay conditions

**License:** Public domain (US government work, no restrictions)

---

### 1.2 ChEMBL (v35+)

**API:** ChEMBL REST API (https://www.ebi.ac.uk/chembl/api/data/)

**Extraction Strategy: Explicit "Not Active" + Low pChEMBL**

```
# Method 1: activity_comment based
GET https://www.ebi.ac.uk/chembl/api/data/activity.json?activity_comment__icontains=Not%20Active&limit=1000

# Method 2: pChEMBL threshold based
GET https://www.ebi.ac.uk/chembl/api/data/activity.json?pchembl_value__lt=5&limit=1000

# Method 3: SQL query on downloaded ChEMBL (recommended for bulk)
SELECT
  a.molregno,
  a.standard_type,
  a.standard_value,
  a.standard_units,
  a.pchembl_value,
  a.activity_comment,
  cs.canonical_smiles,
  cs.standard_inchi_key,
  ass.assay_type,
  ass.description AS assay_description,
  td.pref_name AS target_name,
  td.chembl_id AS target_chembl_id,
  cp.accession AS uniprot_id
FROM activities a
JOIN compound_structures cs ON a.molregno = cs.molregno
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary td ON ass.tid = td.tid
LEFT JOIN target_components tc ON td.tid = tc.tid
LEFT JOIN component_sequences cp ON tc.component_id = cp.component_id
WHERE
  (a.activity_comment IN ('Not Active', 'Inactive', 'inactive', 'Not active')
   OR a.pchembl_value < 5)
  AND a.data_validity_comment IS NULL  -- exclude flagged data
  AND a.standard_type IN ('IC50', 'Ki', 'Kd', 'EC50')
ORDER BY a.pchembl_value ASC;
```

**Key Notes:**
- ChEMBL 37+ will standardize activity_comment to "Not active" (currently inconsistent)
- `data_validity_comment IS NULL` excludes records flagged for quality issues
- Download full MySQL/PostgreSQL dump for bulk processing: https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/
- ~133K explicitly labeled inactive records; ~500K+ with pChEMBL < 5

**Output Format:** SDF, CSV, JSON via API; SQL dump for bulk

**License:** CC BY-SA 3.0 → **Viral clause: derived works must also be CC BY-SA**

---

### 1.3 DAVIS Dataset (Gold Standard Complete Matrix)

**Access:** Direct download from TDC or supplementary materials
```
# Via TDC Python library
from tdc.multi_pred import DTI
data = DTI(name='DAVIS')
df = data.get_data()  # Returns compound SMILES, target sequence, Kd value
```

**Key Properties:**
- 72 kinase inhibitors × 442 kinases = ~30,056 measured Kd values
- Complete matrix — virtually all pairs experimentally measured
- Minimum Kd = 10,000 nM (10 uM) → entries at minimum are weak/non-binders
- pKd ≥ 7 (Kd ≤ 100 nM) commonly used as "active" threshold
- At pKd < 5 (Kd > 10 uM): ~27K entries → confirmed non-interacting pairs

**License:** Public / academic use (original paper supplementary)

---

### 1.4 HCDT 2.0 (Seed Set — License Workaround Required)

**Critical License Issue:**
- HCDT 2.0 is **CC BY-NC-ND 4.0** (No Derivatives, Non-Commercial)
- **Cannot redistribute modified versions** of HCDT 2.0 data
- **Cannot use for commercial purposes**

**Workaround Strategy:**
- HCDT 2.0 compiled data from: BindingDB, ChEMBL, GtoPdb, PubChem, TTD
- All underlying sources have permissive licenses (public domain or CC BY-SA)
- **Independently reproduce** HCDT 2.0's methodology:
  1. Query BindingDB for Kd > 100 uM entries
  2. Query ChEMBL for IC50/Ki/Kd > 100 uM entries
  3. Query PubChem for inactive confirmatory results
  4. Apply same >100 uM threshold for inactivity classification
- Factual data (compound X is inactive against target Y) is **not copyrightable**
- Use HCDT 2.0 only as validation reference (compare our independently derived set)

**Expected Volume:** Should approximate HCDT 2.0's 38,653 negative DTIs when applying same criteria

---

### 1.5 InertDB (Universally Inactive Compounds)

**Access:** GitHub repository (https://github.com/ann081993/InertDB)
- 3,205 Curated Inactive Compounds (CIC)
- 64,368 Generated Inactive Compounds (GIC)

**License:** CC BY-NC (Non-Commercial)

**Integration Strategy:**
- **Do not bundle InertDB data directly** (NC license blocks commercial track)
- Provide a download script that fetches from original repository at runtime
- Users opt-in to InertDB data subject to their own license compliance
- Alternatively: independently identify "universally inactive" compounds from PubChem using same criteria (inactive across all tested assays)

---

### 1.6 BindingDB

**Access:** Bulk download (https://www.bindingdb.org/rwd/bind/index.jsp → Download)
- Tab-separated value (TSV) format, updated monthly
- Contains Ki, IC50, Kd, EC50 values with target UniProt IDs

**Extraction Query (Post-Download):**
```python
import pandas as pd

# Load BindingDB TSV
df = pd.read_csv('BindingDB_All.tsv', sep='\t', low_memory=False)

# Filter for high-confidence negatives (Kd/Ki > 10 uM = 10000 nM)
negatives = df[
    ((df['Ki (nM)'] > 10000) | (df['Kd (nM)'] > 10000)) &
    (df['Target Source Organism According to Curator or DataSource'] == 'Homo sapiens')
]
```

**License:** Creative Commons Attribution (free for academic and commercial use)

---

## 2. License Compatibility Analysis

### 2.1 Source License Summary

| Source | License | Commercial OK? | Derivatives OK? | Share-alike? |
|--------|---------|----------------|-----------------|--------------|
| PubChem | Public domain | Yes | Yes | No requirement |
| ChEMBL | CC BY-SA 3.0 | Yes | Yes | **Yes (viral)** |
| BindingDB | CC BY | Yes | Yes | No requirement |
| DAVIS | Academic/public | Yes (implicit) | Yes | No |
| HCDT 2.0 | **CC BY-NC-ND 4.0** | **No** | **No** | N/A |
| InertDB | **CC BY-NC** | **No** | Yes | No |
| GtoPdb | CC BY-SA 4.0 | Yes | Yes | Yes (viral) |
| UniProt | CC BY 4.0 | Yes | Yes | No requirement |

### 2.2 Recommended NegBioDB License: CC BY-SA 4.0

**Rationale:**
- ChEMBL (CC BY-SA 3.0) is the most restrictive permissive source we use
- CC BY-SA 3.0 requires derivatives to be CC BY-SA 3.0 OR a compatible later version
- CC BY-SA 4.0 is explicitly compatible (one-way upgrade allowed: 3.0 → 4.0)
- PubChem (public domain) and BindingDB (CC BY) are compatible with any license
- GtoPdb (CC BY-SA 4.0) is directly compatible

**Dual-Track Implementation:**
```
Open Track (CC BY-SA 4.0):
├── Core dataset (all ChEMBL-derived data included)
├── PubChem-derived data
├── BindingDB-derived data
├── DAVIS matrix data
└── All metadata and annotations

Commercial Track (separate licensing):
├── Value-added analytics and models
├── API premium tier
├── Custom curation services
└── Pre-trained DTI models (trained on CC BY-SA data — licensing TBD)
```

**Critical Note:** Models trained on CC BY-SA data are generally NOT considered derivatives (the model weights are a transformation, not a copy). This is legally debated but the prevailing interpretation in ML community supports this.

### 2.3 HCDT 2.0 and InertDB: Excluded from Core Distribution

- HCDT 2.0 (ND = No Derivatives): Cannot modify or integrate; use only as external reference
- InertDB (NC = Non-Commercial): Cannot include in commercial track
- **Solution:** Independently derive equivalent data from the same underlying sources
- Provide optional integration scripts for users who want to add these under their own license compliance

---

## 3. Cross-Database Deduplication & ID Mapping

### 3.1 The Deduplication Challenge

The same compound-target pair may appear in multiple databases with different identifiers:
- Compound: CID (PubChem) vs. CHEMBL ID vs. ZINC ID vs. BindingDB Monomer ID
- Target: UniProt accession vs. ChEMBL Target ID vs. Gene Symbol vs. NCBI Gene ID
- Same compound may have different SMILES representations (tautomers, salts, stereoisomers)

### 3.2 Compound Standardization Pipeline

```
Input: Raw compound identifiers from any source
                    │
                    ▼
┌──────────────────────────────────────────┐
│ Step 1: Resolve to Canonical Structure    │
│  - UniChem (EBI): Maps between 40+ DBs   │
│    CID → ChEMBL ID → InChIKey            │
│  - PubChem Identifier Exchange Service    │
│  - If SMILES only: parse and standardize  │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│ Step 2: RDKit Standardization             │
│  - Remove salts (SaltRemover)             │
│  - Neutralize charges                     │
│  - Normalize tautomers (TautomerEnumerator)│
│  - Generate canonical SMILES              │
│  - Compute InChIKey (first 14 chars =     │
│    connectivity layer = dedup key)        │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│ Step 3: Deduplication                     │
│  - Group by InChIKey[0:14] (connectivity) │
│  - Merge records, keep all source IDs     │
│  - Flag stereoisomer variants             │
│  - Record provenance for each source      │
└──────────────────────────────────────────┘
```

**RDKit Code Example:**
```python
from rdkit import Chem
from rdkit.Chem import inchi, SaltRemover, MolStandardize

def standardize_compound(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Remove salts
    remover = SaltRemover.SaltRemover()
    mol = remover.StripMol(mol)

    # Normalize
    normalizer = MolStandardize.rdMolStandardize.Normalizer()
    mol = normalizer.normalize(mol)

    # Uncharge
    uncharger = MolStandardize.rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)

    # Generate identifiers
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    inchikey = inchi.MolToInchi(mol)
    inchikey_str = inchi.InchiToInchiKey(inchikey)

    return {
        'canonical_smiles': canonical_smiles,
        'inchikey': inchikey_str,
        'inchikey_connectivity': inchikey_str[:14],  # dedup key
    }
```

### 3.3 Target Standardization Pipeline

```
Input: Target identifiers from any source
                    │
                    ▼
┌──────────────────────────────────────────┐
│ Step 1: Resolve to UniProt Accession      │
│  - ChEMBL target_components table →       │
│    component_sequences.accession          │
│  - PubChem target → UniProt mapping       │
│  - HGNC gene symbol → UniProt (via        │
│    UniProt ID mapping service)            │
│  - NCBI Gene ID → UniProt                 │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│ Step 2: Target Consolidation              │
│  - UniProt accession = canonical ID       │
│  - Map isoforms to canonical entry        │
│  - Handle multi-component targets:        │
│    record each subunit separately +       │
│    complex ID                             │
│  - Record organism (human vs. other)      │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│ Step 3: Target Classification (DTO)       │
│  - Map to Drug Target Ontology family     │
│  - Assign development level (Tclin/Tchem/ │
│    Tbio/Tdark) from IDG/Pharos            │
│  - Record target class (kinase, GPCR,     │
│    ion channel, nuclear receptor, etc.)   │
└──────────────────────────────────────────┘
```

**UniProt Mapping API:**
```
POST https://rest.uniprot.org/idmapping/run
  from: "ChEMBL"
  to: "UniProtKB"
  ids: "CHEMBL1824,CHEMBL2111"
```

### 3.4 Cross-Database Consistency Checking

After deduplication, validate consistency:

| Check | Method | Action on Conflict |
|-------|--------|--------------------|
| Same compound-target, different activity outcome | Compare assay conditions | Record both with assay context; flag for review |
| Same compound, different structures across DBs | Compare InChIKeys | Use RDKit-standardized version; log discrepancy |
| Same target, different organisms | Check UniProt taxonomy | Separate by organism; do not merge cross-species |
| Duplicate assay results (same paper, same assay) | Compare DOI + assay description | Keep highest-quality record; merge metadata |

**Expected Deduplication Rate:** 15-30% overlap between ChEMBL and PubChem for inactive records (based on UniChem cross-reference statistics)

---

## 4. Benchmark Design: Baseline Models & Evaluation Metrics

### 4.1 Baseline Models for NegBioBench

Models should span multiple paradigms to demonstrate benchmark utility:

#### Sequence-Based Models
| Model | Architecture | Input | Year | Key Feature |
|-------|-------------|-------|------|-------------|
| **DeepDTA** | Dual 1D-CNN | Drug SMILES + target sequence | 2018 | Foundational baseline |
| **MolTrans** | Transformer + attention | Drug SMILES + target sequence | 2021 | Interaction attention map |
| **DTI-LM** | Pre-trained protein LM | Morgan FP + ESM embeddings | 2024 | Uses ESM-2 protein language model |

#### Graph-Based Models
| Model | Architecture | Input | Year | Key Feature |
|-------|-------------|-------|------|-------------|
| **GraphDTA** | GCN/GAT/GIN on molecular graph | Molecular graph + target seq | 2020 | Multiple GNN variants |
| **DrugBAN** | Bilinear attention network | Molecular graph + protein graph | 2023 | Pairwise interaction learning |
| **SP-DTI** | Subgraph prediction | Molecular subgraph + binding pocket | 2024 | Substructure-level predictions |

#### Structure-Based Models
| Model | Architecture | Input | Year | Key Feature |
|-------|-------------|-------|------|-------------|
| **HyperAttentionDTI** | Hypergraph attention | 3D structure-aware | 2023 | Multi-scale attention |

#### Evidential / Uncertainty-Aware
| Model | Architecture | Input | Year | Key Feature |
|-------|-------------|-------|------|-------------|
| **EviDTI** | Evidential deep learning | Compound + target features | 2025 | Dirichlet-based uncertainty; explicitly models negative evidence |

#### Traditional ML Baselines
| Model | Features | Year | Purpose |
|-------|----------|------|---------|
| **Random Forest** | Morgan FP + protein descriptors | Classic | Simple baseline |
| **XGBoost** | Morgan FP + protein descriptors | Classic | Gradient boosting baseline |
| **Logistic Regression** | Morgan FP + protein descriptors | Classic | Linear baseline |

### 4.2 Evaluation Metric Suite

**The AUROC Problem:**
- AUROC is insensitive to class imbalance — a model predicting everything as inactive achieves >0.5 AUROC on imbalanced data
- AUROC does not reflect early enrichment (what matters in virtual screening)
- Most existing DTI benchmarks report only AUROC and AUPRC

**Recommended Primary Metrics:**

| Metric | Type | Why | Formula/Implementation |
|--------|------|-----|------------------------|
| **LogAUC[0.001, 0.1]** | Enrichment | Measures early enrichment on log scale; penalizes late recovery | `sklearn + custom integration over [0.001, 0.1] FPR range` |
| **BEDROC (α=20)** | Enrichment | Boltzmann-enhanced discrimination; emphasizes top-ranked compounds | `rdkit.ML.Scoring.CalcBEDROC` |
| **EF@1%** | Enrichment | Enrichment Factor at 1% false positive rate | `(TP@1%FPR / Total Positives) / 0.01` |
| **EF@5%** | Enrichment | Enrichment Factor at 5% false positive rate | `(TP@5%FPR / Total Positives) / 0.05` |
| **AUPRC** | Ranking | Area Under Precision-Recall Curve; sensitive to class imbalance | `sklearn.metrics.average_precision_score` |
| **MCC** | Classification | Matthews Correlation Coefficient; balanced even with extreme imbalance | `sklearn.metrics.matthews_corrcoef` |
| **AUROC** | Ranking | Standard, for comparability with prior work only | `sklearn.metrics.roc_auc_score` |

**Metric Reporting Requirements:**
- Report all 7 metrics for each model × split combination
- Primary ranking metric: **LogAUC[0.001, 0.1]** (most relevant for drug discovery)
- Secondary ranking: **AUPRC** (most informative for imbalanced classification)
- AUROC reported for backward compatibility but NOT used for ranking

### 4.3 Splitting Strategies (Detailed)

| Split | Method | Purpose | Expected Difficulty |
|-------|--------|---------|-------------------|
| **Random** | Stratified random 70/10/20 | Baseline; easiest | Low |
| **Cold Compound** | All compounds in test unseen in train | Generalization to new chemistry | High |
| **Cold Target** | All targets in test unseen in train | Generalization to new biology | Very High |
| **Cold Both** | Both compound AND target unseen | Hardest realistic scenario | Highest |
| **Temporal** | Train < 2020, Val 2020-2022, Test > 2022 | Prospective simulation | High |
| **Scaffold** | Butina clustering on Murcko scaffolds; cluster-based split | Chemical series independence | Medium-High |
| **DDB (Degree-Balanced)** | Degree Distribution Balanced sampling | Addresses node degree bias | Medium |

**Critical Implementation Note:** For Cold Both split, ensure sufficient test set size by selecting compound-target clusters rather than individual pairs.

---

## 5. Validation Experiments

### 5.1 Core Validation Protocol (8 Experiments)

#### Experiment 1: NegBioDB vs. Random Negatives
- **Hypothesis:** Models trained on NegBioDB's experimentally confirmed negatives outperform models trained with random negative sampling
- **Design:** Train DeepDTA, GraphDTA, DrugBAN on (a) NegBioDB negatives, (b) random negatives from same compound-target space
- **Metric:** Compare LogAUC, AUPRC, MCC on held-out NegBioDB test set + external validation set
- **Expected result:** NegBioDB-trained models show higher precision (fewer false positives)

#### Experiment 2: Confidence Tier Discrimination
- **Hypothesis:** Gold-tier negatives provide more reliable training signal than Bronze/Copper
- **Design:** Train models on Gold-only vs. Silver+Bronze vs. All tiers
- **Metric:** Test set performance stratified by confidence tier
- **Expected result:** Gold-trained models show best precision; All-tiers show best recall

#### Experiment 3: Assay Context Matters
- **Hypothesis:** The same compound-target pair shows different activity depending on assay conditions
- **Design:** Identify pairs with conflicting results across assays; analyze condition-dependency
- **Metric:** Frequency and patterns of conditional negatives
- **Expected result:** 5-15% of compound-target pairs show assay-dependent results

#### Experiment 4: Node Degree Bias Quantification
- **Hypothesis:** Existing DTI benchmarks' performance is inflated by node degree correlation
- **Design:** Compare model performance on DDB-split vs. random-split NegBioDB
- **Metric:** Performance gap between split strategies
- **Expected result:** Random split AUROC 10-20% higher than DDB split (demonstrating bias)

#### Experiment 5: Cross-Database Consistency
- **Hypothesis:** Negative results for same compound-target pair are consistent across databases
- **Design:** Compare activity labels for overlapping pairs between PubChem, ChEMBL, BindingDB
- **Metric:** Agreement rate; Cohen's kappa
- **Expected result:** >80% agreement for confirmed dose-response; lower for single-point data

#### Experiment 6: Temporal Generalization
- **Hypothesis:** Models evaluated on temporal splits show realistic prospective performance
- **Design:** Train on pre-2020 data, validate on 2020-2022, test on 2023+
- **Metric:** Performance degradation over time
- **Expected result:** 5-15% performance drop vs. random split (reflecting real-world deployment)

#### Experiment 7: Target Class Coverage Analysis
- **Hypothesis:** NegBioDB covers more diverse target classes than existing benchmarks
- **Design:** Map all targets to DTO families; compare coverage with DAVIS (kinase-only), TDC, etc.
- **Metric:** Target class distribution; Shannon diversity index
- **Expected result:** NegBioDB covers ≥4 major target families vs. 1-2 for existing benchmarks

#### Experiment 8: LIT-PCBA Recapitulation
- **Hypothesis:** NegBioDB subsumes and extends LIT-PCBA's confirmed negatives
- **Design:** Map LIT-PCBA's 15 targets and confirmed inactives to NegBioDB entries
- **Metric:** Coverage rate; additional negatives provided by NegBioDB
- **Expected result:** 90%+ of LIT-PCBA negatives covered; NegBioDB adds 10-50× more targets

### 5.2 Statistical Rigor Requirements

- All experiments: 3 independent runs with different random seeds
- Report mean ± standard deviation for all metrics
- Statistical significance: paired t-test or Wilcoxon signed-rank (p < 0.05)
- Effect sizes (Cohen's d) reported alongside p-values
- Bonferroni correction for multiple comparisons

---

## 6. Compound Quality Filters

### 6.1 PAINS (Pan-Assay Interference Compounds)

```python
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
catalog = FilterCatalog(params)

def is_pains(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return catalog.HasMatch(mol)
```

- PAINS compounds flagged but NOT removed — their inactivity data may be valid
- Flag in metadata: `pains_alert: True/False`

### 6.2 Aggregator Detection
- Flag compounds matching known aggregator scaffolds (Shoichet lab aggregator advisor)
- Critical: Aggregators may show false ACTIVITY (not false inactivity) — their negative results are often MORE reliable

### 6.3 Drug-Likeness Annotations
- Lipinski Rule of 5 compliance
- QED (Quantitative Estimate of Drug-likeness) score
- For information only — do not filter on drug-likeness (non-drug-like compounds provide valuable negative data)

---

## 7. Infrastructure Architecture

### 7.1 MVP Stack (Zero Cost)

```
Data Processing:
├── Python 3.11+
├── RDKit (compound standardization)
├── Pandas / Polars (data wrangling)
├── SQLite (local development DB)
└── Ollama + Gemini API (LLM pipeline)

Storage & Version Control:
├── GitHub (code + small data files)
├── GitHub LFS (datasets < 5GB)
└── SQLite DB file in repo (MVP)

Web Interface (Phase 2):
├── Vercel (free tier hosting)
├── Next.js or Flask
└── Supabase (free tier PostgreSQL)

CI/CD:
├── GitHub Actions (free for public repos)
└── Pre-commit hooks (linting, tests)
```

### 7.2 Data Pipeline Orchestration

```
Recommended: Simple Python scripts with logging (MVP)
├── scripts/01_fetch_pubchem.py
├── scripts/02_fetch_chembl.py
├── scripts/03_fetch_bindingdb.py
├── scripts/04_standardize_compounds.py
├── scripts/05_standardize_targets.py
├── scripts/06_deduplicate.py
├── scripts/07_assign_confidence.py
├── scripts/08_quality_control.py
├── scripts/09_export_benchmark.py
└── Makefile (orchestrate pipeline)

Future: Snakemake or Nextflow (when pipeline complexity grows)
```

---

## Sources

- PubChem PUG REST: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
- ChEMBL API: https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
- UniChem: https://www.ebi.ac.uk/unichem/
- UniProt ID Mapping: https://www.uniprot.org/help/id_mapping
- RDKit: https://www.rdkit.org/docs/
- BEDROC: https://pubs.acs.org/doi/10.1021/ci600426e
- LogAUC: https://pubs.acs.org/doi/10.1021/ci300604z
- MCC for imbalanced data: https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7
- PAINS: https://pubs.acs.org/doi/10.1021/jm901137j
- DDB sampling: https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-025-02231-w
- CC BY-SA compatibility: https://creativecommons.org/share-your-work/licensing-considerations/compatible-licenses
- EviDTI: https://www.nature.com/articles/s41467-025-62235-6
