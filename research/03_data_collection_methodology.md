# Data Collection & Curation Methodology

> Methodologies for collecting, curating, and structuring negative/null results in DTI research (2026-03-02)

---

## 1. Text Mining from Literature

### 1.1 Key Tools and Models

| Tool | Purpose | Performance |
|------|---------|-------------|
| **BioBERT** | Biomedical NER + relation extraction | Pre-trained on 4.5B words (PubMed) + 13.5B words (PMC) |
| **NegBERT** | Negation detection + scope resolution | F1: 95.68 (BioScope Abstracts), 91.24 (BioScope Full Papers) |
| **LLM-IE** (2025) | Biomedical information extraction pipeline | NER >70% strict F1; RE recall 0.978 |
| **BioReader** | Biomedical literature classification | Trainable on positive/negative example corpora |

### 1.2 Negation & Hedge Detection is Critical

Negative DTI results are typically expressed through negation and hedging:
- "compound X **did not inhibit** target Y"
- "**appeared to have no significant effect**"
- "**at the concentrations tested**, no binding was observed"

**BioScope corpus**: Annotated biomedical texts for uncertainty, negation, and their scopes (foundational training resource)

Hedge cues: "probable," "likely," "may," "might," "suggest," "indicate"
- Caution: Non-hedge usage reaches 90% for some cue words → precision critical

### 1.3 Current Gap

- Virtually **no existing research** on mining negative DTI results from PubMed/PMC
- Most text mining focuses on extracting positive interactions
- 85% of researchers do not publish null findings → literature coverage inherently limited

---

## 2. Supplementary Materials Mining

### 2.1 The Hidden Negative Data Problem

- Genomics text mining: only 3-8% of mutations recoverable from prose alone
- **Including supplementary tables → ~50% recovery rate**
- Same pattern likely applies to DTI negative results (inactive compounds in supplementary tables)

### 2.2 Automated Table Extraction

| Tool/Method | Description | Performance |
|------------|-------------|-------------|
| **Table Transformer** | Detects table bounding boxes + cell predictions in PDFs | Grid Table Similarity: 0.12 → 0.90 |
| 7-step methodology | Detection → functional → structural → semantic → pragmatic → selection → extraction | Systematic pipeline |

### 2.3 Challenges
- Heterogeneous file formats (PDF, Excel, Word, CSV)
- Non-standardized table layouts
- Inconsistent column headers for activity/inactivity
- Supplementary files are typically not indexed or searchable

### 2.4 Proposed Pipeline

```
1. Programmatically download supplementary files via PMC/Publisher APIs
2. Apply table detection models to identify activity/screening tables
3. NLP to classify columns as containing activity outcomes
4. Extract "inactive," "no effect," or below-threshold results
5. Link to compound identifiers, targets, and assay conditions
```

---

## 3. High-Throughput Screening (HTS) Data Processing

### 3.1 Distinguishing True Negatives from Artifacts

In HTS, >99% of compounds are inactive. The challenge: true biological inactivity vs. technical artifacts.

**Assay Quality Metrics:**

| Metric | Gold Standard | Meaning |
|--------|-------------|---------|
| **Z-factor (Z')** | > 0.5 = excellent | 0-0.5 = doable; < 0 = screening impossible |
| **SSMD** | Supplements Z-factor limitations | Better control of false-negative/positive rates |
| **B-score** | Row/column positional effect correction | Plate-based normalization |

### 3.2 Statistical Methods for Inactivity Classification

- **Percent inhibition**: < 30% inhibition at screening concentration → inactive
- **Z-score methods**: Compare to negative control distribution (robust z*-score recommended)
- **Bayesian approaches**: Model three hypotheses (no effect / activation / inhibition) with posterior probabilities
- **Frequent hitter detection**: Identify promiscuous compounds active across unrelated assays (likely artifacts)

### 3.3 Dose-Response Confirmation of Inactivity

- All activity < 50% at highest concentration → IC50 = "> highest concentration tested"
- Active/inactive boundary is **inherently arbitrary**: fragments (1 mM) vs. drug-like (10 uM)
- Compounds failing dose-response curve fitting → usually discarded, but this is **valuable negative data**

### 3.4 Metadata to Record for HTS Negatives

- Z' factor (plate/assay quality)
- Inactivity threshold used
- Concentration(s) tested
- "Confirmed inactive" (dose-response) vs. "Screening inactive" (single-point) distinction
- Assay conditions (cell line, buffer, temperature, detection method)

---

## 4. Electronic Lab Notebook (ELN) Data

### 4.1 Current State

- ELN market: $700M (2024) → $1.4B projected (2034)
- **Most negative data is locked inside ELNs**
- Industry trend: ELN-centric → data-centric ecosystem transition (Lab of the Future USA 2025)
- Open Lab Notebooks advocated but adoption remains minimal

### 4.2 Accessible Pharma Industry Initiatives

| Initiative | Feature | Negative Data Access |
|-----------|---------|---------------------|
| **Structural Genomics Consortium (SGC)** | All data/reagents/tools must be public. Failed experiments shared. | **High** |
| **Target 2035** | Modulator for every human protein by 2035. **AIRCHECK** platform: positive + negative DEL screening data released openly | **High** |
| **IMI / Open PHACTS** | Public-private partnership. Integrated open/confidential data | Medium |
| **Open Pharma** | 77% of pharma publications now OA → but positive results primarily | Low |

**Priority Partners: SGC, Target 2035/AIRCHECK**

---

## 5. Crowdsourcing & Community Approaches

### 5.1 DREAM Challenges

- 30+ open science competitions since 2006 (Sage Bionetworks)
- When all teams fail a challenge = **systematic generation of "negative results about methods"**
- **First DREAM Target 2035 Drug Discovery Challenge**: Uses positive + negative screening data for prediction tasks

### 5.2 Registered Reports

- **300+ journals** (including Nature) now accept Registered Reports
- Publication decision made before results are known → dramatic increase in null result publication
- Pre-registered confirmatory studies in pharmacology shown to reduce animal use

### 5.3 Incentive Models

- NINDS 2024 meeting proposals: "null-results summary" programs, "null-data clinics"
- Survey of 11,000+ researchers: 55% unaware of institutional/funder support for null results
- **Awareness gap is as much a problem as the incentive gap**

---

## 6. LLM-Based Approaches

### 6.1 LLMs for Negative Data Labeling

| Approach | Result | Limitation |
|----------|--------|-----------|
| GPT fine-tuning (Ada/Babbage/Curie) | Applied to antimicrobial/hemolytic classification | GPT-3.5 < RNN/SVM (fingerprint-based) |
| Y-Mol (2024) | Knowledge-guided virtual screening | Demonstrates LLM versatility |
| Nature BME 2025 | Collaborative LLM for drug analysis | Early stage |

**Current consensus:** LLMs are promising but traditional ML still outperforms for specific bioactivity classification tasks.

### 6.2 LLMs for Structured Data Extraction

- **LLM-IE** (JAMIA Open 2025): NER + entity attribute + relation extraction pipeline
- Claude 3 Opus, GPT-4, Llama 3-70b: Excellent on synthetic EHR; variable on real data
- Structured output formats consistently improve performance

### 6.3 Quality Control Requirements

1. **Hallucination**: Relationship fabrication → especially dangerous for negative results DB
2. **Distribution bias**: Over-exclusion when relevant/irrelevant ratio is skewed
3. **Reproducibility**: GPT outputs vary between runs
4. **Context window**: Full papers + supplementary may exceed limits
5. **Validation essential**: Human review, cross-referencing, consistency checking

### 6.4 Recommended LLM Pipeline

```
Stage 1: Coarse Filtering
  → LLM identifies papers likely containing negative DTI results

Stage 2: Fine-grained Extraction
  → Extract compound, target, assay conditions, outcome
  → Force structured JSON output

Stage 3: Confidence Scoring
  → LLM self-assessment or ensemble approaches

Stage 4: Human-in-the-Loop
  → Priority validation for novel/high-impact negative results
```

---

## 7. Data Quality & Validation

### 7.1 Negative Result Type Classification

| Type | Meaning | How to Identify |
|------|---------|-----------------|
| **True negative** | Genuinely inactive | Multiple assays/conditions, adequate controls, sufficient concentration range |
| **Experimental failure** | Assay malfunction | Z' < 0, control anomalies |
| **Conditional negative** | Inactive only under tested conditions | pH, temperature, cell line, concentration-dependent |
| **Inconclusive** | Cannot determine status | Insufficient data, ambiguous dose-response, interference |

### 7.2 Statistical Power

- Detecting interactions requires **4× the sample size** needed for main effects
- Negative result for weak interaction needs substantially more replicates
- Multi-center studies show significant result variation even for well-established assays
- G*Power recommended for power calculations

### 7.3 Proposed Confidence Tier System

| Tier | Criteria | Example |
|------|----------|---------|
| **Gold (Tier 1)** | Multiple orthogonal assays, full dose-response, Z' > 0.5, replicates | Multi-center validated inactivity |
| **Silver (Tier 2)** | Single confirmed dose-response, good QC | Confirmatory screen inactive |
| **Bronze (Tier 3)** | Single-point screening inactive, validated assay | HTS primary screen at 10 uM |
| **Copper (Tier 4)** | Literature-reported, NLP/LLM extracted, not yet validated | Text-mined from supplementary |

---

## 8. Ontologies & Standards

### 8.1 BioAssay Ontology (BAO)

- First formal ontology for HTS assays and outcomes
- Covers: assay format, design, technology, target, endpoint
- Bioactivity outcomes: chemical probe, active, **inactive**, inconclusive, unspecified
- 350+ assays curated
- **Essential foundation for standardizing "inactive" definitions in NegBioDB**

### 8.2 Drug Target Ontology (DTO)

- Framework for classifying druggable protein targets
- Four major families: GPCRs, kinases, ion channels, nuclear receptors
- Integrated with IDG (Illuminating the Druggable Genome) portal
- Target development levels: Tclin, Tchem, Tbio, **Tdark** (little-studied → highest value for negative data)

### 8.3 MIABE (Minimum Information About a Bioactive Entity)

- Checklist for reporting bioactive molecules
- Covers: chemical identity, biological assays, ADMET, provenance
- **Needs extension for negative results**: max concentration tested, Z', replicate count, inactivity threshold

### 8.4 FAIR Principles

- **Findable**: Rich machine-readable metadata + persistent identifiers
- **Accessible**: Open or clearly defined access protocols
- **Interoperable**: Standard ontologies (BAO, DTO, ChEBI, UniProt IDs)
- **Reusable**: Clear provenance, licensing, quality annotations

### 8.5 The Critical Standardization Gap

> PubChem does not have standardized rules for applying the "inactive" outcome category
> — each depositor defines their own criteria

**This is the single most important standardization gap NegBioDB needs to address.**

---

## 9. Recommended End-to-End Data Pipeline

```
┌─────────────────────────────────────────────────────┐
│                 DATA SOURCES                         │
├─────────────┬──────────────┬────────────┬───────────┤
│ PubChem     │ ChEMBL       │ Literature │ Community │
│ BioAssay    │ "Not Active" │ Text Mining│ Submission│
│ (confirmed  │ records +    │ + Suppl.   │ Portal    │
│  inactive)  │ pChEMBL < 5  │ Tables     │           │
└──────┬──────┴──────┬───────┴─────┬──────┴─────┬─────┘
       │             │             │            │
       ▼             ▼             ▼            ▼
┌─────────────────────────────────────────────────────┐
│              INGESTION & EXTRACTION                  │
│  - PubChem API (confirmatory > primary)              │
│  - ChEMBL SQL/API (activity_comment + pChEMBL)      │
│  - LLM pipeline (filter → extract → score)           │
│  - Community web forms (controlled vocabularies)     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              QUALITY CONTROL & VALIDATION             │
│  - Assay QC metrics (Z', SSMD)                       │
│  - Compound integrity (PAINS filter, aggregation)    │
│  - Cross-reference check (multi-DB consistency)      │
│  - Statistical power assessment                      │
│  - Confidence tier assignment (Gold/Silver/Bronze)   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              STANDARDIZATION & STORAGE               │
│  - BAO-based assay classification                    │
│  - DTO-based target classification                   │
│  - Standardized identifiers (ChEMBL ID, UniProt,    │
│    InChIKey, SMILES)                                 │
│  - FAIR-compliant metadata                           │
│  - Negative result type annotation                   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              ACCESS & DISTRIBUTION                   │
│  - Web interface (search, browse, download)          │
│  - Python API (pip install negbiodb)                 │
│  - REST API (tiered access)                          │
│  - ML-ready formatted datasets                      │
│  - Standardized benchmark splits                     │
└─────────────────────────────────────────────────────┘
```

---

## Sources

- BioBERT: https://pmc.ncbi.nlm.nih.gov/articles/PMC11020656/
- NegBERT: https://arxiv.org/abs/1911.04211
- BioScope: https://pmc.ncbi.nlm.nih.gov/articles/PMC2586758/
- Table extraction: https://biodatamining.biomedcentral.com/articles/10.1186/s13040-025-00438-9
- LLM-IE: https://academic.oup.com/jamiaopen/article/8/2/ooaf012/8071856
- Z-factor: https://en.wikipedia.org/wiki/Z-factor
- SSMD: https://pmc.ncbi.nlm.nih.gov/articles/PMC7509605/
- SGC: https://www.thesgc.org/about/open-science
- Target 2035 AIRCHECK: https://www.nature.com/articles/s41570-025-00737-z
- DREAM Challenges: https://pmc.ncbi.nlm.nih.gov/articles/PMC7529695/
- BAO: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-12-257
- DTO: https://jbiomedsem.biomedcentral.com/articles/10.1186/s13326-017-0161-x
- FAIR: https://www.nature.com/articles/sdata201618
- InertDB: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-00999-1
- Registered Reports: https://www.nature.com/articles/d41586-018-07118-1
