# DTI Negative Data Landscape

> Survey of existing Drug-Target Interaction negative/inactive data sources (2026-03-02)

---

## Executive Summary

A fundamental paradox defines the DTI negative data landscape: the vast majority of compound-target combinations are biologically inactive, yet negative results are systematically underrepresented in public databases due to publication bias, proprietary data hoarding, and the inherent design of bioactivity databases.

**Key Numbers:**
- Pharma internal: MELLODDY project across 10 pharma companies = **2.6 billion** data points (private)
- PubChem: **295 million** bioactivities, 91.4% inactive (public, quality issues)
- ChEMBL: **21.1 million** activities, only ~133K explicitly labeled "Not Active" (public, positive-biased)

---

## 1. ChEMBL (v35, Dec 2024)

**URL:** https://www.ebi.ac.uk/chembl/

### Scale
| Metric | Value |
|--------|-------|
| Total bioactivity records | ~21.1M |
| Unique compounds | ~2.5M |
| Assays | ~1.74M |
| Targets | ~16,003 |

### How Inactive/Negative Data is Stored
- **pChEMBL values**: Negative log of activity (molar). Commonly used thresholds for "inactive": pChEMBL < 5 (potency > 10 uM); also 5.5 and 6 used
- **activity_comment / TEXT_VALUE**: Qualitative annotations ("Not Active", "Inactive", "Inconclusive"). Standardization to "Not active" underway from ChEMBL 37. ~133,421 narrowly-defined "Not Active" entries; **~763K total** when including all activity_comment variants (see research/08 §1 for verified counts)
- **data_validity_comment**: Data reliability flags (since v15)

### Limitations
- **Strong positive bias**: Curated from published literature, which over-reports active compounds
- **Absence ≠ negative**: Missing compound-target pair = "untested," not "inactive"
- **Threshold dependency**: No universal definition of "inactive"
- **Assay heterogeneity**: Same compound may appear active/inactive depending on assay format

### AI/ML Relevance
Most commonly used DTI training source, but models inherit positive bias → excessive false positives.

---

## 2. PubChem BioAssay

**URL:** https://pubchem.ncbi.nlm.nih.gov/docs/bioassays

### Scale (Sep 2024)
| Metric | Value |
|--------|-------|
| Total substances | 322M |
| Total compounds | 119M |
| Total bioactivities | 295M |
| Biological assays | 1.67M |
| Active fraction | ~2.8% |
| **Inactive fraction** | **~91.4%** |
| Inconclusive/unspecified | ~5.8% |

### Key Feature
**The only major database that explicitly records inactive experimental outcomes.** The NIH Molecular Libraries Program (MLPCN, 2005-2014) screened ~350K compounds against hundreds of targets and deposited all results (active AND inactive). Tox21 continues this practice.

### Quality Issues
- **Extreme class imbalance**: 91% inactive → models achieve high accuracy by predicting everything as inactive
- **Single-concentration limitation**: Primary HTS screens at 10 uM → "inactive" = "no response at this concentration" only
- **Confirmatory vs. primary**: Primary screen has high false-negative/positive rates. Confirmatory dose-response is more reliable but covers far fewer compounds
- **No standardized "inactive" criteria**: Each depositor defines their own thresholds

### AI/ML Relevance
**The single most important public source of genuine negative data** for DTI modeling. However: raw, uncurated, not ML-ready.

---

## 3. BindingDB

**URL:** https://www.bindingdb.org/

### Scale (2024)
| Metric | Value |
|--------|-------|
| Binding measurements | ~3.17M |
| Drug-like molecules | ~1.39M |
| Protein targets | ~11,382 |

### Negative Data Handling
- **No explicit "inactive" flag** (unlike PubChem)
- Quantitative affinity values (Kd, Ki, IC50, EC50) recorded
- Researchers must apply their own thresholds (commonly Kd > 10 uM)
- HCDT 2.0 extracted 38,653 negative DTIs using > 100 uM threshold
- Patent-derived data tends to include more balanced SAR (both active and inactive analogs)

### Limitations
- Positive bias from published literature
- Sparse coverage at high concentrations needed to establish inactivity

---

## 4. ExCAPE-DB (2017)

**URL:** https://zenodo.org/records/2543724

### Scale
| Metric | Value |
|--------|-------|
| Unique compounds | 998,131 |
| SAR data points | 70.85M |
| Targets | 1,667 |

### Key Feature
**Specifically designed to integrate ChEMBL + PubChem inactive data.** Demonstrated that models trained with true inactive compounds outperform those using random negatives. Realistic active/inactive ratios (active < 10% for many targets).

### Current Status
- **No updates since 2017** (built from ChEMBL 22 + PubChem circa 2016-17)
- CC BY-SA 4.0 license
- Excellent design, but 7+ years outdated

---

## 5. Notable Recent Resources (2025)

### InertDB (2025)
- **3,205 curated inactive compounds (CIC)**: From 4.6M+ PubChem records — inactive across ALL tested assays
- **64,368 generated inactive compounds (GIC)**: Deep generative AI model
- 97.2% satisfy Rule of Five
- GitHub: https://github.com/ann081993/InertDB
- **Limitation:** Small scale; only "universally inactive" compounds

### HCDT 2.0 (2025)
- **38,653 experimentally confirmed negative DTIs** (26,989 drugs, 1,575 genes)
- Sources: BindingDB, ChEMBL, GtoPdb, PubChem, TTD
- Definition: binding affinity > 100 uM
- **One of the few curated negative DTI resources**

### LCIdb (2024)
- Large Compound-Interaction Database (for Komet algorithm)
- Balanced negative DTIs via random sampling from unlabeled pairs

---

## 6. Pharmaceutical Company Data Sharing

### MELLODDY (Largest Scale)
- 10 pharma companies (Amgen, AZ, Bayer, BI, GSK, Janssen, Merck KGaA, Novartis, Servier, etc.)
- **2.6B+ activity data points**, 21M+ molecules, 40,000+ assays
- Federated learning (data remains private)
- **Not publicly accessible**

### Other Initiatives
| Initiative | Content | Accessibility |
|-----------|---------|---------------|
| AstraZeneca Open Innovation | Partial preclinical data sharing | Limited |
| Eli Lilly OIDD | 1.8M+ data points (45K+ crowdsourced compounds) | Submitting researchers only |
| GSK/Novartis/St.Jude Malaria | ~20K antimalarial hits (2M+ screened) | Deposited in ChEMBL |
| European Lead Factory | 500K+ compounds, ~270 targets | Private (Honest Data Broker) |

### Reality
- Pharma's internal negative data = **billions** of data points (10-100x all public DBs combined)
- Only 25-33% of large pharma meet data sharing standards (for clinical trials; preclinical sharing is even rarer)

---

## 7. HTS (High-Throughput Screening) Data

### Key Public Sources
- **PubChem BioAssay (MLPCN)**: ~350K compounds × hundreds of targets. Largest public HTS negative source
- **Tox21/ToxCast**: ~10K chemicals × 1,000+ assay endpoints. All results public
- **TDC HTS Task**: Leverages PubChem primary screening data

### HTS Negative Data Characteristics
- Hit rate 0.1-1% → 99-99.9% are "inactive"
- Single-concentration limitation (typically 10 uM)
- PAINS, aggregators, and other artifacts confound results

---

## 8. Systematically Missing Negative Data (Critical Gaps)

| Gap | Description | Priority |
|-----|-------------|----------|
| **Complete interaction matrices** | Outside DAVIS (72×442), nearly no complete matrices exist | Highest |
| **Off-target/selectivity panels** | Pharma routinely runs these but data is almost entirely proprietary | Highest |
| **Concentration-dependent negatives** | Multi-concentration inactivity confirmation is much rarer than single-point | High |
| **Target class coverage** | Public negative data heavily biased toward kinases/GPCRs; emerging targets (PPI, RNA, epigenetic) extremely scarce | High |
| **Allosteric/MOA negatives** | "Binds but doesn't inhibit" vs. "doesn't bind" distinction absent | Medium |
| **Cell-based vs. biochemical discordance** | Biochemically active but cellularly inactive (or vice versa) poorly captured | Medium |
| **Species-specific negatives** | Cross-species activity differences sparse and unsystematic | Medium |
| **Temporal/kinetic negatives** | Residence time, wash-out experiment data essentially absent | Lower |

---

## 9. Key Implications for NegBioDB

### Priority Data Sources

1. **PubChem BioAssay confirmatory data**: Largest experimentally confirmed negative source. Prioritize dose-response inactive over single-point. License: Public domain.
2. **ChEMBL "Not Active" records**: pChEMBL < 5 + activity_comment based. License: CC BY-SA 3.0 (viral — derived works must be CC BY-SA).
3. **BindingDB (Kd/Ki > 10 uM)**: Quantitative binding measurements. License: CC BY.
4. **DAVIS complete matrix**: Benchmark gold standard. License: Public/academic.
5. **HCDT 2.0 curated negatives**: 38,653 high-confidence negative DTIs. **License: CC BY-NC-ND 4.0 — cannot redistribute derivatives. Use as validation reference only; independently extract from underlying sources (BindingDB, ChEMBL, GtoPdb, PubChem, TTD) using same >100 uM threshold.**
6. **InertDB universally inactive compounds**: 3,205 compounds. **License: CC BY-NC — cannot include in commercial track. Provide optional download script only.**
7. **Literature text mining**: Inactive data from supplementary materials. No license restriction on extracted facts.

### Essential Metadata to Record
- Assay format / technology
- Concentration range tested
- Primary vs. confirmatory screen
- Target construct (full-length vs. domain)
- Cell type (if cell-based)
- Z-factor (assay quality)
- Number of replicates

---

## Sources

- ChEMBL: https://www.ebi.ac.uk/chembl/ | NAR 2023: https://academic.oup.com/nar/article/52/D1/D1180/7337608
- PubChem 2025 update: https://pmc.ncbi.nlm.nih.gov/articles/PMC11701573/
- BindingDB 2024: https://academic.oup.com/nar/article/53/D1/D1633/7906836
- ExCAPE-DB: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0203-5
- InertDB: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-00999-1
- HCDT 2.0: https://www.nature.com/articles/s41597-025-04981-2
- MELLODDY: https://pubs.acs.org/doi/10.1021/acs.jcim.3c00799
- DTI Prediction Comprehensive Review 2025: https://www.sciencedirect.com/science/article/pii/S2001037025005719
