# 19. Gene Essentiality (GE) Domain — DepMap Design Document

**Date:** 2026-03-22
**Status:** Design Complete → Implementation
**Domain:** Gene Essentiality (GE)
**Primary Source:** DepMap Cancer Dependency Map (Broad Institute)

---

## 1. Motivation

### 1.1 Why Gene Essentiality as NegBioDB's 4th Domain?

CRISPR knockout screens systematically test whether each gene is required for cell viability. In a typical screen of ~18,000 genes, **~85% show no effect on viability** — they are non-essential. This vast space of "gene knockouts that had no effect" is the largest single source of structured biological negative results, yet the community treats it as background noise, focusing only on the ~15% of essential genes.

**Scale comparison:**

| Domain | Negative Results | Entities | Source |
|--------|-----------------|----------|--------|
| DTI | 30.5M | 919K compounds × 3.7K targets | ChEMBL, PubChem, BindingDB, DAVIS |
| CT | 132K | 216K trials × 176K interventions | AACT, CTO |
| PPI | 2.2M | 18.4K proteins | IntAct, HuRI, hu.MAP, STRING |
| **GE** | **~28-30M** | **~18K genes × ~1,865 cell lines** | **DepMap, DEMETER2** |

### 1.2 Structural Parallel to DTI

The DepMap gene×cell_line matrix is structurally identical to the DTI compound×target matrix:
- **DTI:** compound + target → binding? (yes/no)
- **GE:** gene + cell_line → essential? (yes/no)
- Both produce sparse positive matrices embedded in a vast negative space
- Cold splits map directly: cold_gene ↔ cold_compound, cold_cell_line ↔ cold_target

### 1.3 Cross-Domain Hypothesis

Adding GE enables testing the "failure-blindness" hypothesis across 4 domains:
- **DTI (MCC ≤ 0.18):** Failures hidden in file drawers → LLMs are blind
- **CT (MCC ~ 0.5):** ClinicalTrials.gov makes failures public → partial LLM awareness
- **PPI (TBD):** Mixed public/private → intermediate?
- **GE (predicted):** DepMap is fully public, widely cited → LLMs may know essentials but NOT non-essentials

---

## 2. Data Landscape

### 2.1 DepMap CRISPR (Primary Source)

**Project:** Cancer Dependency Map, Broad Institute
**URL:** https://depmap.org/portal/
**Current release:** 25Q3 (quarterly updates)

| File | Content | Format | Size |
|------|---------|--------|------|
| `CRISPRGeneEffect.csv` | Chronos gene effect scores | Wide CSV (ModelID rows × Gene cols) | ~419 MB |
| `CRISPRGeneDependency.csv` | Dependency probability (0-1) | Wide CSV (same structure) | ~412 MB |
| `Model.csv` | Cell line metadata (50+ columns) | CSV | ~2 MB |
| `AchillesCommonEssentialControls.csv` | Genes essential in ≥90% of lines | Single column | <1 MB |
| `AchillesNonessentialControls.csv` | Hart reference nonessential genes | Single column | <1 MB |

**Matrix dimensions:** ~1,865 cell lines × ~18,009 genes = ~33.5M entries
**Score system (Chronos algorithm):**
- 0 = no effect on cell viability
- -1 = median effect of known essential genes (standard threshold)
- < -1 = stronger than typical essential gene
- > 0 = possible growth advantage (artifact in most cases)

**Column format:** `"HUGO_SYMBOL (EntrezID)"` e.g., `"TP53 (7157)"`, `"BRAF (673)"`
**Row identifier:** `ModelID` e.g., `ACH-000001`

**CRISPRGeneEffect vs CRISPRGeneDependency:**
- GeneEffect: Raw Chronos scores (continuous)
- GeneDependency: EM-derived probability that the effect represents true dependency (0-1)
- For binary calls: use GeneDependency ≥ 0.5
- For continuous analysis: use GeneEffect scores

### 2.2 DEMETER2 RNAi (Orthogonal Validation)

**Project:** Combined Achilles + DRIVE + Marcotte RNAi screens
**URL:** Figshare (https://figshare.com/articles/dataset/DEMETER2_data/6025238)
**Scope:** 712 cell lines × ~17K genes

| File | Content | Size |
|------|---------|------|
| `D2_combined_gene_dep_scores.csv` | DEMETER2 dependency scores | ~200 MB |

**Score interpretation:** Same scale as Chronos (0 = no effect, -1 = essential)
**Cell line IDs:** CCLE names (e.g., `184A1_BREAST`), NOT ModelID (ACH-*) — requires mapping
**Value:** RNAi is noisier than CRISPR but orthogonal. CRISPR+RNAi concordance = high-confidence negative.

### 2.3 PRISM Drug Sensitivity (Cross-Domain Bridge)

**Project:** Profiling Relative Inhibition Simultaneously in Mixtures
**URL:** https://depmap.org/repurposing/

| File | Content | Size |
|------|---------|------|
| Primary screen matrix | 4,518 drugs × 578 cell lines (log-fold change) | ~108 MB |
| Secondary dose-response | 1,448 drugs × 499 cell lines (IC50, AUC) | ~252 MB |
| Repurposing Hub sample sheet | broad_id → name, SMILES, InChIKey | ~5 MB |

**Bridge to DTI:** InChIKey in PRISM compounds → ChEMBL compound → DTI negative results.
This creates a DepMap↔DTI cross-domain link: "Compound X has no effect on Cell Line Y (PRISM) AND no binding to Target Z (ChEMBL DTI)."

**Known gap:** Not all broad_ids in PRISM have SMILES in the Repurposing Hub sample sheet. Coverage must be empirically determined.

### 2.4 Omics Data (for ML Features)

| File | Content | Size |
|------|---------|------|
| `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | Gene expression (log2 TPM+1) | ~500 MB |
| `OmicsCNGene.csv` | Gene-level copy number | ~500 MB |
| `OmicsSomaticMutationsMatrixDamaging.csv` | Binary: damaging mutation present | ~200 MB |

These provide pair-specific biological features: a gene's expression level, copy number, and mutation status in a specific cell line.

### 2.5 License Analysis

**DepMap license:** CC BY 4.0
**Terms of Use overlay:** "Data cannot be used to train, develop, or enhance ML/AI models other than for internal research use, or shared for non-profit research purposes."
**NegBioDB use case:** Academic research benchmark, non-commercial, shared for non-profit research purposes. **This is explicitly permitted.**
**Compatibility:** CC BY 4.0 is compatible with NegBioDB's CC BY-SA 4.0.

---

## 3. Schema Design

### 3.1 Entity Tables

**genes:** One row per gene symbol. Includes reference set flags.
- `gene_id` (PK), `entrez_id` (UNIQUE), `gene_symbol`, `ensembl_id`, `description`
- `is_common_essential` (flag), `is_reference_nonessential` (flag)

**cell_lines:** One row per cell line model.
- `cell_line_id` (PK), `model_id` (ACH-*, UNIQUE), `ccle_name`, `stripped_name`
- `lineage`, `primary_disease`, `subtype`, `sex`, `primary_or_metastasis`
- `stripped_name` for DEMETER2 cross-referencing

**ge_screens:** One row per screen configuration.
- `screen_id` (PK), `source_db`, `depmap_release`, `screen_type` (crispr/rnai), `library`, `algorithm`

### 3.2 Fact Table

**ge_negative_results:** One row per non-essential gene-cell_line observation.
- Links: `gene_id`, `cell_line_id`, `screen_id`
- Scores: `gene_effect_score`, `dependency_probability`
- Classification: `confidence_tier`, `evidence_type`, `extraction_method`
- Provenance: `source_db`, `source_record_id`
- Dedup: UNIQUE on `(gene_id, cell_line_id, screen_id, source_db)`

### 3.3 Aggregation Table

**gene_cell_pairs:** One row per unique gene-cell_line pair, aggregated across screens.
- `num_screens`, `num_sources`, `best_confidence`
- Score ranges: `min_gene_effect`, `max_gene_effect`, `mean_gene_effect`
- Network topology: `gene_degree` (# cell lines), `cell_line_degree` (# genes)

### 3.4 PRISM Bridge Tables

**prism_compounds:** Drug compounds with chemical identifiers.
- `broad_id` (UNIQUE), `smiles`, `inchikey`, `chembl_id` (DTI cross-ref)

**prism_sensitivity:** Drug-cell_line sensitivity measurements.
- Links: `compound_id`, `cell_line_id`
- Values: `log_fold_change`, `auc`, `ic50`, `ec50`

---

## 4. Confidence Tiering

### 4.1 Tiering Logic

| Tier | Criteria | Estimated Count |
|------|----------|----------------|
| **Gold** | gene_effect > -0.2 AND dep_prob < 0.3 AND RNAi concordant AND in Hart nonessential set | ~2-3M |
| **Silver** | gene_effect > -0.5 AND dep_prob < 0.5 AND (RNAi concordant OR ≥3 lineage-diverse cell lines) | ~8-10M |
| **Bronze** | gene_effect > -0.8 AND dep_prob < 0.5, single source or single cell line | ~15-18M |

### 4.2 Evidence Types

| Type | Description | Tier Eligibility |
|------|-------------|-----------------|
| `crispr_nonessential` | Chronos score above threshold, single CRISPR screen | All |
| `rnai_nonessential` | DEMETER2 score above threshold, single RNAi screen | All |
| `multi_screen_concordant` | CRISPR + RNAi agree → automatic upgrade | Silver minimum |
| `reference_nonessential` | Gene in Hart/Blomen reference nonessential set | Gold minimum |
| `context_nonessential` | Essential elsewhere but NOT in this lineage | All |

### 4.3 Concordance Tier Upgrade

After loading both CRISPR and RNAi data:
```sql
UPDATE ge_negative_results SET confidence_tier = 'silver'
WHERE confidence_tier = 'bronze'
  AND gene_id IN (
    SELECT DISTINCT r1.gene_id FROM ge_negative_results r1
    JOIN ge_negative_results r2 ON r1.gene_id = r2.gene_id
      AND r1.cell_line_id = r2.cell_line_id
    JOIN ge_screens s1 ON r1.screen_id = s1.screen_id
    JOIN ge_screens s2 ON r2.screen_id = s2.screen_id
    WHERE s1.screen_type = 'crispr' AND s2.screen_type = 'rnai'
  );
```

---

## 5. ML Benchmark Design

### 5.1 GE-M1: Binary Essentiality Classification

**Task:** Predict whether a gene is essential (positive) or non-essential (negative) in a given cell line.

**Positive source:**
- Common essential genes (~2,100): in AchillesCommonEssentialControls AND dep_prob > 0.5
- Selective essential: gene_effect < -1.0 AND dep_prob > 0.5, NOT in common essential set
- Total estimated positives: ~3-5M gene-cell_line pairs

**Negative source:** Curated NegBioDB GE negatives (silver + gold tiers preferred)

**Dataset sizes:**
- Balanced: ~50K positive + 50K negative (1:1)
- Realistic: ~50K positive + 500K negative (1:10)

**Conflict resolution:** If a gene-cell_line pair appears in BOTH positive AND negative sets (e.g., essential in CRISPR but non-essential in RNAi), exclude from both. Same pattern as PPI HuRI conflict removal.

### 5.2 GE-M2: 3-Way Essentiality Classification

**Classes:**
- **Common essential:** Gene in Achilles common essential set AND dep_prob > 0.5
- **Selective essential:** dep_prob > 0.5 AND gene NOT in common essential set
- **Non-essential:** dep_prob < 0.5 AND gene_effect > -0.5

### 5.3 Split Strategies

| Split | Train | Val | Test | Analogy |
|-------|-------|-----|------|---------|
| `random` | 70% | 10% | 20% | Baseline |
| `cold_gene` | Genes A | Genes B | Genes C | cold_compound (DTI) |
| `cold_cell_line` | Cell lines X | Cell lines Y | Cell lines Z | cold_target (DTI) |
| `cold_both` | Neither gene nor cell line in test seen in train | | | Metis partition (PPI) |

**Hypothesis:** cold_cell_line will be hardest because cell line context is critical for selective essentiality prediction.

### 5.4 Control Negatives

- `uniform_random`: Random gene-cell_line pairs from the tested space, not in any essential set
- `degree_matched`: Match gene_degree distribution of NegBioDB negatives

### 5.5 Features (~75 dimensions)

**Gene-level (8 dims):**
- mean, std, min, max gene_effect across all cell lines (4)
- fraction_essential: proportion of cell lines where dep_prob > 0.5 (1)
- is_common_essential flag (1)
- is_reference_nonessential flag (1)
- rnai_concordance: fraction of cell lines where CRISPR+RNAi agree (1)

**Cell line features (~64 dims):**
- Lineage one-hot (~30)
- Primary disease one-hot (~33)
- Mutation burden (1)

**Omics features (pair-specific, 3 dims):**
- Gene expression TPM in this cell line (1)
- Gene copy number in this cell line (1)
- Damaging mutation indicator for this gene in this cell line (1)

### 5.6 Models

- **XGBoost:** Tabular data — expected best performer (tabular domain)
- **MLP:** 2-3 hidden layers on concatenated features

---

## 6. LLM Benchmark Design

### 6.1 GE-L1: 4-Way MCQ Classification (1,200 items)

**Question:** Classify gene essentiality status in a given cell line context.
**Classes:** A=Common essential, B=Selective essential, C=Non-essential, D=Unknown/Untested
**Gold answers:** From dependency scores + reference sets
**Difficulty:** easy (gene function + lineage), medium (gene name + lineage), hard (gene name only)
**Split:** 60 fewshot + 60 val + 180 test per class

### 6.2 GE-L2: Structured Extraction (500 items)

**Task:** Extract gene essentiality fields from constructed evidence descriptions.
**Fields:** gene_name, cell_line_or_tissue, screen_method, essentiality_status, dependency_score
**Source:** Template-based evidence descriptions (not PubMed — DepMap publications don't reference individual pairs)

### 6.3 GE-L3: Reasoning (200 items)

**Task:** Explain why a gene is non-essential in a given cell line context.
**Context:** Gene function, pathways, cell line lineage/disease
**Judge dimensions:** biological_plausibility, pathway_reasoning, context_specificity, mechanistic_depth

### 6.4 GE-L4: Discrimination (500 items)

**Task:** Determine if a gene-cell_line pair has been tested for CRISPR essentiality.
**Temporal contamination design:**
- Compare DepMap 22Q2 (1,086 cell lines) vs 25Q3 (~1,865 cell lines)
- "Tested" = pairs present in 22Q2 (likely in LLM training data)
- "Untested" = pairs ONLY in 25Q3 (new cell lines added after LLM cutoff)
- 250 tested + 250 untested
- Contamination gap = accuracy(old) - accuracy(new)

### 6.5 Models (same as DTI/CT/PPI)

5 models × 4 tasks × 4 configs = 80 LLM runs:
- Llama 3.3 70B (vLLM local)
- Qwen 2.5 72B (vLLM local)
- GPT-4o-mini (OpenAI API)
- Claude Haiku 4.5 (Anthropic API)
- Gemini 2.5 Flash (Google API)

---

## 7. DEMETER2 Cell Line ID Mapping

**Challenge:** DEMETER2 uses CCLE names (e.g., `184A1_BREAST`), DepMap uses ModelID (e.g., `ACH-000001`).

**Mapping strategy:**
1. Load Model.csv → build `{ccle_name: model_id}` dictionary
2. Also build `{stripped_name: model_id}` for fuzzy matching
3. For each DEMETER2 row:
   - Try exact match on `ccle_name`
   - Try exact match on `stripped_name`
   - If no match → log and skip
4. Report mapping statistics (matched/unmatched/total)

**Expected coverage:** ~650-700 of 712 DEMETER2 cell lines should match.

---

## 8. PRISM Cross-Domain Bridge

### 8.1 Bridge Architecture

```
PRISM compound (broad_id)
  ↓ Repurposing Hub
InChIKey / SMILES
  ↓ ChEMBL lookup
chembl_id
  ↓ NegBioDB DTI
DTI negative results for this compound
```

### 8.2 Expected Coverage

- PRISM: ~4,518 compounds with broad_id
- Repurposing Hub: ~3,500-4,000 with SMILES/InChIKey (some missing)
- ChEMBL match: ~30-50% of those with InChIKey
- Final bridge: ~1,000-2,000 compounds with BOTH PRISM AND DTI data

### 8.3 Cross-Domain Analysis

"Compound X has no effect on Cell Line Y (PRISM log_fc ≈ 0) AND no binding to Target Z in Cell Line Y's lineage (ChEMBL pChEMBL < 5)."

This multi-modal negative corroboration is unprecedented in the literature.

---

## 9. Known Challenges

### 9.1 Score Threshold Sensitivity
The -0.5 gene_effect threshold for non-essential is somewhat arbitrary. Sensitivity analysis should test -0.3, -0.5, -0.8 thresholds and report how many negatives and what tier distributions result.

### 9.2 Screen Quality Variation
Not all DepMap screens have equal quality. Some cell lines have higher noise floors. The dependency_probability (from EM algorithm) partially accounts for this, but per-cell-line quality metrics may be needed.

### 9.3 Common Essential Dominance
~2,100 common essential genes appear as essential in ≥90% of cell lines. These dominate the positive class and are "trivially" predictable. The GE-M2 task explicitly separates them from selective essentials.

### 9.4 Non-Essential Space Is Vast
~85% of gene-cell_line pairs are non-essential. Random baseline AUROC will be high simply due to class imbalance. The balanced dataset (1:1) and control negatives (uniform_random, degree_matched) address this.

### 9.5 Temporal Leakage for LLM L4
DepMap data is widely discussed in publications. LLMs likely "know" many essentiality results even from recent releases. The 22Q2 vs 25Q3 comparison tests whether new cell lines (added after training cutoff) are truly unknown to LLMs.

---

## 10. References

- Dempster et al. (2021). Chronos: Inferring tumor clonal composition from high-throughput CRISPR screens. *Genome Biology*. [DOI:10.1186/s13059-021-02540-7](https://doi.org/10.1186/s13059-021-02540-7)
- McFarland et al. (2018). Improved estimation of cancer dependencies from large-scale RNAi screens using model-based normalization and data integration (DEMETER2). *Nature Communications*. [DOI:10.1038/s41467-018-06916-5](https://doi.org/10.1038/s41467-018-06916-5)
- Corsello et al. (2020). Discovering the anticancer potential of non-oncology drugs by systematic viability profiling (PRISM). *Nature Cancer*. [DOI:10.1038/s43018-019-0018-6](https://doi.org/10.1038/s43018-019-0018-6)
- Hart et al. (2015). High-Resolution CRISPR Screens Reveal Fitness Genes and Genotype-Specific Cancer Liabilities. *Cell*. [DOI:10.1016/j.cell.2015.11.015](https://doi.org/10.1016/j.cell.2015.11.015)
- Blomen et al. (2015). Gene essentiality and synthetic lethality in haploid human cells. *Science*. [DOI:10.1126/science.aac7557](https://doi.org/10.1126/science.aac7557)
- Pacini et al. (2021). Integrated cross-study datasets of genetic dependencies in cancer. *Nature Communications*. [DOI:10.1038/s41467-021-21898-7](https://doi.org/10.1038/s41467-021-21898-7)
- Tsherniak et al. (2017). Defining a Cancer Dependency Map. *Cell*. [DOI:10.1016/j.cell.2017.06.010](https://doi.org/10.1016/j.cell.2017.06.010)
- DepMap Portal: https://depmap.org/portal/
- PRISM Portal: https://depmap.org/repurposing/
- Broad Repurposing Hub: https://repo-hub.broadinstitute.org/repurposing
