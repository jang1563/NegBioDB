# 18. Future Directions: V-JEPA Insights, New Domains, and High-Impact Extensions

**Date:** 2026-03-22
**Status:** Ideation / Deep Research
**Triggered by:** Meta V-JEPA paper analysis + BioNexus Co-Scientist discussion

---

## 1. Motivation: V-JEPA and the "Failure Blindness" of AI

### 1.1 The V-JEPA Discovery

Meta's V-JEPA (Video Joint Embedding Predictive Architecture) family consists of three papers:

| Paper | arXiv | Date | Key Finding |
|-------|-------|------|-------------|
| V-JEPA | [2404.08471](https://arxiv.org/abs/2404.08471) | 2024.02 | Self-supervised video representation via abstract space prediction |
| Intuitive Physics | [2502.11831](https://arxiv.org/abs/2502.11831) | 2025.02 | Object permanence emerges without supervision (IntPhys 98%) |
| V-JEPA 2 | [2506.09985](https://arxiv.org/abs/2506.09985) | 2025.06 | 1B params, zero-shot robotic manipulation |

**Core mechanism:** Mask video regions, predict in **learned abstract representation space** (not pixels). Uses EMA target encoder to prevent collapse.

**Key insight for NegBioDB:** V-JEPA learns "what should happen" and detects violations as prediction errors. This is structurally identical to detecting negative results: learn the expected outcome of experiments, flag deviations (failures) as anomalies.

**Critical comparison — what fails:**
- VideoMAE (pixel-space prediction): near chance on physics violations
- Gemini 1.5 Pro, Qwen2-VL-7B (multimodal LLMs): near chance
- V-JEPA (abstract-space prediction): 98% accuracy

This directly parallels NegBioDB findings:
- LLMs (text surface): DTI L4 MCC ≤ 0.18 (can't distinguish real vs fake negatives)
- MLPFeatures (abstract biological features): PPI cold_both AUROC 0.954
- **Conclusion: Language-based understanding ≠ structural understanding**

### 1.2 Connection to BioNexus/Co-Scientist Discussion

> "성공한 논문만 읽은 AI는 결국 정답지만 본 학생에 가깝고, AI가 전문가 수준으로 가려면 실패와 시행착오가 쌓인 현실의 실험 데이터까지 함께 배워야 한다"

NegBioDB provides the first quantitative evidence for this claim:
- DTI (MCC≤0.18) vs CT (MCC~0.5): failure data accessibility directly correlates with AI discrimination ability
- This is a natural experiment: ClinicalTrials.gov makes failures public, while DTI failures remain in file drawers

### 1.3 Paper Framing Opportunity

**Proposed narrative for NegBioDB paper:**

> Just as V-JEPA learned physical intuitions by predicting abstract representations of video,
> scientific AI needs to learn from the full distribution of experimental outcomes — including
> failures — to develop genuine understanding. NegBioDB demonstrates that LLMs trained
> primarily on published (positive) literature are "failure-blind": they cannot distinguish
> real negative results from fabricated ones (DTI L4 MCC ≤ 0.18). This parallels V-JEPA's
> finding that pixel-space prediction models lack physical intuition while abstract-space
> models develop it spontaneously. The missing ingredient is not more parameters or better
> architectures, but exposure to the full spectrum of experimental outcomes.

---

## 2. JEPA Family in Biomedical Domains (Current Landscape)

### 2.1 Existing JEPA-Based Biomedical Models

| Model | Modality | Reference | Status |
|-------|----------|-----------|--------|
| **EchoJEPA** | Echocardiography video | [arXiv:2602.02603](https://arxiv.org/abs/2602.02603) | 2026.02, Bo Wang Lab |
| **US-JEPA** | Ultrasound images | [arXiv:2602.19322](https://arxiv.org/abs/2602.19322) | 2026.02, SALT objective |
| **CryoLVM** | Cryo-EM density maps | [arXiv:2602.02620](https://arxiv.org/abs/2602.02620) | 2026.02, JEPA+SCUNet |
| **Laya (LeJEPA)** | EEG/electrophysiology | [arXiv:2603.16281](https://arxiv.org/abs/2603.16281) | 2026.03, first EEG JEPA FM |
| **Graph-JEPA** | Graph-level representations | [arXiv:2309.16014](https://arxiv.org/abs/2309.16014) | TMLR accepted |
| **Polymer-JEPA** | Molecular graphs | [arXiv:2506.18194](https://arxiv.org/abs/2506.18194) | 2025.06 |
| **T-JEPA** | Tabular data | ICLR 2025 | EHR applicable |
| **TS-JEPA** | Time series | [arXiv:2509.25449](https://arxiv.org/abs/2509.25449) | 2025 |
| **MTS-JEPA** | Multi-resolution TS anomaly | [arXiv:2602.04643](https://arxiv.org/abs/2602.04643) | 2026.02 |
| **VL-JEPA** | Vision-language | [arXiv:2512.10942](https://arxiv.org/abs/2512.10942) | 2025.12, 50% fewer params |
| **ST-JEMA** | fMRI brain connectivity | [arXiv:2403.06432](https://arxiv.org/abs/2403.06432) | 2024.03, UK Biobank pretrained |
| Multimodal JEPA | CT + EHR (pulmonary nodule) | [arXiv:2509.15470](https://arxiv.org/abs/2509.15470) | 2025.09 |

**Key gap:** No JEPA has been applied to:
- Detecting negative/failed experimental results
- Drug discovery outcome prediction
- Biological "violation of expectation" paradigm

**Critical theoretical support from Laya (LeJEPA):** "Reconstruction-based objectives bias representations toward high-variance artifacts rather than task-relevant neural structure." This directly parallels NegBioDB's finding — LLMs (reconstruction-based) fail at DTI L4 because they learn surface statistics of publications, not the biological semantics of interaction failure.

### 2.2 Related Self-Supervised Biomedical Work

| Paper | Method | Domain | Reference |
|-------|--------|--------|-----------|
| Screener | Self-supervised pathology detection | CT imaging | [arXiv:2502.08321](https://arxiv.org/abs/2502.08321) |
| CEBRA | Contrastive neural latent dynamics | Neuroscience | [Nature 2023, PMID:37138088](https://pubmed.ncbi.nlm.nih.gov/37138088/) |
| Virchow2 | DINOv2 pathology FM | Histopathology | [arXiv:2408.00738](https://arxiv.org/abs/2408.00738) |
| UNI | DINOv2 pathology FM | Histopathology | [Nat Med 2024](https://www.nature.com/articles/s41591-024-02857-3) |
| Cell Painting SSL | DINO/MAE on JUMP | Drug screening | [Sci Reports 2025](https://www.nature.com/articles/s41598-025-88825-4) |
| CellPaint-POSH | SSL + CRISPR screening | Gene function | [Nat Comms 2025](https://www.nature.com/articles/s41467-025-66778-6) |
| **ViTally Consistent** | 1.9B ViT-G/8 MAE on 8B+ microscopy crops | Cell microscopy | [arXiv:2411.02572](https://arxiv.org/abs/2411.02572) |
| CWA-MSN | Masked Siamese Network, 22M params | Cell painting | [arXiv:2509.19896](https://arxiv.org/abs/2509.19896) |
| RigidSSL | SSL from 1.3K MD trajectories | Protein dynamics | [arXiv:2603.02406](https://arxiv.org/abs/2603.02406) |
| SurGBSA | SSL from 1.4M+ MD trajectory reps | Molecular dynamics | [arXiv:2509.03084](https://arxiv.org/abs/2509.03084) |
| EyeWorld | "World model" for eye disease | Ophthalmology | [arXiv:2603.14039](https://arxiv.org/abs/2603.14039) |

### 2.3 Virtual Cell / Biological World Model Concepts

Rapidly emerging field (2025-2026):

| Paper | Key Concept | Reference |
|-------|-------------|-----------|
| Virtual Cells: Frameworks to Applications | Comprehensive review | [arXiv:2509.18220](https://arxiv.org/abs/2509.18220) |
| LLMs Meet Virtual Cell (survey) | LLMs as cell prediction engines | [arXiv:2510.07706](https://arxiv.org/abs/2510.07706) |
| CellForge | Agentic design of virtual cell models | [arXiv:2508.02276](https://arxiv.org/abs/2508.02276) |
| Central Dogma Transformer | Virtual Cell Embedding (DNA→RNA→Protein) | [arXiv:2601.01089](https://arxiv.org/abs/2601.01089) |
| scDFM | Distributional flow matching for perturbation prediction | [arXiv:2602.07103](https://arxiv.org/abs/2602.07103) |
| MicroVerse | Video generation for microscale simulation | [arXiv:2603.00585](https://arxiv.org/abs/2603.00585) |
| **GeneJepa** | JEPA applied to transcriptomics (V-JEPA for gene expression) | 2025 |
| **AlphaCell** | Cell-level world model predicting perturbation outcomes | 2025-2026 |
| **VCWorld** | Virtual Cell World model | 2025 |
| TWIN-GPT | Clinical trial digital twin generator (13 indications) | 2024-2025 |

**Connection to NegBioDB:** Virtual Cells predict "what should happen" when a cell is perturbed. When the prediction differs from reality → the perturbation failed → negative result. This is the V-JEPA paradigm applied to biology.

**Critical gap:** None of these systems systematically model **failure/non-interaction**. TWIN-GPT predicts trial outcomes but not WHY trials fail. NegBioDB's 7-category failure taxonomy provides the structured failure ontology these systems lack.

### 2.4 GeneJepa: Proof-of-Concept for Biological JEPA

GeneJepa directly applies V-JEPA's paradigm to transcriptomics:
- Predicts in **latent space** (not observation space) for gene expression
- Validates the JEPA architecture for biological data
- However, does NOT model negative interactions/non-effects

**The next step:** Extend GeneJepa/Graph-JEPA to interaction prediction, especially negative interactions — this is the "Negative-JEPA" concept (see Section 4).

### 2.5 Geneformer Limitations (Relevant Warning)

[arXiv:2603.02952](https://arxiv.org/abs/2603.02952) analyzed Geneformer V2-316M with sparse autoencoders:
- Only **6.2% of transcription factors** show regulatory-target-specific feature responses
- Attention patterns encode statistical co-expression, NOT causal regulatory logic
- [arXiv:2602.17532](https://arxiv.org/abs/2602.17532): "trivial gene-level baselines outperform both attention and correlation edges"

**Lesson:** Foundation models that appear to "understand" biology may be learning surface statistics, not causal mechanisms — exactly the NegBioDB thesis about publication bias.

---

## 3. New Domain Candidates (Ranked by Composite Score)

### 3.1 Rankings Summary

| Rank | Direction | Score | Data Scale | Novelty | Feasibility | Impact |
|------|-----------|-------|------------|---------|-------------|--------|
| **1** | **CRISPR Screen Negatives (DepMap)** | **9.5** | 36M pairs | High | Very High | Very High |
| **2** | **GWAS Negative Associations** | **9.2** | 100M+ | High | High | Very High |
| **3** | **AI Hypothesis Falsification** | **9.0** | Internal | Extreme | High | Extreme |
| **4** | **Cell Painting Visual Negatives** | **8.5** | 116K compounds, ~70-87K inactive | High | High | Very High |
| 5 | Gene Expression Non-Responses | 7.5 | 30-50K expts | High | Mod-High | High |
| 6 | Reproducibility Crisis DB | 6.5 | 500-1K studies | High | Moderate | Extreme |
| 7 | Metabolomics Negatives | 5.5 | Growing | High | Moderate | Moderate |
| 8 | Multi-Modal Negatives | 5.0 | Assembly | Moderate | Moderate | Moderate |
| 9 | Cryo-EM Failures | 4.5 | Minimal | Very High | Low | Moderate |
| 10 | Antibody/Biologics Failures | 4.0 | Minimal | High | Low | High |
| 11 | Synthetic Biology Failures | 4.0 | Scattered | Very High | Low | Moderate |

**Note:** Cell Painting score revised upward from 6.0 → 8.5 after deep research revealed:
CC0 license (not NC as feared), pre-computed profiles available (~50-100 GB, not 656 TB),
MotiVE fast-start dataset (4.5 GB with ChEMBL cross-refs), and DTI domain cross-referencing via InChIKey.

---

### 3.2 Tier 1: CRISPR Screen Negatives (DepMap) — Score 9.5

**The single richest source of structured negative results in biology.**

#### Data
- **Source:** [DepMap](https://depmap.org/portal/) (Cancer Dependency Map, Broad Institute)
- **Scale:** ~2,000 cancer cell lines × ~18,000 genes = **~36M gene-cell line pairs**
- **Negative results:** ~80-90% are non-essential (gene effect score ≈ 0) = **~29-32M negatives**
- **Format:** Clean matrix CSV, stable gene IDs (HGNC/Entrez)
- **Additional:** PRISM drug sensitivity (~1,500 drugs × 900 cell lines) adds DTI-like dimension
- **License:** CC BY 4.0 (fully compatible)
- **Updates:** Quarterly releases

#### Why This Is Perfect for NegBioDB
1. **Structural parallel to DTI:** Systematic screen → matrix → most entries are "no effect"
2. **Clean data, trivial ETL:** Matrix format, 2-3 days development
3. **Natural confidence tiering:**
   - **Gold:** Gene effect > -0.2, concordant across ≥5 cell lines, validated by RNAi
   - **Silver:** Gene effect > -0.2, 2-4 cell lines
   - **Bronze:** Single cell line or borderline scores
4. **ML benchmark maps directly:**
   - GE-M1: Binary (essential vs non-essential) = DTI M1
   - GE-M2: Conditional essentiality (essential in lineage A but not B) = novel
   - Cold splits: cold_gene, cold_cell_line = cold_drug, cold_target
5. **LLM benchmark maps directly:**
   - GE-L1: "Is gene X essential in cell line Y?" (MCQ)
   - GE-L4: Distinguish essential from non-essential (discrimination)
6. **Contamination testable:** Pre-2022 DepMap releases are in LLM training data

#### Novel Framing
**No one has framed DepMap non-essentials as a "negative results database."** The community uses it to find essential genes — the vast non-essential space (30M pairs) is treated as background noise. NegBioDB would be the first to systematically curate and benchmark "gene knockouts that had no effect."

#### Implementation Sketch
```
New domain: GE (Gene Essentiality)

Schema:
  genes (gene_id, symbol, entrez_id, ensembl_id, description)
  cell_lines (cell_line_id, name, lineage, disease, sex, ...)
  gene_essentiality_results (
    id, gene_id, cell_line_id,
    gene_effect_score,     -- Chronos (0 = no effect, <-1 = essential)
    confidence_tier,       -- gold/silver/bronze
    source_db,             -- 'depmap'
    depmap_release,        -- '25Q3'
    screen_type            -- 'CRISPR' or 'RNAi'
  )

ML benchmark:
  GE-M1: Binary essential vs non-essential
  GE-M2: 3-way (essential/conditional/non-essential)
  Models: XGBoost, GNN on gene interaction networks, MLP
  Splits: random, cold_gene, cold_cell_line, temporal

LLM benchmark:
  GE-L1: MCQ (4-way)
  GE-L2: Extract essentiality info from abstracts
  GE-L3: Reasoning about why gene is (non-)essential
  GE-L4: Discrimination (essential vs non-essential pair)

Estimated: ~30M negative results, ~18K genes, ~2K cell lines
Development: 5-7 days ETL + schema + tests + benchmark design
```

---

### 3.3 Tier 1: GWAS Negative Associations — Score 9.2

**The largest untapped source of structured negative results in genomics.**

#### Data
- **Source:** [GWAS Catalog Summary Statistics](https://www.ebi.ac.uk/gwas/) (EBI/NHGRI)
- **Scale:** ~280K series deposited, each testing ~10M variants. >99.99% are non-significant.
- **Estimated negatives:** **100M+ variant-phenotype pairs** tested and found null
- **Additional sources:** UK Biobank (4,000 phenotypes), FinnGen (2,800 endpoints), Pan-UKB (7,000 phenotypes × 6 ancestries)
- **Format:** Harmonized TSV from EBI pipeline
- **License:** Public domain (most GWAS summary stats are openly deposited)

#### Why This Is Novel
- GWAS Catalog curates only significant hits (p < 5e-8). **Nobody catalogs the non-significant ones.**
- A variant tested in a well-powered study (N>100K) with p > 0.05 is a **reliable negative**
- Statistical power allows principled tiering: well-powered null ≠ underpowered null

#### Confidence Tiering
- **Gold:** N > 100K, p > 0.05, well-imputed variant (info > 0.9)
- **Silver:** N = 10K-100K, p > 0.05
- **Bronze:** N < 10K or imputed with low quality

#### Impact
- "The Catalog of Non-Associations: What Genes Are NOT Associated With What Diseases"
- Directly useful for PRS calibration — PRS models trained with proper negatives may be better calibrated
- Nature Genetics tier

#### Connection to NegBioDB
- Same ETL pattern as PubChem (streaming large TSV files)
- Confidence tiering via statistical power (unique advantage)
- LLM benchmark: can LLMs predict which variant-disease pairs will be non-significant?

---

### 3.4 Tier 1: AI for Scientific Hypothesis Falsification — Score 9.0

**The most novel conceptual direction. Uses existing NegBioDB data.**

#### Concept
Train AI systems specifically to predict **which experiments will produce null results** — before they are run.

This is NOT the same as existing work:
- Drug attrition models predict failure from Phase I data (positive examples of failure)
- Virtual screening predicts inactivity (biased negatives)
- NegBioDB is the FIRST dataset large enough to train a genuine **"failure predictor"** with experimentally confirmed negatives

#### Research Design

```
Phase 1: Within-domain failure prediction
  - DTI: Train on pre-2020 inactive pairs, predict post-2020 outcomes
  - CT: Train on pre-2018 trial failures, predict 2020+ failures
  - PPI: Train on pre-2015 non-interactions, predict post-2020 outcomes
  - GE (if added): Train on early DepMap releases, predict latest release

Phase 2: Cross-domain transfer
  - Does a DTI failure model predict CT failures?
  - Shared features: "experimental testability" across domains
  - Drug → intervention, target → condition/protein mapping

Phase 3: Human expert comparison
  - Present experts with candidate experiments
  - Compare human vs AI predictions of null results
  - Calibration: are experts or AI better calibrated for failure?

Phase 4: Prospective validation
  - Predict outcomes of ongoing ClinicalTrials.gov trials
  - Wait for results → evaluate predictions
  - First prospective negative result prediction benchmark
```

#### Impact
- Science/Nature cover potential: **"AI Can Now Predict Which Experiments Will Fail"**
- Practical: save billions by predicting failures earlier
- Philosophical: if failures are predictable, publication bias is even more damaging
- Becomes more powerful with each domain added to NegBioDB

---

### 3.5 Tier 2: Gene Expression Non-Responses — Score 7.5

#### Data
- **Source:** [recount3](https://rna.recount.bio/) — 750K+ uniformly processed human RNA-seq samples
- **Concept:** Differential expression analysis on all treatment-vs-control experiments. Experiments with 0 DEGs at FDR < 0.05 = **"treatment had no transcriptional effect"** = negative result
- **Estimated scale:** If 10-20% show no DEGs → ~30-50K negative experiments

#### Why Valuable
- Cross-domain synergy: CT drug failure + no transcriptional effect = **mechanistic corroboration**
- "Most treatments have no effect on gene expression. Here is the first catalog."
- LLM benchmark: can LLMs predict which treatments will change expression?

#### Challenges
- GEO metadata is messy (defining treatment vs control requires parsing)
- Processing pipeline needed (DESeq2/edgeR on recount3 matrices)
- Development: 2-3 weeks

---

### 3.6 Cell Painting Visual Negatives — Score 8.5 (REVISED UP from 6.0)

**Score revised after deep research uncovered CC0 license and pre-computed profiles.**

#### Data Sources (JUMP Only — RxRx Excluded Due to NC License)

| Dataset | Compounds | Size | Profiles? | License |
|---------|-----------|------|-----------|---------|
| **cpg0016-jump** (main) | **115,796** (all with InChIKey + SMILES) | 358 TB raw, **~50-100 GB profiles** | Yes (1,700 CellProfiler features/well) | **CC0 1.0** |
| cpg0034 MotiVE | 3,600 + 11K genes | **4.5 GB total** | Yes + ChEMBL graph | **CC0 1.0** |
| cpg0012-wawer (CDRP) | 30,000 | 10.7 TB | Yes | CC0 1.0 |
| cpg0004-lincs | 1,571 (6 doses) | 65.7 TB | Yes | CC0 1.0 |
| cpg0036-EU-OS | 2,464 | 3.5 TB | Yes | CC0 1.0 |

**RxRx1/2 (CC BY-NC-SA) and RxRx3 (Non-Commercial EULA) are excluded** — license incompatible.

#### Key Numbers
- **115,796 compounds** with InChIKey + SMILES in JUMP metadata
- **60-75% are morphologically inactive** (DMSO-like) → **~70,000-87,000 visual negative results**
- Cross-reference with ChEMBL via InChIKey: estimated 30-50% overlap (35-58K compounds)
- **~20,000-40,000 multi-modal negatives** (inactive in BOTH Cell Painting AND ChEMBL)
- Cell line: U2OS (osteosarcoma), Cell Painting v3 (6 fluorescent channels)
- Consortium: 12 pharma companies + Broad Institute

#### Novel Framing
**Nobody has explicitly framed Cell Painting inactives as a "negative result database."**
The community treats the 60-75% inactive rate as a limitation, not as valuable data.
- Pahl et al. (2023) studied "dark chemical matter" but searched FOR activity in inactives — opposite direction
- MotiVE (NeurIPS 2024) built a DTI graph from Cell Painting but focused on active interactions
- Becker et al. (Nat Comms 2023, cited 86×) evaluated activity prediction, not inactivity cataloging

#### Confidence Tiering
- **Gold:** DMSO-like in Cell Painting + ChEMBL pChEMBL < 5 (multi-modal negative)
- **Silver:** DMSO-like in Cell Painting only (percent replicating < threshold)
- **Bronze:** Single replicate or borderline morphological score

#### Feasibility: HIGH (revised)
Three implementation options:

**Option A: MotiVE Fast Start (recommended first step)**
- 4.5 GB download, ChEMBL cross-refs already done for 3,600 compounds
- `aws s3 sync --no-sign-request s3://cellpainting-gallery/cpg0034-arevalo-su-motive/broad/workspace/publication_data/2024_MOTIVE .`
- Immediate proof of concept, 1-2 days

**Option B: Full JUMP Profiles**
- Download pre-processed profiles from Morphmap pipeline (~50-100 GB)
- `s3://cellpainting-gallery/cpg0016-jump-assembled/source_all/workspace/profiles/jump-profiling-recipe_2024_a917fa7/`
- Processing: well position correction, normalization, sphering, Harmony batch correction already applied
- Standard laptop with 32 GB RAM (no GPU needed)
- 1-2 weeks development

**Option C: Raw Images (NOT recommended)**
- 358 TB — requires cloud computing
- Only needed for V-JEPA/deep learning experiments

#### Cross-Domain Bridge to DTI
InChIKey matching enables direct connection to NegBioDB DTI:
1. JUMP compound InChIKey → ChEMBL compound → DTI negative results
2. "Compound X showed no morphological effect (Cell Painting) AND no binding (ChEMBL DTI)"
3. This multi-modal corroboration is unprecedented

#### Key References
- Chandrasekaran et al. (2024). JUMP Cell Painting dataset. [Nat Methods](https://doi.org/10.1038/s41592-024-02241-6)
- Weisbart et al. (2024). Cell Painting Gallery. [Nat Methods](https://doi.org/10.1038/s41592-024-02399-z)
- Becker et al. (2023). Predicting compound activity. [Nat Comms](https://doi.org/10.1038/s41467-023-37570-1) (cited 86×)
- Fredin Haslum et al. (2024). Cell Painting bioactivity. [Nat Comms](https://doi.org/10.1038/s41467-024-47171-1) (cited 32×)
- Arevalo & Su et al. (2024). MotiVE. NeurIPS 2024 Datasets Track
- Pahl et al. (2023). Dark Chemical Matter. bioRxiv

---

## 3.7 Biological Modalities with Video-Like Temporal Structure

Seven modalities are natural V-JEPA targets:

| Modality | Temporal Structure | Existing Data | NegBioDB Relevance |
|----------|-------------------|---------------|-------------------|
| **Cell microscopy time-lapse** | Hours-long drug response video | JUMP-CP, OrganoID (1K organoids) | Drug has no effect = frames don't change |
| **MD trajectories** | Protein conformational dynamics | RigidSSL (1.3K), SurGBSA (1.4M+) | Non-binding = no conformational change |
| **Calcium imaging** | Neural response sequences | CEBRA datasets | No pharmacological response |
| **Organoid growth** | Developmental time-series | Retinal organoid (1K, PMID:41592127) | Failed differentiation |
| **Cryo-EM movie frames** | Dose-fractionated stacks | EMDB | CryoLVM already uses JEPA |
| **EEG/electrophysiology** | Neural dynamics | Laya datasets, UK Biobank | Drug non-response |
| **Flow cytometry TS** | High-dim cell profiles over time | Unexplored | **Open opportunity** |

**Inverted anomaly detection insight:** In standard anomaly detection, anomalies are rare deviations. In drug screening, **negative results (no effect) are the norm** (~99% of compounds are inactive). V-JEPA trained on time-lapse microscopy would learn "normal cell behavior" → positive drug effects become the anomalies, while negative results are the expected inliers. This inverts the traditional framing but is methodologically sound.

---

## 4. Biological World Model: A Unifying Vision

### 4.1 The Grand Idea

V-JEPA learned physical laws by predicting video in abstract space. Can we build a **Biological JEPA** that learns biological laws by predicting experimental outcomes in abstract space?

```
Physical World:
  Input: Video frames → V-JEPA → Abstract representations
  Task: Predict masked frames → learns object permanence, continuity
  Anomaly: Prediction error on impossible physics = "violation detected"

Biological World:
  Input: Experimental conditions → Bio-JEPA → Abstract representations
  Task: Predict masked outcomes → learns what experiments should produce
  Anomaly: Prediction error on unexpected result = "negative result detected"
```

### 4.2 Architecture Concept: Negative-JEPA

```
Context Encoder (ViT/GNN):
  Input: Drug structure + Target sequence + Assay conditions
  Output: Abstract representation of experimental context

Target Encoder (EMA):
  Input: Experimental outcome (activity values, clinical endpoints, interaction data)
  Output: Abstract representation of expected outcome

Predictor:
  Input: Context representation
  Output: Predicted outcome representation

Training: L1 loss between predicted and target representations
Inference: High prediction error → unexpected outcome → potential negative result

Key innovation vs. standard prediction models:
  - Predicts in ABSTRACT space, not in raw measurement space
  - This forces the model to learn generalizable biological principles
  - Surface-level features (assay noise, measurement variance) are ignored
  - Only semantically meaningful biological structure is captured
```

### 4.3 Graph-JEPA as Foundation

[Graph-JEPA (arXiv:2309.16014)](https://arxiv.org/abs/2309.16014) already provides the machinery:
- Masks subgraphs and predicts latent representations of masked regions
- Uses hyperbolic coordinates for hierarchical structure
- Accepted at TMLR

[Polymer-JEPA (arXiv:2506.18194)](https://arxiv.org/abs/2506.18194) applies this to molecular graphs:
- Self-supervised pretraining on polymer graphs
- Enhances downstream prediction when labeled data is scarce

**Proposed extension:** Apply Graph-JEPA to biological interaction networks:
1. PPI network: mask protein-protein edges, predict if interaction exists
2. DTI network: mask drug-target edges, predict binding
3. Training on positive edges → high prediction error on true negatives

This would be a **fundamentally different approach** to negative result detection:
instead of classifying results as positive/negative, learn what "should" happen and detect deviations.

### 4.4 Novelty Assessment: "Negative-JEPA" Is an Open Gap

The following chain validates the approach but reveals **no one has done the final step:**

1. **Graph-JEPA** (Skenderi et al., TMLR) → JEPA works on graphs
2. **Polymer-JEPA** (Piccoli et al., 2025) → JEPA works on molecular graphs
3. **GeneJepa** (2025) → JEPA works on biological data (transcriptomics)
4. **Sousa et al.** (KR 2023) → Negative statements improve KG embeddings for PPI
5. **NegBioDB** → Largest curated negative evidence resource (33M results)
6. **??? → No existing JEPA predicts non-interactions**

**The combination — JEPA for predicting non-interaction in abstract biological representation space, trained on curated negative evidence — is novel and compelling.**

### 4.5 Connection to Existing Work

| Existing Approach | What It Does | Limitation |
|-------------------|-------------|------------|
| Supervised ML (NegBioDB current) | Classify pos/neg from features | Needs labeled negatives |
| LLM benchmark (NegBioDB current) | Test if LLMs know negatives | LLMs are "failure-blind" |
| Virtual Cells (scDFM, CellForge) | Predict perturbation outcomes | Don't explicitly model failure |
| GeneJepa / Cell-JEPA | JEPA on transcriptomics | Don't model negative interactions |
| TWIN-GPT | Clinical trial digital twin | Predicts outcome, not failure reason |
| **Negative-JEPA (proposed)** | **JEPA trained on curated negatives** | **Detects failures as anomalies** |

### 4.6 KG Negative Statements as Prior Art

[arXiv:2307.11719](https://arxiv.org/abs/2307.11719): Benchmark datasets for biomedical KGs with **negative statements**
- Showed that incorporating negative statements improves KG embedding performance
- Created PPI/gene-disease datasets with explicit negatives
- NegBioDB provides orders of magnitude more negative statements than these benchmarks

---

## 5. Recommended Strategy

### Phase A: Current Paper (Immediate)
1. Add V-JEPA framing to NegBioDB paper introduction/discussion
2. Use DTI L4 vs CT L4 contrast as "natural experiment" for failure data accessibility
3. Reference the "failure-blind LLM" finding as parallel to VideoMAE's lack of physics intuition

### Phase B: 4th Domain for Paper (1-2 weeks)
**Add DepMap CRISPR Negatives** as 4th domain:
- 5-7 days development
- Adds ~30M negatives (comparable to DTI)
- 4-domain cross-comparison is significantly more impressive than 3
- Immediate ML/LLM benchmark reuse

### Phase C: Next Paper — GWAS Negatives (4-6 weeks)
- "The Catalog of Non-Associations" as standalone Nature Genetics submission
- 100M+ variant-phenotype negative associations
- Connects to PRS and genomic medicine
- Power-based confidence tiering is methodologically novel

### Phase D: High-Impact Follow-Up — Hypothesis Falsification
- Uses all NegBioDB domains as training data
- "Can AI Predict Which Experiments Will Fail?"
- Cross-domain transfer learning
- Prospective validation on ongoing trials
- Science/Nature tier

### Phase E: Long-Term Vision — Bio-JEPA
- Apply JEPA architecture to biological interaction networks
- Train on NegBioDB positive + negative data
- Detect failures as prediction errors in abstract space
- Novel architecture contribution (NeurIPS/ICML)

---

## 6. Key References

### V-JEPA Family
- V-JEPA: [arXiv:2404.08471](https://arxiv.org/abs/2404.08471)
- Intuitive Physics: [arXiv:2502.11831](https://arxiv.org/abs/2502.11831)
- V-JEPA 2: [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)

### JEPA in Biomedical
- EchoJEPA: [arXiv:2602.02603](https://arxiv.org/abs/2602.02603)
- US-JEPA: [arXiv:2602.19322](https://arxiv.org/abs/2602.19322)
- Graph-JEPA: [arXiv:2309.16014](https://arxiv.org/abs/2309.16014)
- Polymer-JEPA: [arXiv:2506.18194](https://arxiv.org/abs/2506.18194)
- T-JEPA: ICLR 2025
- TS-JEPA: [arXiv:2509.25449](https://arxiv.org/abs/2509.25449)
- MTS-JEPA: [arXiv:2602.04643](https://arxiv.org/abs/2602.04643)
- VL-JEPA: [arXiv:2512.10942](https://arxiv.org/abs/2512.10942)

### Virtual Cell / Biological World Models
- Virtual Cells Review: [arXiv:2509.18220](https://arxiv.org/abs/2509.18220)
- LLMs Meet Virtual Cell: [arXiv:2510.07706](https://arxiv.org/abs/2510.07706)
- CellForge: [arXiv:2508.02276](https://arxiv.org/abs/2508.02276)
- Central Dogma Transformer: [arXiv:2601.01089](https://arxiv.org/abs/2601.01089)
- scDFM: [arXiv:2602.07103](https://arxiv.org/abs/2602.07103)
- MicroVerse: [arXiv:2603.00585](https://arxiv.org/abs/2603.00585)
- AI Scientist: [arXiv:2408.06292](https://arxiv.org/abs/2408.06292)

### Cell Painting / Image-Based Screening
- Cell Painting Bioactivity: [Nat Comms 2024](https://www.nature.com/articles/s41467-024-47171-1)
- SSL on JUMP: [Sci Reports 2025](https://www.nature.com/articles/s41598-025-88825-4)
- CellPaint-POSH (Insitro): [Nat Comms 2025](https://www.nature.com/articles/s41467-025-66778-6)
- SemiSupCon: [JCIM 2024](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00835)

### Pathology FMs
- Virchow2 (Meta): [arXiv:2408.00738](https://arxiv.org/abs/2408.00738)
- UNI (Harvard): [Nat Med 2024](https://www.nature.com/articles/s41591-024-02857-3)
- Phikon-v2 (Owkin): [arXiv:2409.09173](https://arxiv.org/html/2409.09173v1)
- Benchmark: [Nat Comms 2025](https://www.nature.com/articles/s41467-025-58796-1)

### Self-Supervised Anomaly Detection
- Screener (pathology in CT): [arXiv:2502.08321](https://arxiv.org/abs/2502.08321)
- CEBRA (neural dynamics): [Nature 2023](https://pubmed.ncbi.nlm.nih.gov/37138088/)
- V-JEPA ultrasound: [arXiv:2507.18424](https://arxiv.org/abs/2507.18424)

### KG Negative Statements
- Benchmark datasets: [arXiv:2307.11719](https://arxiv.org/abs/2307.11719)
- TrueWalks: [arXiv:2308.03447](https://arxiv.org/abs/2308.03447)

### Geneformer Analysis
- Sparse AE analysis: [arXiv:2603.02952](https://arxiv.org/abs/2603.02952)
- Attention pattern study: [arXiv:2602.17532](https://arxiv.org/abs/2602.17532)

### Other
- MDGEN (MD trajectories): [arXiv:2409.17808](https://arxiv.org/abs/2409.17808) (NeurIPS 2024)
- Protein Dynamics DL: [PNAS 2025](https://www.pnas.org/doi/10.1073/pnas.2502444122)
- DepMap: [depmap.org](https://depmap.org/portal/)
- GWAS Catalog: [ebi.ac.uk/gwas](https://www.ebi.ac.uk/gwas/)
- JUMP Cell Painting: [jump-cellpainting.broadinstitute.org](https://jump-cellpainting.broadinstitute.org/)
- RxRx Datasets: [github.com/recursionpharma/rxrx-datasets](https://github.com/recursionpharma/rxrx-datasets)

---

## Appendix A: Data Source Comparison for New Domains

| Domain | Source | Records | Neg% | Format | License | ETL Effort |
|--------|--------|---------|------|--------|---------|------------|
| GE (DepMap) | CRISPRGeneEffect.csv | ~36M | ~85% | Matrix CSV | CC BY 4.0 | 2-3 days |
| GWAS | EBI Summary Stats | ~100M+ | >99.99% | Harmonized TSV | Public domain | 2-3 weeks |
| GEX | recount3 | ~30-50K expts | 10-20% | Count matrices | Public domain | 2-3 weeks |
| Cell Painting | JUMP cpg0016 profiles | ~116K cpds | ~60-75% | 1,700-feat profiles | CC0 1.0 | 1-2 weeks |

## Appendix B: License Compatibility

| Data Source | License | Compatible with CC BY-SA 4.0? |
|-------------|---------|------------------------------|
| DepMap | CC BY 4.0 | Yes |
| GWAS Catalog | Public domain | Yes |
| JUMP Cell Painting | CC0 1.0 | Yes (public domain) |
| RxRx1/RxRx2 | CC BY-NC-SA 4.0 | **No** (NC clause) |
| RxRx3 | Recursion Non-Commercial | **No** |
| recount3 | Public domain | Yes |
| UK Biobank | Restricted access | Requires application |
