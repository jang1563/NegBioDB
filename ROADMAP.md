# NegBioDB — Execution Roadmap

> Last updated: 2026-03-02 (v4 — dual ML+LLM benchmark design added)

---

## Critical Findings (Added March 2026)

1. **HCDT 2.0 License: CC BY-NC-ND 4.0** — Cannot redistribute derivatives. Must independently recreate from underlying sources (BindingDB, ChEMBL, GtoPdb, PubChem, TTD) using >100 uM threshold. Factual data is not copyrightable.
2. **InertDB License: CC BY-NC** — Cannot include in commercial track. Provide optional download script only.
3. **NeurIPS 2026 D&B deadline: ~May 15, 2026** (~11 weeks from project start). Requires: downloadable data, Croissant metadata, code available, Datasheet for Datasets.
4. **LIT-PCBA compromised** (2025 audit found data leakage) — Creates urgency for NegBioDB as replacement gold-standard.
5. **Recommended NegBioDB License: CC BY-SA 4.0** — Compatible with ChEMBL (CC BY-SA 3.0) via one-way upgrade.
6. **No direct competitor exists** as of March 2026.
7. **No LLM benchmark tests negative DTI tasks** — ChemBench, Mol-Instructions, MedQA, SciBench all lack negative result evaluation. NegBioBench LLM track is first-of-kind.
8. **LLM evaluation also free** — Gemini Flash free tier as LLM-as-Judge + ollama local models as baselines. Flagship models (GPT-4, Claude) added post-stabilization only.

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
- NeurIPS D&B: No publication fee (if accepted)

---

## Accelerated Phase 1: NeurIPS 2026 Sprint (Weeks 0-11)

**Deadline: ~May 15, 2026 (paper) | ~May 1, 2026 (abstract)**

### Week 1-2: Schema + Data Pipeline Setup

- [ ] Finalize database schema (SQLite for MVP)
- [ ] Implement compound standardization pipeline (RDKit: salt removal, normalization, InChIKey)
- [ ] Implement target standardization pipeline (UniProt accession as canonical ID)
- [ ] Set up cross-DB deduplication (InChIKey[0:14] connectivity layer)

### Week 2-4: Data Extraction

**Data Sources (License-Safe Only):**

| Source | Target Volume | Method | License |
|--------|--------------|--------|---------|
| PubChem BioAssay (confirmatory inactive) | ~50K+ | PUG REST API, filter `activity_outcome=inactive` | Public domain |
| ChEMBL "Not Active" + pChEMBL < 5 | ~133K | SQL query on downloaded dump | CC BY-SA 3.0 |
| BindingDB (Kd/Ki > 10 uM) | ~30K+ | Bulk TSV download + filter | CC BY |
| DAVIS complete matrix (pKd < 5) | ~27K | TDC Python download | Public/academic |
| Independent HCDT-style extraction | ~38K | Query underlying sources at >100 uM threshold | Derived from public domain + CC BY-SA |

**NOT bundled (license issues):**
- HCDT 2.0 (CC BY-NC-ND) — Use as validation reference only
- InertDB (CC BY-NC) — Optional download script for users

- [ ] Build PubChem BioAssay extraction script (confirmatory > primary > counter-screen)
- [ ] Build ChEMBL extraction SQL query (activity_comment + pChEMBL threshold)
- [ ] Build BindingDB extraction script (filter Kd/Ki > threshold)
- [ ] Integrate DAVIS matrix from TDC
- [ ] Independently extract HCDT-equivalent negatives from underlying sources
- [ ] Run compound/target standardization on all extracted data
- [ ] Run cross-DB deduplication
- [ ] Assign confidence tiers (Gold/Silver/Bronze/Copper)

### Week 4-6: Benchmark Construction (ML + LLM)

**ML Track:**
- [ ] Implement 7 splitting strategies (Random, Cold-Compound, Cold-Target, Cold-Both, Temporal, Scaffold, DDB)
- [ ] Implement ML evaluation metrics: LogAUC[0.001,0.1], BEDROC, EF@1%, EF@5%, AUPRC, MCC, AUROC

**LLM Track:**
- [ ] Design prompt templates for L1, L2, L4 (priority tasks)
- [ ] Construct L1 dataset: 2,000 MCQ from NegBioDB entries (semi-automated)
- [ ] Construct L2 dataset: annotate 200 PubMed abstracts with gold-standard JSON
- [ ] Construct L4 dataset: 500 tested/untested compound-target pairs
- [ ] Implement automated evaluation scripts (L1: accuracy/F1, L2: schema/entity F1, L4: classification F1)
- [ ] Set up LLM-as-Judge pipeline (Gemini 2.5 Flash free tier)
- [ ] (If time) Construct L3 dataset: 200 reasoning examples with expert rubrics

**Shared:**
- [ ] Generate Croissant machine-readable metadata (NeurIPS mandatory)
- [ ] Write Datasheet for Datasets (Gebru et al. template)

### Week 5-8: Baseline Experiments (ML + LLM)

**ML Baselines (minimum for submission):**

| Model | Type | Priority |
|-------|------|----------|
| DeepDTA | Sequence CNN | Must have |
| GraphDTA | Graph neural network | Must have |
| DrugBAN | Bilinear attention | Must have |
| Random Forest | Traditional ML | Must have |
| XGBoost | Traditional ML | Should have |
| DTI-LM | Language model-based | Nice to have |
| EviDTI | Evidential/uncertainty | Nice to have |

**LLM Baselines (all free — minimum for submission):**

| Model | Access | Priority |
|-------|--------|----------|
| Gemini 2.5 Flash | Free API | Must have |
| Llama 3.3 70B | Ollama local | Must have |
| Mistral 7B | Ollama local | Must have |
| Phi-3.5 3.8B | Ollama local | Should have |
| Qwen2.5 7B | Ollama local | Should have |
| Gemini 2.5 Flash-Lite | Free API | Nice to have |

**LLM flagship models (post-stabilization, NOT for NeurIPS sprint):**
- GPT-4/4.1, Claude Sonnet/Opus, Gemini Pro — added to leaderboard later

**Core validation experiments (minimum for paper):**
- [ ] Exp 1: NegBioDB confirmed negatives vs. random negatives (ML training comparison)
- [ ] Exp 4: Node degree bias quantification (DDB vs. random split performance gap)
- [ ] Exp 5: Cross-database consistency (agreement rate for overlapping pairs)
- [ ] Exp 7: Target class coverage analysis vs. existing benchmarks
- [ ] **Exp 9 (NEW): LLM vs. ML comparison on negative DTI prediction (L1 vs. M1)**
- [ ] **Exp 10 (NEW): LLM extraction quality (L2 entity F1 across models)**

**Additional experiments (strengthen paper):**
- [ ] Exp 2: Confidence tier discrimination
- [ ] Exp 3: Assay context dependency
- [ ] Exp 6: Temporal generalization
- [ ] Exp 8: LIT-PCBA recapitulation and extension
- [ ] Exp 11: Prompt strategy comparison (zero-shot vs few-shot vs CoT on LLM tasks)
- [ ] Exp 12: LLM-as-Judge reliability (kappa vs human annotations)

### Week 8-10: Paper Writing

- [ ] Write NeurIPS D&B paper (8 pages + unlimited appendix)
- [ ] Create key figures (see research/06_paper_narrative.md for figure plan)
- [ ] Python download script: `pip install negbiodb` or simple wget script
- [ ] Host dataset (Zenodo DOI + GitHub release)
- [ ] Author ethical statement

### Week 10-11: Review & Submit

- [ ] Internal review and polish
- [ ] Submit abstract (~May 1)
- [ ] Submit full paper (~May 15)
- [ ] Post ArXiv preprint (same day or before submission)

---

## Phase 1b: Post-Submission Expansion (Months 3-6)

### Data Expansion (if not at 10K+ for submission)
- [ ] Complete PubChem BioAssay extraction (full confirmatory set)
- [ ] LLM text mining pipeline activation (PubMed abstracts)
- [ ] Supplementary materials table extraction (pilot)

### Benchmark Refinement
- [ ] Add remaining baseline models
- [ ] Complete all 8 validation experiments
- [ ] Build public leaderboard (simple GitHub-based)

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
- [ ] Tutorial at relevant workshop (ICLR/ICML)
- [ ] Researcher incentive design (citation credit, DOI per submission)

### 2.3 Publication Strategy

| Month | Target | Type |
|-------|--------|------|
| 2-3 | ArXiv preprint | Establish priority (concurrent with NeurIPS) |
| 3 | NeurIPS 2026 D&B Track | Primary benchmark paper (~May 15) |
| 3-6 | Perspective paper | Publication bias in DTI (parallel) |
| 8-12 | J. Cheminformatics / Nature Sci Data | Database descriptor |
| 12-18 | NAR Database Issue | Contact editor July; publish January |

**Backup if NeurIPS rejects:** ICLR 2027 D&B (submission ~Oct 2026) or ICML 2027

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
| **M2: Negative Confidence Prediction** | (SMILES, sequence, assay features) | Gold/Silver/Bronze/Copper | Weighted F1, MCC |
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
- All LLM tasks evaluated in: zero-shot, 3-shot, 5-shot, CoT, CoT+3-shot
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
- [ ] Expand tasks: Failure Diagnosis, Experimental Design Critique, Literature Contradiction Detection
- [ ] Multi-modal: integrate protein structures, assay images
- [ ] LLM-specific evaluation tasks (reasoning about negative results)
- [ ] Regular leaderboard updates

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
DTI (Phase 1-3)
  │
  ├── Gene Function (CRISPR KO/KD negatives)
  │     └── Leverage CRISPR screen data, DepMap
  │
  ├── Clinical Trial Failure
  │     └── ClinicalTrials.gov terminated/failed trials
  │
  ├── Chemistry Domain Layer
  │     └── Failed reactions, yield = 0 data
  │
  └── Materials Science Domain Layer
        └── HTEM DB integration, failed synthesis conditions
```

---

## Key Milestones (Revised)

| Milestone | Target Date | Deliverable |
|-----------|------------|-------------|
| Schema v1.0 finalized | Week 2 (Mar 2026) | SQLite schema + standardization pipeline |
| MVP dataset (5K+ entries) | Week 4-6 (Apr 2026) | Curated negative DTI dataset |
| Baseline experiments complete | Week 8 (Apr 2026) | 4+ models × 3+ splits × 7 metrics |
| **ArXiv preprint** | **Week 9 (May 2026)** | **Priority establishment** |
| **NeurIPS 2026 submission** | **Week 11 (~May 15, 2026)** | **Benchmark paper** |
| Perspective paper submitted | Month 4-6 | Publication bias in DTI |
| Python library v0.1 | Month 8 | `pip install negbiodb` |
| NeurIPS decision | Month 7 (~Sep 2026) | Accept/reject notification |
| Web platform launch | Month 12 | Public access + leaderboard |
| Database descriptor paper | Month 8-12 | J. Cheminformatics or Nature Sci Data |
| NIH R24 funding | Month 12-18 | Multi-year sustainability |
| 100K+ entries | Month 24 | Scale milestone |
| NAR Database Issue | Month 24-30 | Gold standard DB recognition |

---

## Risk Assessment (Updated)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NeurIPS 2026 deadline too tight (11 weeks) | Medium-High | High | Focus on core experiments; ArXiv as fallback; ICLR 2027 backup |
| HCDT 2.0 license blocks integration | **Confirmed** | Medium | Already mitigated: independently extract from underlying sources |
| Insufficient data quality | Medium | High | Strict QC pipeline + confidence tiers |
| Low community adoption | Medium | High | TDC-style easy access + workshop tutorials |
| Competitive entry before NeurIPS | Low | Medium | First-mover advantage + ArXiv priority |
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
| [research/06_paper_narrative.md](research/06_paper_narrative.md) | Paper title/abstract, NeurIPS strategy, competitive positioning |
| [research/07a_llm_benchmark_landscape_survey.md](research/07a_llm_benchmark_landscape_survey.md) | Survey of existing bio/chem LLM benchmarks and evaluation methods |
| [research/07b_llm_benchmark_design.md](research/07b_llm_benchmark_design.md) | LLM benchmark tasks, evaluation methods, dual-track architecture |
