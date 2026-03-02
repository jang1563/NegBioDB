# NegBioDB — Execution Roadmap

> Last updated: 2026-03-02

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
- DB: Supabase free / SQLite local
- Web: Vercel free tier
- Storage: GitHub LFS / S3 free tier (<5GB)
- CI/CD: GitHub Actions free

### Only Paid Cost: Publication
- OA APC: ~$2,500-3,000 (J. Cheminformatics or Nature Sci Data)
- Conference registration: ~$200-400

---

## Phase 1: Foundation & MVP (Months 0-6)

### 1.1 Data Pipeline MVP

**Goal:** Build initial curated dataset of ~5,000-10,000 experimentally confirmed negative DTIs

**Data Sources (Priority Order, All Free):**

| Source | Target Volume | Method |
|--------|--------------|--------|
| HCDT 2.0 | 38,653 negative DTIs (seed set) | Direct integration (> 100 uM threshold) |
| PubChem BioAssay confirmatory | ~50K+ inactive dose-response | API extraction, filter confirmatory only |
| ChEMBL "Not Active" | ~133K inactive records | SQL/API query (activity_comment + pChEMBL < 5) |
| InertDB | 3,205 universally inactive compounds | Direct integration |
| DAVIS complete matrix | ~27K entries (full negative matrix) | Direct integration as gold standard |

**Key Technical Tasks:**
- [ ] Design database schema (see Schema Design below)
- [ ] Build PubChem BioAssay extraction pipeline (confirmatory > primary)
- [ ] Build ChEMBL extraction pipeline (activity_comment + pChEMBL threshold)
- [ ] Implement confidence tier assignment (Gold/Silver/Bronze/Copper)
- [ ] Integrate BAO ontology for assay standardization
- [ ] Integrate DTO for target classification
- [ ] Implement compound standardization (InChIKey, canonical SMILES)
- [ ] Build QC pipeline (Z-factor check, PAINS filter, cross-DB validation)
- [ ] Set up LLM text mining pipeline (Mistral 7B local + Gemini free tier)
- [ ] Build PubMed abstract coarse filter (negative DTI result detection)
- [ ] Build fine extraction prompts (structured JSON output)

### 1.2 Schema Design

**Common Layer:**
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

**Biology/DTI Domain Layer:**
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

### 1.3 Initial Benchmark Design (NegBioBench v0.1)

**Task Types:**

| Task | Input | Output | Metric |
|------|-------|--------|--------|
| **DTI Binary Prediction** | (compound SMILES, target sequence) | Interacting / Non-interacting | AUPRC, AUROC |
| **Negative Confidence Prediction** | (compound, target, assay context) | Confidence tier | Weighted F1 |
| **Counterfactual Reasoning** | Negative result + condition change | Predicted new outcome | Accuracy + reasoning quality |

**Splitting Strategies:**
- Random split
- Cold compound split
- Cold target split
- Cold compound + target split
- Temporal split (by publication date)
- Degree-distribution-balanced split (address node degree bias)

---

## Phase 2: Community & Publication (Months 6-18)

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
| 3-6 | ArXiv preprint | Establish priority |
| 6-9 | ICLR/ICML workshop paper | Early visibility in ML community |
| 9-12 | NeurIPS D&B Track | Primary benchmark paper (May submission) |
| 12-15 | J. Cheminformatics / Nature Sci Data | Database descriptor |
| 15-18 | Briefings in Bioinformatics | Perspective on publication bias in DTI |

### 2.4 Funding Applications

| Month | Target | Amount |
|-------|--------|--------|
| 3-6 | NIH PAR-23-236 (R24) | Up to $350K/yr × 4yr |
| 6-9 | CZI Open Science | Varies |
| 12-18 | NSF IDSS / Cyberinfrastructure | Varies |

---

## Phase 3: Scale & Sustainability (Months 18-36)

### 3.1 Data Expansion
- [ ] Expand to 100K+ curated negative DTIs
- [ ] Add LLM-based literature mining pipeline (NLP extraction from PubMed/PMC)
- [ ] Supplementary materials table extraction
- [ ] Integrate with Target 2035 AIRCHECK data as it becomes available
- [ ] Begin Gene Function (KO/KD) negative data collection (domain expansion Phase 1)

### 3.2 Benchmark Evolution (NegBioBench v1.0)
- [ ] Expand task types: Failure Diagnosis, Experimental Design Critique, Literature Contradiction Detection
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

### Expansion Path

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

## Key Milestones

| Milestone | Target Date | Deliverable |
|-----------|------------|-------------|
| Schema v1.0 finalized | Month 2 | Database schema + ontology mappings |
| MVP dataset (5K+ entries) | Month 4 | Curated negative DTI dataset |
| ArXiv preprint | Month 6 | Priority establishment |
| Python library v0.1 | Month 8 | `pip install negbiodb` |
| NeurIPS submission | Month 12 | Benchmark paper |
| Web platform launch | Month 14 | Public access |
| NIH R24 funding | Month 12-18 | Multi-year sustainability |
| 100K+ entries | Month 24 | Scale milestone |
| First pharma partnership | Month 24-30 | Commercial validation |
| NAR Database Issue | Month 30 | Gold standard DB recognition |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Insufficient data quality | Medium | High | Strict QC pipeline + confidence tiers |
| Low community adoption | Medium | High | TDC-style easy access + workshop tutorials |
| Competitive entry | Low | Medium | First-mover advantage + deep curation expertise |
| Funding gap | Medium | High | Multiple funding sources + early commercial track |
| Schema over-engineering | Medium | Medium | Start minimal, iterate based on user feedback |
| Pharma resistance to sharing | High | Medium | Start with public data; build trust first |
