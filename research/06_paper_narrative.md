# Paper Narrative & Publication Strategy

> Title options, abstract structure, novelty argument, NeurIPS strategy, and competitive positioning (2026-03-02)

---

## 1. Title Options

### Primary Candidates

| # | Title | Style | Target Venue |
|---|-------|-------|-------------|
| 1 | **NegBioDB: A Curated Database and Benchmark for Experimentally Confirmed Negative Drug-Target Interactions** | Descriptive | NeurIPS D&B / J. Cheminformatics |
| 2 | **Beyond Assumed Negatives: NegBioDB Brings Experimental Rigor to Drug-Target Interaction Benchmarking** | Provocative | NeurIPS D&B |
| 3 | **The Missing Negatives: A Large-Scale Curated Resource of Confirmed Drug-Target Non-Interactions** | Problem-framing | Nature Scientific Data |
| 4 | **NegBioDB: Bridging the Negative Data Gap in Computational Drug Discovery** | Gap-focused | General |

**Recommended: Title #2** for NeurIPS (provocative, clearly states the problem and solution)
**Recommended: Title #1** for J. Cheminformatics / Nature Scientific Data (descriptive, precise)

---

## 2. Abstract Structure (NeurIPS D&B Format)

### Draft Abstract (~250 words)

> Machine learning models for drug-target interaction (DTI) prediction are trained and evaluated using benchmarks where "negative" examples — compound-target pairs with no interaction — are either assumed from untested pairs or randomly sampled from unknown interaction space. Since less than 1% of all possible drug-target combinations have been experimentally tested, this "untested = negative" assumption introduces systematic bias that inflates reported performance and obscures real-world generalization. Recent audits have shown that even carefully designed benchmarks like LIT-PCBA suffer from data leakage and methodological flaws.
>
> We introduce NegBioDB, the first large-scale database dedicated to **experimentally confirmed negative** drug-target interactions. NegBioDB integrates inactive results from PubChem BioAssay confirmatory screens, ChEMBL quantitative assays, BindingDB binding measurements, and literature extraction, encompassing [X]K curated negative DTI pairs across [Y] targets spanning kinases, GPCRs, ion channels, nuclear receptors, and emerging target classes. Each entry is assigned a hierarchical confidence tier (Gold through Copper) based on assay quality, dose-response confirmation, and replication.
>
> We present NegBioBench, a standardized evaluation suite with seven splitting strategies (including cold-compound, cold-target, temporal, and degree-balanced splits), early-enrichment metrics (LogAUC, BEDROC, EF@1%), and baselines from [N] established DTI models. We demonstrate that (1) models trained on experimentally confirmed negatives show significantly higher precision than those trained with random negatives, (2) existing benchmark performance is inflated by 10-20% due to node degree bias, and (3) assay context critically determines inactivity classification for 5-15% of compound-target pairs.
>
> NegBioDB is freely available under CC BY-SA 4.0 with a Python API, standardized downloads, and public leaderboards.

---

## 3. Novelty Argument (Reviewer Rebuttal Preparation)

### 3.1 "Why is this novel? PubChem already has inactive data."

**Response:**
PubChem contains ~270M inactive bioactivity records, but this data is:
1. **Uncurated**: No standardized inactivity criteria — each depositor defines their own thresholds
2. **Mixed quality**: Primary single-point screens (high false-negative rate) treated equally with confirmatory dose-response
3. **Not ML-ready**: No standardized splits, no benchmark tasks, no metadata standardization
4. **Missing cross-reference**: Same compound-target pair inactive in one assay may be active in another — PubChem does not reconcile these

NegBioDB is not a data dump — it is a **curated, standardized, confidence-tiered, ML-ready resource** with a benchmark suite. The relationship is analogous to UniProt vs. raw protein sequence databases, or ImageNet vs. raw image collections.

### 3.2 "What about ChEMBL? It has activity data."

**Response:**
ChEMBL's design is fundamentally positive-biased: it curates from published literature, which systematically over-reports active compounds. Only ~133K of 21M records are explicitly labeled "Not Active." ChEMBL's absence of a compound-target pair means "untested," not "inactive." NegBioDB specifically addresses this gap by centering the resource on confirmed negatives.

### 3.3 "How do you handle the inherent noise in negative data?"

**Response:**
Through our hierarchical confidence tier system:
- **Gold**: Multiple orthogonal assays, full dose-response, Z' > 0.5, replicated
- **Silver**: Single confirmed dose-response, validated assay quality
- **Bronze**: Single-point screening inactive, established assay platform
- **Copper**: Literature-extracted, LLM-mined, awaiting experimental validation

Researchers can filter by confidence tier. Our validation experiments demonstrate that Gold-tier negatives provide more reliable training signal, while including lower tiers increases coverage without degrading precision when appropriately weighted.

### 3.4 "Is this just a data aggregation exercise?"

**Response:**
No. Our contributions beyond aggregation include:
1. **Confidence tier system** — novel quality-aware categorization of negative evidence
2. **Cross-database reconciliation** — resolving conflicting activity labels across sources with assay context
3. **Assay-context-aware inactivity** — capturing conditional negatives (compound inactive under condition A, active under condition B)
4. **Degree-balanced benchmark design** — addressing the node degree bias that invalidates existing DTI benchmarks
5. **Standardized negative evaluation protocol** — early enrichment metrics and prospective-style temporal splits

### 3.5 "What impact will this have?"

**Response:**
1. **Methodological**: Every DTI prediction paper currently uses assumed negatives. NegBioDB enables true performance assessment.
2. **Practical**: Pharma companies waste resources pursuing computationally predicted "hits" that were already experimentally shown to be inactive. NegBioDB directly reduces this waste.
3. **Scientific**: Negative results contain biological information (why a compound doesn't bind) that is lost when negatives are randomly sampled.
4. **Community**: Provides infrastructure and incentives for depositing negative results, addressing the publication bias problem.

---

## 4. NeurIPS 2026 Datasets & Benchmarks Strategy

### 4.1 Timeline

| Date (Estimated) | Event | Action |
|-------------------|-------|--------|
| **March-April 2026** | Development sprint | Build MVP dataset + benchmark |
| **~May 1, 2026** | Abstract deadline | Submit 1-page abstract |
| **~May 15, 2026** | Paper deadline | Submit full paper (8 pages + unlimited appendix) |
| **~Aug 2026** | Reviews returned | Address reviewer feedback |
| **~Sep 2026** | Decision notification | Accept/reject |
| **~Dec 2026** | NeurIPS 2026 conference | Present paper |

**From today (March 2, 2026): ~10-11 weeks to paper deadline.**

### 4.2 NeurIPS D&B Mandatory Requirements

| Requirement | Status | Action Needed |
|-------------|--------|---------------|
| Data downloadable by reviewers at submission | Not started | Must have hosted dataset by May |
| Croissant machine-readable metadata | Not started | Generate Croissant JSON-LD |
| Code publicly available by camera-ready | Not started | GitHub repo (already exists) |
| Datasheet for dataset | Not started | Write using Gebru et al. template |
| Author statement on ethical/legal use | Not started | Write; address license compliance |
| Hosting plan beyond 3 years | Not started | GitHub + Zenodo DOI |

### 4.3 What to Prioritize for NeurIPS Submission

**Must Have (for submission):**
- [ ] MVP dataset: ≥5,000 curated negative DTIs from PubChem + ChEMBL
- [ ] Confidence tier assignment for all entries
- [ ] 3+ splitting strategies implemented (random, cold compound, cold target)
- [ ] 3+ baseline models evaluated (DeepDTA, GraphDTA, DrugBAN)
- [ ] Core metrics: LogAUC, AUPRC, MCC (minimum)
- [ ] Experiment 1 (NegBioDB vs. random negatives) completed
- [ ] Experiment 4 (node degree bias) completed
- [ ] Python download script (`pip install negbiodb` or simple script)
- [ ] Croissant metadata
- [ ] Datasheet

**Nice to Have (strengthens paper):**
- [ ] 10K+ entries
- [ ] All 7 splitting strategies
- [ ] All 8 validation experiments
- [ ] Web interface
- [ ] LLM text mining pipeline results
- [ ] All 7 baseline models

**Can Defer to Camera-Ready:**
- [ ] Community submission portal
- [ ] Full web platform
- [ ] Leaderboard system

### 4.4 NeurIPS D&B Review Criteria (From Past Calls)

1. **Utility of the dataset/benchmark** — Does it fill a genuine gap?
   - Our case: Strong. The "assumed negative" problem is widely documented.

2. **Quality of construction** — Is it carefully built?
   - Our case: Confidence tiers, cross-DB validation, standardized IDs.

3. **Novelty** — Is this new?
   - Our case: First dedicated curated negative DTI resource with benchmark.

4. **Documentation** — Datasheet, Croissant, clear access.
   - Must complete.

5. **Potential for broad impact** — Will the community use it?
   - Our case: Strong. Every DTI prediction paper needs negatives.

6. **Ethical considerations** — Data provenance, license compliance.
   - Our case: All public domain or CC BY-SA sources. No patient data.

---

## 5. ArXiv Preprint Strategy

### 5.1 Timeline

| Target | Action |
|--------|--------|
| **April 2026** | Post initial ArXiv preprint (before NeurIPS submission) |
| Purpose | Establish priority; solicit community feedback; get citations before review |

### 5.2 ArXiv vs. NeurIPS Paper

The ArXiv preprint can be the same paper submitted to NeurIPS. NeurIPS explicitly allows concurrent ArXiv posting. Benefits:
- Priority timestamp
- Community can cite immediately
- Feedback may improve final paper
- Visibility even if NeurIPS rejects (resubmit to ICLR 2027 or ICML 2027)

---

## 6. Competitive Positioning (Updated March 2026)

### 6.1 No Direct Competitor Exists

As of March 2026, no published resource matches NegBioDB's scope:

| Existing Resource | Gap NegBioDB Fills |
|-------------------|--------------------|
| PubChem BioAssay | Raw, uncurated, no ML-ready format, no benchmark |
| ChEMBL | Positive-biased; ~133K negatives buried in 21M records |
| ExCAPE-DB | Outdated (2017); no updates in 9 years |
| LIT-PCBA | Only 15 targets; data leakage discovered (2025 audit) |
| InertDB | Only 3,205 compounds; universally inactive (not target-specific) |
| HCDT 2.0 | Non-derivative license; no benchmark; no ML-ready format |
| WelQrate | Quality curation pipeline (positive + negative); not negative-dedicated |
| TDC | Random negatives only; no confirmed inactives |

### 6.2 Key Differentiators (Elevator Pitch)

"NegBioDB is the first resource that treats experimentally confirmed negative drug-target interactions as **first-class citizens** rather than afterthoughts. While PubChem buries inactive data in uncurated dumps and ChEMBL barely acknowledges it, NegBioDB curates, standardizes, confidence-scores, and benchmarks negative DTI data at scale. Every compound-target pair in NegBioDB was experimentally tested and found inactive — no assumptions, no random sampling."

### 6.3 Potential Competitors to Watch

| Potential Competitor | Risk | Mitigation |
|---------------------|------|------------|
| TDC v2 adding curated negatives | Medium | Move fast; establish first; deeper curation |
| ChEMBL adding negative flags (v37+) | Low | ChEMBL is source, not competitor; we add value on top |
| Pharma consortium releasing negative data | Low-Medium | Welcome integration; position as the public infrastructure |
| WelQrate expanding to negatives | Low | Different scope (quality curation vs. negative-dedicated) |

---

## 7. Multi-Paper Strategy

### 7.1 Paper 1: Database + Benchmark (Primary)
- **Venue:** NeurIPS 2026 D&B (primary) → ICLR 2027 (backup)
- **Content:** NegBioDB description + NegBioBench + validation experiments
- **Timeline:** May 2026 submission

### 7.2 Paper 2: Data Descriptor
- **Venue:** Nature Scientific Data or J. Cheminformatics
- **Content:** Detailed data curation methodology, FAIR compliance, schema design
- **Timeline:** Month 8-12 (after NeurIPS decision)
- **Key:** Different framing — data reuse focus, not ML benchmark focus

### 7.3 Paper 3: Perspective / Review
- **Venue:** Briefings in Bioinformatics or Drug Discovery Today
- **Content:** Publication bias in DTI prediction; the "assumed negative" problem; call to action
- **Timeline:** Can write in parallel (Month 3-6)
- **Purpose:** Frame the problem + cite NegBioDB as the solution

### 7.4 Paper 4: NAR Database Issue
- **Venue:** Nucleic Acids Research (Database Issue)
- **Content:** Web platform description, community features, update metrics
- **Timeline:** Contact editor by July 2027; publish January 2028
- **Prerequisite:** Web platform must be live and demonstrably used

### 7.5 Paper 5: High-Impact Follow-Up
- **Venue:** Nature Chemical Biology / Nature Methods
- **Content:** Novel biological insights from negative data analysis (e.g., systematic selectivity patterns, target class-specific inactivity signatures)
- **Timeline:** Month 18-24

---

## 8. Key Figures for the Paper

### 8.1 Figure 1: The Problem
- Panel A: Venn diagram showing <1% of DTI space is experimentally tested
- Panel B: How existing benchmarks generate negatives (random sampling illustration)
- Panel C: Performance inflation — same model, assumed vs. confirmed negatives

### 8.2 Figure 2: NegBioDB Overview
- Panel A: Data sources and pipeline architecture
- Panel B: Confidence tier system with examples
- Panel C: Database statistics (targets, compounds, pairs by tier and target class)

### 8.3 Figure 3: NegBioBench Evaluation
- Panel A: Baseline model comparison across 7 metrics
- Panel B: Performance across splitting strategies (heatmap)
- Panel C: NegBioDB vs. random negatives (Experiment 1 results)

### 8.4 Figure 4: Insights from Negative Data
- Panel A: Node degree bias quantification (Experiment 4)
- Panel B: Assay context-dependent inactivity (Experiment 3)
- Panel C: Target class coverage comparison vs. existing benchmarks (Experiment 7)

---

## 9. Related Work Section Outline

1. **DTI Prediction Landscape** — Brief survey of methods (DeepDTA → DrugBAN → LLM-based)
2. **The Negative Data Problem** — Publication bias, "untested = negative," performance inflation
3. **Existing Benchmarks** — MoleculeNet, TDC, DAVIS, DUD-E, LIT-PCBA — and their limitations
4. **Recent Negative Data Resources** — HCDT 2.0, InertDB, ExCAPE-DB, WelQrate
5. **Quality-Aware Data Curation** — BAO ontology, WelQrate hierarchical QC, Z-factor standards
6. **Positioning NegBioDB** — First to combine: (a) experimentally confirmed, (b) curated + tiered, (c) ML-ready benchmark, (d) multi-source, (e) diverse target classes

---

## Sources

- NeurIPS 2025 D&B Call: https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks
- Croissant metadata: https://github.com/mlcommons/croissant
- Datasheets for Datasets (Gebru et al.): https://arxiv.org/abs/1803.09010
- TDC NeurIPS 2021: https://arxiv.org/abs/2102.09548
- WelQrate NeurIPS 2024: https://arxiv.org/abs/2411.09820
- LIT-PCBA 2025 audit: https://arxiv.org/html/2507.21404v1
- EviDTI 2025: https://www.nature.com/articles/s41467-025-62235-6
- Node degree bias 2025: https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-025-02231-w
