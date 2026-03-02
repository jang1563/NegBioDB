# Publication & Commercial Strategy

> Academic publication strategy, funding, commercialization, and competitive landscape (2026-03-02)

---

## 1. Academic Publication Venues

### Tier 1: Primary Targets

#### NeurIPS Datasets and Benchmarks Track
- **Fit: Highest** (primary target for benchmark paper)
- 2023: 163 papers accepted. Growing AI-for-science representation
- **Requirements:**
  - Data downloadable by reviewers at submission time
  - Croissant machine-readable metadata (mandatory since 2025)
  - Code/data publicly available by camera-ready
- **Precedents:** TDC (NeurIPS 2021), WelQrate (NeurIPS 2024), Lo-Hi (NeurIPS 2023)
- **Submission:** Typically mid-May (abstract) → 4 days later (full paper)
- **Success factors:** Clear gap identification, leaderboards, standardized splits, reproducibility

#### Nature Scientific Data
- **Fit: High** (database descriptor paper)
- Impact factor: 6.92 (2024)
- "Data Descriptors" format: promotes data reuse, not hypothesis testing
- FAIR-compliant public repository deposit required
- Technical jargon minimized (accessible to any scientist)

#### Nucleic Acids Research (NAR) Database Issue
- **Fit: High** (after web DB is built)
- Most prestigious venue for biological database papers
- 2026: 182 papers (84 new, 86 updates)
- **Invitation only**: Pre-submission inquiry to Executive Editor by July 1
- Free web access required (no login/registration)
- Must clearly differentiate from similar resources

### Tier 2: Complementary Venues

| Venue | Type | Best For |
|-------|------|----------|
| **Journal of Cheminformatics** | Database article / Data Note | DB technical description. Full reproducibility required |
| **Bioinformatics (Oxford)** | Application Note (4p) / Original Paper | DB/web service description |
| **ICLR/ICML Workshops** | Workshop paper | Early visibility. MLGenX, Large Drug Discovery Models, etc. |
| **Briefings in Bioinformatics** | Review/Perspective | Publication bias framing paper |
| **PLOS ONE** | Journal | Explicitly accepts negative results; broad reach |
| **Database (Oxford)** | Journal | Biological DB specialist venue |

### What Makes DB/Benchmark Papers Successful

1. **Clear gap identification** — "experimentally confirmed negatives are not available" (widely cited)
2. **Rigorous curation** — WelQrate's hierarchical curation was key to acceptance
3. **Easy programmatic access** — TDC's "3 lines of code" accessibility
4. **Standardized evaluation** — Leaderboards, canonical splits, defined metrics
5. **Community infrastructure** — GitHub, docs, tutorials
6. **Sustained maintenance** — ZINC: 5 major versions over 20 years

---

## 2. Model Projects to Emulate

### TDC (Therapeutics Data Commons)

| Stage | Timing | Content |
|-------|--------|---------|
| ArXiv preprint | Feb 2021 | Priority + community feedback |
| NeurIPS 2021 | Dec 2021 | Datasets & Benchmarks Track (inaugural) |
| Nature Chemical Biology | 2022 | Higher-impact follow-up |
| NeurIPS 2024 | 2024 | AIDrugX paper (extension) |

**Scale:** 66 datasets, 22 tasks, 29 leaderboards
**Key success:** `pip install PyTDC` + public leaderboards + meaningful splits + Harvard backing

### MoleculeNet
- Chemical Science 2018. 1,800+ citations
- 16 datasets across multiple domains
- Integrated with DeepChem open-source library
- **Timing was key**: Arrived when the community desperately needed a standard

### ZINC Database
- 20 years of iterative expansion: ZINC (728K) → ZINC-22 (54.9B)
- **Lesson:** Start focused and high-quality, scale iteratively
- >85% purchasing success rate; UCSF institutional backing

### Open Targets
- **Public-private partnership**: Wellcome Sanger, EMBL-EBI + GSK, BMS, Sanofi, Pfizer
- Wellcome Trust + pharma contributions
- **CC0 license** (maximally open)
- **Lesson:** Pharma will invest in pre-competitive infrastructure that reduces their costs

---

## 3. Funding Opportunities

### NIH (Best Fit)

| Program | Mechanism | Amount | Deadlines | Fit |
|---------|-----------|--------|-----------|-----|
| **PAR-23-236** | R24 | $350K/yr × 4yr | Sep 2025, Jan 2026 | **Optimal.** Early-stage data repository. No wet-lab. FAIR required |
| PAR-23-237 | U24 | Varies | Sep 2025, Jan 2026 | For established repositories (post-launch) |
| PAR-25-238 (NLM) | R01 | Standard | Rolling | Broader data science scope |

**Note:** PAR-23-236/237 proposals with wet-lab/primary data generation are **returned without review**

### NSF

| Program | Amount | Notes |
|---------|--------|-------|
| IDSS (Integrated Data Systems & Services) | Varies | National-scale data infrastructure for AI-driven research |
| Capacity: Cyberinfrastructure | Varies | Explicitly supports "databases from new or existing data sources" |

### Foundation & Other

| Source | Features |
|--------|----------|
| **CZI (Chan Zuckerberg Initiative)** | Open science infrastructure + open-source tools. Now part of Biohub (2025) |
| **Wellcome Trust** | Accessible through partnership grants. Funded OpenAlex (2.9M GBP) |
| **ARPA-H** | $1.5B+ budget. Favors bold moonshots. Frame as "eliminating systematic bias in AI drug discovery" |

---

## 4. Commercial Models

### How Bio Databases Monetize

| Model | Description | Example |
|-------|-------------|---------|
| **Freemium** | Core free, premium features paid | DrugBank |
| **SaaS subscription** | Monthly/annual platform access | Drug discovery SaaS: $10.8B (2025) → $42.9B (2035) |
| **Data licensing** | Curated datasets to pharma | IQVIA, Clarivate |
| **API tiers** | Free (rate-limited) / Paid (bulk) | Standard pattern |
| **Consulting** | Bespoke analytics using DB expertise | Specialized DBs |
| **Consortium membership** | Pharma pays membership for pre-competitive access | Open Targets model |
| **Insight-as-a-Service** | Sell analytics/insights, not raw data | Higher margins, lower privacy risk |

### Reference: AI Drug Discovery Company Strategies

| Company | Strategy | Revenue Model |
|---------|----------|---------------|
| **Recursion** | 50+ PB proprietary data (phenomics, transcriptomics) | Platform licensing + partnerships |
| **Insilico Medicine** | End-to-end Pharma.AI platform | Mega-partnerships ($1.2B Sanofi) |
| **Relay Therapeutics** | Dynamo platform (protein dynamics) | Internal pipeline |
| **Insitro** | HTS phenotypic screening + ML | Pharma partnerships (BMS, etc.) |

**Key insight:** Most successful companies treat data as a **strategic moat.** NegBioDB's commercial value = curation expertise + negative data completeness/quality.

---

## 5. IP & Licensing

### Recommended Approach

| Component | License | Rationale |
|-----------|---------|-----------|
| Core data | **CC BY-SA 4.0** | Required by ChEMBL CC BY-SA 3.0 viral clause (one-way upgrade 3.0→4.0). See [research/05_technical_deep_dive.md](research/05_technical_deep_dive.md) §2 for full license analysis |
| Value-added (API, analytics) | Commercial license | Revenue generation (separate from core data) |
| Open-source code | MIT or Apache 2.0 | Community contributions |

### Dual-track Model

```
Open Track (Academic/Community):
├── Core dataset: CC BY-SA 4.0 (required by ChEMBL viral clause)
├── Python library: MIT license
├── Web interface: free access
└── Leaderboards: open participation

Commercial Track (Pharma/Enterprise):
├── Bulk API access: subscription
├── Custom curation: consulting fees
├── Early access: consortium membership
├── Trained models: licensing
└── Analytics dashboard: SaaS
```

---

## 6. Competitive Landscape

### Current Resources Addressing Negative Data

| Project | Focus | How NegBioDB Differs |
|---------|-------|---------------------|
| PubChem BioAssay | 91.4% inactive (raw, uncurated) | Curated + ML-ready |
| InertDB (2025) | 3,205 universally inactive compounds | Target-specific negatives |
| LCIdb (2024) | Balanced negatives (random sampled) | Experimentally confirmed |
| WelQrate (2024) | Quality curation pipeline | Dedicated to negatives |
| ChEMBL / BindingDB | Predominantly positive | Negative-centric |

### NegBioDB Differentiators

1. **Dedicated to experimentally confirmed negatives** — no other resource centers this
2. **Hierarchical quality tiers** (Gold/Silver/Bronze/Copper)
3. **ML-ready formatting with proper negative sampling** — addresses #1 methodological complaint in DTI prediction
4. **Standardized benchmark with balanced positive/negative representation**
5. **Provenance & confidence scoring** — tracks how each negative was determined
6. **Systematic publication bias infrastructure** — incentives for depositing negative results

### Potential Objections and Responses

| Objection | Response |
|-----------|----------|
| "PubChem already has inactive data" | Raw, uncurated, not ML-ready. Curation is the value-add |
| "Negative results are noisy" | Confidence tiers + hierarchical validation (WelQrate approach) |
| "Limited academic interest" | Publication bias is a hot topic; ML community actively demanding better negative data |
| "Pharma won't share data" | Start with public data (PubChem, ChEMBL) → build platform → earn trust → pharma partnerships |

---

## 7. Recommended Publication Strategy (Timeline)

### Phase 1: Establish Credibility (Month 0-12)
1. **Workshop paper** at ICLR/ICML (MLGenX, etc.) — low barrier, high ML community visibility
2. **ArXiv preprint** — establish priority + community feedback

### Phase 2: Flagship Publication (Month 6-18)
3. **NeurIPS Datasets & Benchmarks** — primary benchmark venue (May submission → Sep decision)
4. **J. Cheminformatics Database** or **Nature Scientific Data** — database descriptor

### Phase 3: Establish as Standard (Month 12-30)
5. **NAR Database Issue** — contact Editor by July → next January publication
6. **Nature Chemical Biology** or equivalent — high-impact follow-up (TDC trajectory)

### Parallel Track
- **Perspective paper** on publication bias in DTI (Briefings in Bioinformatics / Drug Discovery Today) — frame the problem + cite NegBioDB as the solution

---

## 8. Recommended Funding Strategy

| Timeline | Source | Purpose |
|----------|--------|---------|
| **Month 0-12** | NIH PAR-23-236 (R24) | Early-stage repository. $350K/yr × 4yr. **Top priority** |
| **Month 0-12** | CZI Open Science | Open-source tooling/infrastructure |
| **Month 12-24** | NSF IDSS / Cyberinfrastructure | Scaling |
| **Month 12-24** | Industry partnerships | Pre-competitive consortium (Open Targets model) |
| **Month 24+** | NIH PAR-23-237 (U24) | Established repository maintenance |
| **Month 24+** | ARPA-H | Transformative infrastructure framing |

---

## 9. Recommended Commercialization Roadmap

### Phase 1 — Build Adoption (Open)
- Core dataset: CC BY-SA 4.0
- Python library (`pip install negbiodb`)
- Free web interface
- Public leaderboards

### Phase 2 — Introduce Premium (Hybrid)
- Tiered API (free rate-limited → paid bulk/enterprise)
- Premium analytics dashboard
- Custom curation for pharma
- Consortium early access

### Phase 3 — Scale Revenue
- Pre-competitive consortium (pharma membership)
- Insight-as-a-Service (trained models, predictive analytics)
- Data licensing for commercial ML pipelines
- Consulting on negative-data-aware model training

---

## Sources

- NeurIPS 2025 D&B Call: https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks
- Nature Scientific Data: https://www.nature.com/sdata/submission-guidelines
- NAR Database Issue: https://academic.oup.com/nar/pages/ms_prep_database
- TDC: https://tdcommons.ai/ | NeurIPS 2021
- WelQrate NeurIPS 2024: https://arxiv.org/abs/2411.09820
- MoleculeNet: https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/
- ZINC-22: https://pubs.acs.org/doi/10.1021/acs.jcim.2c01253
- Open Targets: https://www.opentargets.org/
- NIH PAR-23-236: https://grants.nih.gov/grants/guide/pa-files/PAR-23-236.html
- CZI Open Science: https://chanzuckerberg.com/science/programs-resources/open-science/
- ARPA-H: https://arpa-h.gov/explore-funding/programs
- Drug Discovery SaaS Market: https://www.towardshealthcare.com/insights/drug-discovery-saas-platforms-market-sizing
