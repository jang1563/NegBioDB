# NeurIPS Datasets & Benchmarks Track: Comprehensive Research Report

*Research conducted: March 2, 2026*

---

## 1. Exact Submission Requirements for NeurIPS 2025 D&B Track

### Page Limits & Format
- **9 content pages maximum** (including all figures and tables)
- References, checklist, and optional technical appendices do NOT count toward page limit
- Camera-ready version: +1 additional content page permitted (10 total)
- Must use the **NeurIPS 2025 LaTeX template** (`neurips_2025.sty`)
  - Use `\usepackage[preprint]{neurips_2025}` for single-blind D&B submissions
- Text confined within 5.5 inches (33 picas) wide x 9 inches (54 picas) long
- Left margin: 1.5 inches (9 picas)
- 10pt type with 11pt vertical spacing
- **Violations of style/format = rejection without review**

### Review Model
- **Single-blind by default** (author identities visible to reviewers)
- Authors may opt for double-blind if proper anonymization is feasible
- Reviews are published post-decision

### Key Deadlines (NeurIPS 2025)
| Milestone | Date |
|-----------|------|
| Abstract deadline | May 11, 2025 AoE |
| Full paper + data/code | May 15, 2025 AoE |
| Supplemental materials | May 22, 2025 AoE |
| Author notification | Sep 18, 2025 AoE |
| Camera-ready | Oct 23, 2025 AoE |

### Mandatory Items

#### 1. Croissant Metadata (NOW REQUIRED - was optional in 2024)
- Must include Croissant machine-readable JSON file with submission
- Validate using: https://huggingface.co/spaces/JoaquinVanschoren/croissant-checker
- Auto-generated on Kaggle, OpenML, HuggingFace, Dataverse
- Must self-generate for custom hosting sites
- Minimum fields: license info, dataset descriptions, URLs
- **Invalid/inaccessible Croissant file = desk rejection**
- Flexibility: Can provide dataset-level metadata only when file types aren't Croissant-compatible
- Multiple datasets: submit individual .json files combined in a .zip folder

#### 2. Dataset & Code Submission
- Datasets and code are **NOT supplementary materials** -- must be submitted in full and final form by May 15
- Code must be documented and executable
- **Non-compliance justifies desk rejection**
- Host on: GitHub, Bitbucket, or submit as supplementary ZIP

#### 3. Data Hosting on Persistent Public Repositories
Approved platforms:
- Harvard Dataverse (supports private URLs for review)
- Kaggle (supports private URLs for review)
- Hugging Face
- OpenML
- Custom/bespoke sites (must guarantee long-term access + provide Croissant)

Requirements:
- Must be publicly available by camera-ready deadline (Oct 23, 2025)
- **Failure to release = removal from conference and proceedings**
- Private datasets cannot be listed as contributions
- Hold-out test sets may remain private while releasing public portions

#### 4. Supplementary Documentation
Encouraged frameworks (not strictly mandatory but expected):
- **Datasheets for Datasets** (Gebru et al.)
- Dataset Nutrition Labels
- Data Statements for NLP
- Data Cards
- Accountability Frameworks

#### 5. Credentialized/Gated Access
- Permitted only when necessary for public good
- Must be open to large populations, provide rapid access, have long-term guarantees
- Misuse of DTA/restricted access = desk rejection

#### 6. Ethics & Responsible Use
- Author liability statement required
- Data license confirmation required
- Must address ethical implications with guidelines for responsible use

### Scope (What's In-Scope)
- New or redesigned dataset collections
- Data generators and RL environments
- Data quality/utility measurement tools
- Collection and curation practices
- Dataset audits and responsible development frameworks
- Benchmarks and benchmarking tools
- ML challenge analyses
- Novel dataset evaluations
- Advanced practices in data collection and curation
- Frameworks for responsible dataset development
- Systematic analyses of existing systems on novel datasets

### Dual Submissions
- **Dual submissions to both main track AND D&B track = desk rejection**

---

## 2. What Changed from 2024 to 2025

| Feature | 2024 | 2025 |
|---------|------|------|
| Croissant metadata | Encouraged/recommended | **MANDATORY** |
| Review model | Single-blind (optional anonymity) | Single-blind (same, more explicit) |
| Data hosting | Flexible | **Must use persistent public repositories** |
| Alignment with main track | Separate standards | **Aligned with main track** review processes |
| Reviewer recruitment | Separate | **Joint recruitment** with main track |
| Automated compliance | None | **Automated Croissant checklists** in OpenReview |
| Scoring system | 1-10 scale | **New 1-6 scale** (aligned with main track) |
| Page limit | 9 pages | 9 pages (unchanged) |

Key evolution summary:
- Croissant moved from *encouraged* to *required* due to improved tooling maturity
- Strategic alignment with main track ensures D&B papers held to same rigor as main track proceedings
- Introduction of automated metadata reports and compliance checklists
- Joint reviewer/AC recruitment with main track

---

## 3. Acceptance Statistics

### Historical D&B Track Data

| Year | Submissions | Accepted | Acceptance Rate |
|------|-------------|----------|-----------------|
| 2021 | ~170 | ~60 | ~35% |
| 2022 | ~480 | ~180 | ~37% |
| 2023 | 985-987 | 322 | 32.7% |
| 2024 | 1,820 | 459 | 25.2% |
| 2025 | 1,995 | ~497* | ~25%* |

*Note: PaperCopilot reported 591 papers with public scores (497 accepted, 84.09%), but this only reflects papers with public review scores. The actual total submission count was 1,995 with ~25% acceptance based on chair blog posts. The 84% figure is an artifact of only counting papers with published scores.*

### 2024 Breakdown by Presentation Type
| Category | Count | % of Total |
|----------|-------|-----------|
| Oral | 11 | 0.60% |
| Spotlight | 56 | 3.08% |
| Poster | 392 | 21.54% |

### 2024 Review Metrics
- Average rating for accepted papers: 6.58 (on old 1-10 scale)
- Rating range: 3.30-9.30
- Orals average: 7.60 (range 6.00-9.30)
- Posters average: 6.61 (range 5.60-8.00)
- Reviewer confidence: 4.33/5 average

### Dominant Topics
- **LLM evaluation** (primary focus area in 2024-2025)
- AI for science applications
- Domain-specific implementations
- Socially beneficial AI initiatives
- 84% of accepted papers introduced new datasets as part of benchmark contributions

### Review Infrastructure (2025)
- 41 Senior Area Chairs
- 281 Area Chairs
- 2,680 Reviewers
- No more than 3 submissions per reviewer
- No more than 5 submissions per AC

---

## 4. Review Criteria & Rubric

### NeurIPS 2025 Overall Score Scale (1-6)
| Score | Label | Description |
|-------|-------|-------------|
| 6 | Strong Accept | Technically flawless, groundbreaking impact, exceptional evaluation/reproducibility |
| 5 | Accept | Technically solid, high impact on at least one sub-area, good-to-excellent evaluation |
| 4 | Borderline Accept | Technically solid, reasons to accept outweigh reasons to reject |
| 3 | Borderline Reject | Technically solid, reasons to reject outweigh reasons to accept |
| 2 | Reject | Technical flaws, weak evaluation, inadequate reproducibility |
| 1 | Strong Reject | Well-known results or unaddressed ethical considerations |

### Confidence Score (1-5)
- 5: Absolutely certain, very familiar with related work, checked details carefully
- 4: Confident but not absolutely certain
- 3: Fairly confident, may have missed some parts

### 12 Review Dimensions
1. **Summary and Contributions** - concise overview that authors should agree with
2. **Strengths** - significance, relevance, quality, clarity, ethical implications; for datasets: accessibility, accountability, transparency
3. **Opportunities for Improvement** - limitations across same evaluative axes
4. **Limitations** - whether authors adequately addressed limitations (authors rewarded for transparency)
5. **Correctness** - accuracy of claims; for datasets: sound construction; for benchmarks: appropriate methods
6. **Clarity** - writing quality and presentation
7. **Relation to Prior Work** - clear differentiation from prior contributions
8. **Documentation** - for datasets: data collection, organization, availability, maintenance, ethical use; for benchmarks: reproducibility
9. **Ethical Concerns** - flagging for ethics review
10. **Overall Score** - numerical assessment (1-6)
11. **Confidence Score** - assessment certainty
12. **Code of Conduct Acknowledgement**

### D&B-Specific Evaluation Standards
The rubric uses a two-tier evaluation:

**Tier 1: Minimum Standard (Pass/Fail)**
- Pass if ALL aspects under minimum standard were discussed
- Fail if only partially discussed or not discussed at all
- Evaluations meant to be "generous"

**Tier 2: Standard of Excellence (Full/Partial/None)**
- Only assessed if minimum standard received Pass
- "Full" only if ALL criteria satisfied
- Significantly harder to attain
- High level of criticality advocated

**Important**: Failures should be based on documentation quality, NOT dataset quality itself.

### SAC Ranking Process (2025-specific)
- Papers scored below 4.25 average required SACs to provide detailed merit descriptions
- SACs produced relative rankings of papers within their stack
- Combined with qualitative justifications
- Dataset submissions tend to have higher average scores with tighter distributions than main track

---

## 5. Common Rejection Reasons & Pitfalls

### Desk Rejection Triggers
1. **Missing/invalid Croissant metadata file** (new in 2025)
2. **Inaccessible dataset** - data cannot be found/obtained without personal request to PI
3. **Undocumented/non-executable code**
4. **Dual submission** to both main track and D&B track
5. **Format/style violations** (wrong template, exceeding page limits)
6. **Misuse of gated access** (e.g., hiding data behind DTA unnecessarily)

### Common Reviewer-Identified Issues
1. **Missing or broken links** to datasets (~10% of reviewers encountered access issues)
2. **Very large datasets (>1TB)** causing platform rate limits and access difficulties
3. **Incomplete metadata**: license info (11.9% missing), descriptions (4.9%), URLs (3.5%)
4. **Insufficient documentation** of data collection, organization, and maintenance plans
5. **Lack of domain expertise match** between reviewers and niche domain papers
6. **Emphasis on methodological novelty over real-world impact** (noted as a reviewer bias concern)
7. **Limited reviewer engagement in rebuttals** (author concern)
8. **Reliance on AI-generated feedback** (author concern)

### Structural Weaknesses That Lead to Rejection
1. **Insufficient novelty** - paper doesn't clearly differentiate from prior datasets/benchmarks
2. **Poor documentation** - missing Datasheets, unclear collection methodology
3. **No maintenance plan** - no long-term preservation strategy
4. **Weak ethical considerations** - not addressing potential misuse, bias, privacy
5. **Unrealistic benchmarks** - evaluation setup too different from practical applications
6. **No clear ML community impact** - dataset/benchmark doesn't enable new research directions
7. **Missing licensing information** - no explicit license (CC or open-source recommended)
8. **No persistent identifier** (DOI recommended)

### Tips from Chair Blog Posts & Survey Data
- 82% of authors reported smooth hosting experiences -- use recommended platforms
- 77% of reviewers found datasets easily accessible -- ensure links work
- 69% of reviewers found automated metadata reports useful -- complete Croissant properly
- 58% of authors agreed new requirements led to fairer reviews
- Be transparent about limitations -- "authors should be rewarded rather than punished for being upfront"
- Answering "no" to checklist questions is NOT grounds for rejection
- Use established hosting platforms (HuggingFace, Kaggle, Dataverse, OpenML) -- 80%+ of accepted papers did

---

## 6. Recent Accepted Papers Similar to NegBioDB

### WelQrate (NeurIPS 2024 D&B Track - Poster)
**Title**: "WelQrate: Defining the Gold Standard in Small Molecule Drug Discovery Benchmarking"
**URL**: https://neurips.cc/virtual/2024/poster/97684
**Paper**: https://proceedings.neurips.cc/paper_files/paper/2024/file/5f2f8305cd1c5be7e8319aea306388ce-Paper-Datasets_and_Benchmarks_Track.pdf
**Website**: www.WelQrate.org

**Why it succeeded:**
1. **Expert-driven curation pipeline**: Designed by drug discovery experts, not purely computational
2. **Hierarchical quality control**: Goes beyond primary HTS using confirmatory screens + counter screens
3. **PAINS filtering**: Three layers (promiscuity, optical interference, other interference patterns)
4. **9 datasets across 5 therapeutic target classes** -- substantial scope
5. **Addresses real problem**: Existing benchmarks use poorly curated data with unreliable labels
6. **Standardized evaluation framework**: Covers datasets, featurization, 3D conformations, metrics, splits
7. **Publicly available**: All code, data, and curation scripts at WelQrate.org
8. **Clear research questions**: Systematically evaluates effects of models, data quality, featurization, splits

**Key lesson for NegBioDB**: Expert-driven curation + clear quality standards + identifying a gap in existing benchmarks. WelQrate argued existing MoleculeNet benchmarks are unreliable, providing a strong motivation.

---

### TDC - Therapeutics Data Commons (NeurIPS 2021 D&B Track)
**Title**: "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
**URL**: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/4c56ff4ce4aaf9573aa5dff913df997a-Abstract-round1.html
**OpenReview**: https://openreview.net/forum?id=8nvgnORnoWr
**Website**: https://tdcommons.ai

**Why it succeeded:**
1. **Massive scope**: 66 AI-ready datasets across 22 learning tasks
2. **Full pipeline coverage**: Target identification -> hit discovery -> lead optimization -> manufacturing
3. **Multiple modalities**: Small molecules, biologics, gene editing
4. **Rich ecosystem**: 33 data functions, 23 evaluation strategies, 17 molecule generation oracles, 29 leaderboards
5. **Open Python library**: Easy pip-installable access
6. **Identified unsolved challenges**: Showed strongest algorithms fail at distributional shifts, multi-modal learning
7. **Community infrastructure**: Leaderboards, standardized splits, evaluation protocols
8. **Broad applicability**: Not narrow to one target or assay

**Key lesson for NegBioDB**: Build an ecosystem, not just a dataset. Provide tools, leaderboards, standardized access via Python library.

---

### Lo-Hi (NeurIPS 2023 D&B Track - Poster)
**Title**: "Lo-Hi: Practical ML Drug Discovery Benchmark"
**URL**: https://proceedings.neurips.cc/paper_files/paper/2023/hash/cb82f1f97ad0ca1d92df852a44a3bd73-Abstract-Datasets_and_Benchmarks.html
**OpenReview**: https://openreview.net/forum?id=H2Yb28qGLV
**GitHub**: https://github.com/SteshinSS/lohi_neurips2023

**Why it succeeded:**
1. **Practical relevance**: Tasks mirror actual drug discovery workflow (Lead Optimization + Hit Identification)
2. **Novel splitting algorithm**: Balanced Vertex Minimum k-Cut for molecular splitting
3. **Exposed overoptimism**: Showed existing benchmarks are "unrealistic and too different from practice"
4. **Controlled similarity**: Train/test sets have ECFP4 Tanimoto similarity < 0.4
5. **Multiple datasets**: 4 Hi datasets (DRD2, HIV, KDR, Sol) + 3 Lo datasets (DRD2, KCNH2, KDR)
6. **Clear methodology**: Standardized hyperparameter tuning protocol
7. **Single-author paper**: Shows even solo researchers can succeed with a strong contribution

**Key lesson for NegBioDB**: Demonstrate that current benchmarks are flawed/overoptimistic and provide a more realistic alternative.

---

### Other Relevant Accepted Papers (NeurIPS 2024-2025)

#### MassSpecGym (NeurIPS 2024 D&B)
- Benchmark for discovery and identification of molecules via mass spectrometry
- Applications in drug development

#### OP3 - Open Problems Perturbation Prediction (NeurIPS 2024 D&B)
- Benchmark for predicting transcriptomic responses to chemical perturbations
- 146 compounds tested on human blood cells
- Framework for drug discovery ML predictions

#### OligoGym (NeurIPS 2025 D&B)
- Curated datasets and benchmarks for oligonucleotide drug discovery
- Standardized, ML-ready datasets for various oligonucleotide modalities
- Spotlight talk at AI4D3 workshop

#### Unraveling Molecular Structure (NeurIPS 2024 D&B)
- Multimodal spectroscopic dataset for chemistry

#### FGBench (NeurIPS 2025 D&B)
- Regression/classification tasks on 245 functional groups for molecular property reasoning
- Shows current LLMs struggle with fine-grained chemical property reasoning

#### PharmaBench (Published 2024, related)
- ADMET benchmarks enhanced with LLMs
- Multi-agent data mining across 14,401 bioassays

---

### Papers About Negative/Inactive Data
No papers specifically focused on "negative results databases" or "inactive compound collections" were found in the NeurIPS 2024-2025 D&B track. This represents a **clear gap** and potential opportunity for NegBioDB, as:
- WelQrate addresses data quality but focuses on curating positives + confirmed negatives from counter screens
- TDC aggregates existing datasets without specifically addressing negative data quality
- Lo-Hi uses binary classification but doesn't focus on the negative data problem per se
- The problem of unreliable negative labels (assumed inactives) is widely acknowledged but not directly addressed as a dataset contribution

---

## 7. NeurIPS 2026 Status

As of March 2026, the official NeurIPS 2026 Call for Papers for the Datasets & Benchmarks Track has **not yet been published**. Key known information:

- **Conference dates**: December 6-12, 2026
- **Location**: Sydney, Australia
- The D&B track is expected to continue given its growth trajectory
- Requirements will likely build on 2025 changes (mandatory Croissant, persistent hosting, main track alignment)
- Expected submission deadline: likely May 2026 (based on historical pattern)

Check https://neurips.cc for the official 2026 announcement.

---

## 8. Strategic Recommendations for NegBioDB Submission

Based on this research, key factors for a successful NegBioDB submission:

1. **Fill the gap**: No existing D&B paper specifically addresses negative/inactive data quality -- position this as a novel contribution
2. **Expert curation**: Follow WelQrate's model of expert-driven, domain-justified curation pipelines
3. **Practical impact**: Follow Lo-Hi's model of showing current benchmarks are overoptimistic due to unreliable negative labels
4. **Ecosystem tools**: Follow TDC's model of providing Python library, standardized access, leaderboards
5. **Mandatory compliance**: Croissant metadata, persistent hosting (use HuggingFace/Kaggle/Dataverse), executable code
6. **Documentation**: Include comprehensive Datasheet for Datasets, maintenance plan, licensing (CC-BY 4.0 recommended)
7. **Clear research questions**: Systematically evaluate how negative data quality affects model performance
8. **Realistic evaluation**: Include practical molecular splits (scaffold, temporal, etc.)
9. **Transparent limitations**: Be upfront about what the dataset does and doesn't cover

---

## Sources

- NeurIPS 2025 D&B Call for Papers: https://neurips.cc/Conferences/2025/CallForDatasetsBenchmarks
- NeurIPS 2025 D&B FAQ: https://neurips.cc/Conferences/2025/DatasetsBenchmarks-FAQ
- NeurIPS 2024 D&B Call for Papers: https://neurips.cc/Conferences/2024/CallForDatasetsBenchmarks
- 2025 Review Process Blog: https://blog.neurips.cc/2025/09/30/reflecting-on-the-2025-review-process-from-the-datasets-and-benchmarks-chairs/
- D&B Track Overview Blog: https://blog.neurips.cc/2025/12/05/neurips-datasets-benchmarks-track-from-art-to-science-in-ai-evaluations/
- Raising the Bar Blog: https://blog.neurips.cc/2025/03/10/neurips-datasets-benchmarks-raising-the-bar-for-dataset-submissions/
- NeurIPS 2023 Review Guidelines: https://neurips.cc/Conferences/2023/DatasetsAndBenchmarks/ReviewGuidelines
- NeurIPS 2025 Reviewer Guidelines: https://neurips.cc/Conferences/2025/ReviewerGuidelines
- NeurIPS 2024 D&B Statistics: https://papercopilot.com/statistics/neurips-statistics/neurips-2024-statistics-datasets-benchmarks-track/
- NeurIPS 2025 D&B Statistics: https://papercopilot.com/statistics/neurips-statistics/neurips-2025-statistics-datasets-benchmarks-track/
- NeurIPS 6-point Scoring: https://forum.cspaper.org/topic/83/neurips-2025-adopts-a-new-6-point-review-scoring-system-simplified-scoring-for-enhanced-consistency
- WelQrate Paper: https://arxiv.org/abs/2411.09820
- TDC Paper: https://arxiv.org/abs/2102.09548
- Lo-Hi Paper: https://openreview.net/forum?id=H2Yb28qGLV
- OligoGym: https://openreview.net/forum?id=EtQ2YAWHYs
- Croissant Checker: https://huggingface.co/spaces/JoaquinVanschoren/croissant-checker
