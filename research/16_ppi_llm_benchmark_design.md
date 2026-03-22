# 16. PPI Domain LLM Benchmark Design

## 1. Overview

The NegBioDB-PPI LLM benchmark evaluates large language models on protein-protein non-interaction understanding tasks across four difficulty levels. It mirrors the DTI and CT domains' L1-L4 structure but adapts to the unique characteristics of PPI data: sequence-level protein information, diverse evidence sources (curated experiments, systematic screens, ML inference, database absence), and the fundamental challenge of reasoning about why two proteins do not physically interact.

### Database State (as of 2026-03-21)
- **2,220,786 negative pairs** aggregated from 2.23M negative results across 18,412 proteins
- **849 MB** database (`data/negbiodb_ppi.db`)
- **Tier distribution:**

| Source | Tier | Count | Evidence Type |
|--------|------|-------|---------------|
| IntAct | Gold | 69 | Curated experimental non-interactions (co-IP, pulldown) |
| IntAct | Silver | 710 | Curated experimental non-interactions (two-hybrid) |
| HuRI | Gold | 500,000 | Systematic Y2H screen negatives (from 39.9M candidates) |
| hu.MAP | Silver | 1,228,891 | ML-derived negatives from ComplexPortal |
| STRING | Bronze | 500,000 | Zero-score pairs between well-studied proteins |

- **Protein annotations:** gene_symbol, subcellular_location, amino_acid_sequence (99.6% coverage, avg 570.7 AA), function_description, go_terms, domain_annotations (migration 002)
- **Publication data:** 65 unique PMIDs from IntAct experiments, ppi_publication_abstracts table
- **Positive pairs:** 61,728 HuRI interactions (578 conflicts removed from both sides)
- **ML benchmark:** Phase C complete (18/18 seed 42, seeds 43/44 pending)

### Design Principles

1. **Cross-domain consistency:** L1-L4 structure mirrors DTI and CT domains for direct comparison
2. **Evidence quality focus:** L1 classifies evidence quality (not interaction prediction), following DTI-L1/CT-L1 pattern
3. **Source-blind context:** Evidence descriptions never name the source database to prevent trivial answers
4. **Biological reasoning:** L3 tests mechanistic understanding of why proteins don't interact
5. **Contamination awareness:** L4 uses temporal splits with IntAct dates and HuRI's 2020 publication

---

## 2. Level Definitions

| Level | Task | Size | Input | Output | Eval | Difficulty |
|-------|------|------|-------|--------|------|------------|
| PPI-L1 | Evidence Quality Classification (4-way MCQ) | 1,200 | Protein info + evidence description | A/B/C/D letter | Accuracy, macro_F1, MCC | Low |
| PPI-L2 | Non-Interaction Evidence Extraction | 500 | Constructed evidence summaries | Structured JSON | Schema compliance, entity_F1 | Medium |
| PPI-L3 | Non-Interaction Reasoning | 200 | Full protein annotations | Free text explanation | LLM-as-Judge (4 dim) | High |
| PPI-L4 | Interaction Testing Discrimination | 500 | Minimal protein identifiers | "tested" / "untested" | Accuracy, F1, MCC | Highest |

### Evidence Quality Mapping

**PPI-L1 uses 4-way classification** (one category per major evidence source):

| MCQ Option | Evidence Quality | Source(s) | Records Available |
|------------|-----------------|-----------|-------------------|
| A) Direct experimental | Curated lab non-interaction with detection method | IntAct gold/silver | 779 |
| B) Systematic screen | High-throughput binary screen negative | HuRI gold | 500,000 |
| C) Computational inference | ML model predicts non-membership in complexes | hu.MAP silver | 1,228,891 |
| D) Database score absence | Well-studied proteins with zero interaction score | STRING bronze | 500,000 |

**Rationale for 4-way over finer granularity:**
- IntAct gold (69) vs silver (710) distinction is too small for a separate class at 300/class
- 4-way matches DTI L1's complexity and keeps cognitive load manageable
- The 4 categories represent genuinely different evidence quality levels: wet-lab curation > systematic screen > ML prediction > database absence

---

## 3. PPI-L1: Evidence Quality Classification (MCQ)

### Task Definition

Given information about two proteins and a description of the evidence for their non-interaction, classify the quality tier of the evidence into one of 4 categories.

### Data Source Selection

| Category | Tier(s) | Available Records | Context Richness |
|----------|---------|-------------------|------------------|
| A) Direct experimental | IntAct gold + silver | 779 | Detection method, publication, curator notes |
| B) Systematic screen | HuRI gold | 500,000 | Screen methodology, scale, controls |
| C) Computational inference | hu.MAP silver | 1,228,891 | Complex membership, ML confidence |
| D) Database score absence | STRING bronze | 500,000 | Protein study depth, score components |

**Category A pool constraint:** Only 779 IntAct records exist. After allocating 60 fewshot + 60 val + 180 test = 300, this leaves 479 spare records. Category A is the binding constraint on dataset size. All other categories sample 300 from much larger pools.

### Difficulty Stratification

| Difficulty | Criteria | % of Dataset |
|------------|----------|--------------|
| Easy (40%) | Clear evidence markers -- e.g., explicit detection method for A, "0 of N tested" for B | 40% |
| Medium (35%) | Partial evidence -- e.g., ML confidence score near threshold for C, moderate protein study depth for D | 35% |
| Hard (25%) | Ambiguous indicators -- e.g., rare detection method for A, borderline zero-score for D | 25% |

**Difficulty operationalization:**
- **Easy:** Context text contains explicit quality indicators (named assay method, large-scale screen statistics, high ML confidence, zero across all STRING evidence channels)
- **Medium:** Context provides some but not all quality signals (publication exists but method is generic, ML confidence moderate, some STRING channels non-zero but combined score is zero)
- **Hard:** Context is deliberately sparse or uses ambiguous language (uncommon detection method names, minimal screen context, borderline computational scores)

### Dataset Construction

**Total: 1,200 records** (300 per class x 4 classes)

| Split | Size | Per Class | Purpose |
|-------|------|-----------|---------|
| Fewshot pool | 240 | 60 | 3-shot example selection (3 per class = 12 total) |
| Validation | 240 | 60 | Hyperparameter tuning |
| Test | 720 | 180 | Final evaluation |

**Sampling strategy per class:**
1. **Category A:** Use all 779 IntAct records; sample 300 with stratification by detection method type (co-IP, pulldown, two-hybrid, etc.)
2. **Category B:** Sample 300 from HuRI 500K; stratify by protein functional diversity (ensure coverage across GO biological process categories)
3. **Category C:** Sample 300 from hu.MAP 1.23M; stratify by ML confidence score (tertiles)
4. **Category D:** Sample 300 from STRING 500K; stratify by combined_score components and protein study depth
5. **Deduplication:** MAX_PER_PROTEIN = 15 (prevent single hub protein from dominating)
6. **Organism filter:** Human proteins only (all sources are human-centric)

### Context Text Generation

Context text is source-blind: it describes the evidence without naming the source database.

**Category A (Direct experimental):**
```
Protein A: {gene_symbol_1} ({uniprot_accession_1})
  Function: {function_description_1}
  Location: {subcellular_location_1}

Protein B: {gene_symbol_2} ({uniprot_accession_2})
  Function: {function_description_2}
  Location: {subcellular_location_2}

Evidence: A non-interaction was reported between these proteins using
{detection_method} methodology. The result was curated from a peer-reviewed
publication (year: {publication_year}). The experiment specifically tested
for physical binding and found no detectable interaction.
```

**Category B (Systematic screen):**
```
Protein A: {gene_symbol_1} ({uniprot_accession_1})
  Function: {function_description_1}
  Location: {subcellular_location_1}

Protein B: {gene_symbol_2} ({uniprot_accession_2})
  Function: {function_description_2}
  Location: {subcellular_location_2}

Evidence: These proteins were tested as part of a high-throughput binary
interaction screen covering approximately {screen_size} protein pairs.
Both proteins were successfully expressed and tested in the assay system,
but no interaction signal was detected above the scoring threshold.
```

**Category C (Computational inference):**
```
Protein A: {gene_symbol_1} ({uniprot_accession_1})
  Function: {function_description_1}
  Location: {subcellular_location_1}

Protein B: {gene_symbol_2} ({uniprot_accession_2})
  Function: {function_description_2}
  Location: {subcellular_location_2}

Evidence: A machine learning model trained on protein complex membership
data predicts that these two proteins do not co-occur in the same
macromolecular complex. The prediction confidence is {confidence_score}.
Neither protein was found in any known complex containing the other.
```

**Category D (Database score absence):**
```
Protein A: {gene_symbol_1} ({uniprot_accession_1})
  Function: {function_description_1}
  Location: {subcellular_location_1}

Protein B: {gene_symbol_2} ({uniprot_accession_2})
  Function: {function_description_2}
  Location: {subcellular_location_2}

Evidence: Both proteins are well-characterized in the literature
(protein A: {num_publications_1} publications, protein B:
{num_publications_2} publications). Despite extensive study of both
proteins, no evidence of interaction has been reported across multiple
evidence channels (text mining, co-expression, experimental, database).
The combined interaction score is zero.
```

### Prompt Design

```python
# Uses PPI_SYSTEM_PROMPT (defined in section 8)

PPI_L1_QUESTION = (
    "Based on the protein pair and evidence description below, classify the "
    "quality of evidence for this non-interaction.\n\n"
    "{context_text}\n\n"
    "Categories:\n"
    "A) Direct experimental — Non-interaction confirmed by targeted laboratory assay\n"
    "B) Systematic screen — Non-interaction from high-throughput binary screening\n"
    "C) Computational inference — Non-interaction predicted by machine learning model\n"
    "D) Database score absence — No interaction evidence despite extensive study of both proteins\n"
)

PPI_L1_ANSWER_FORMAT = "Respond with ONLY a single letter (A, B, C, or D)."
```

### Evaluation Metrics

| Metric | Primary? | Description |
|--------|----------|-------------|
| Accuracy | Yes | Overall correct classification rate |
| Macro F1 | Yes | Unweighted average F1 across 4 classes |
| MCC | Yes | Matthews correlation coefficient (multiclass) |
| Weighted F1 | Supplementary | Weighted average F1 |
| Per-class accuracy | Supplementary | Breakdown by A/B/C/D |
| Parse rate | Supplementary | % of responses successfully parsed as A-D |
| Difficulty breakdown | Supplementary | Accuracy by easy/medium/hard |

---

## 4. PPI-L2: Non-Interaction Evidence Extraction

### Task Definition

Given a constructed evidence summary about a set of protein non-interactions, extract structured information about the non-interacting pairs and their evidence.

### Design Rationale: Fallback from PubMed Abstracts

The original L2 design (following DTI-L2 pattern) planned to use PubMed abstracts as input for extraction. However, PPI non-interaction data comes from fundamentally different publication patterns than DTI:

- **IntAct:** Only 65 unique PMIDs across 779 records. Many PMIDs report dozens of non-interactions in supplementary tables, not in abstract text.
- **HuRI:** Single publication (PMID 32296183, Luck et al. 2020). The abstract does not enumerate individual non-interacting pairs.
- **hu.MAP and STRING:** No publication-level records for individual pair non-interactions.

**Consequence:** There are insufficient distinct, abstract-extractable non-interaction reports to build a 500-record extraction benchmark from real PubMed abstracts.

**Fallback design:** Construct evidence summaries from database fields that simulate the information density of a results paragraph. These summaries are rendered at dataset build time and stored as `context_text` in the JSONL records.

### Data Source

All tiers are usable because the extraction source is constructed text, not raw abstracts.

| Tier | Records Used | Context Richness |
|------|-------------|------------------|
| IntAct gold/silver | ~200 | Detection method, PMID, curator annotations |
| HuRI gold | ~150 | Screen-level statistics, protein pair details |
| hu.MAP silver | ~100 | ML confidence, complex membership data |
| STRING bronze | ~50 | Score breakdown, publication counts |

### Target Schema

```json
{
  "non_interacting_pairs": [
    {
      "protein_a": "BRCA1",
      "protein_b": "TP53",
      "uniprot_a": "P38398",
      "uniprot_b": "P04637",
      "evidence_type": "experimental_non_interaction",
      "detection_method": "co-immunoprecipitation",
      "confidence": "high"
    }
  ],
  "total_negative_count": 5,
  "positive_interactions_mentioned": true,
  "evidence_source_type": "targeted_experiment"
}
```

### Context Text Generation

Each record contains a constructed evidence summary describing 2-5 non-interacting pairs in a paragraph format:

**Example (IntAct-derived):**
```
A study investigating protein complex assembly in the DNA damage response
tested physical interactions between several nuclear proteins using
co-immunoprecipitation. BRCA1 (P38398) was found to interact with BARD1
and RAD51, but showed no detectable binding with TP53 (P04637) or CHEK2
(O96017) under the same conditions. Similarly, PALB2 (Q86YC2) did not
co-precipitate with MDC1 (Q14676). In total, 3 non-interactions were
reported alongside 2 positive interactions. All experiments were performed
in HEK293T cell lysates with endogenous expression levels.
```

**Example (HuRI-derived):**
```
As part of a systematic binary interaction mapping effort, pairwise yeast
two-hybrid (Y2H) tests were conducted on a panel of human proteins involved
in signal transduction. Among the tested pairs, EGFR (P00533) and INSR
(P06213) showed no interaction signal, nor did KRAS (P01116) and HRAS
(P01112). Both pairs were tested with both bait-prey orientations.
The screen identified 2 non-interactions out of the 8 pairs tested in
this subset, with the remaining 6 showing detectable interaction signals.
```

### Difficulty Stratification

| Difficulty | Criteria | Example | % |
|------------|----------|---------|---|
| Easy (40%) | Explicit pair listing, named proteins, clear evidence type | "BRCA1 did not bind TP53" | 40% |
| Medium (35%) | Multiple pairs, mixed positive/negative, implicit evidence | "3 of 8 pairs showed no signal" | 35% |
| Hard (25%) | Aggregate statistics, ambiguous language, nested references | "The remaining pairs lacked evidence" | 25% |

### Dataset Construction

**Total: 500 records**

| Split | Size | Purpose |
|-------|------|---------|
| Fewshot pool | 50 | 3-shot example selection |
| Validation | 50 | Schema validation tuning |
| Test | 400 | Final evaluation |

**Sampling strategy:**
1. Distribute across evidence types: ~200 IntAct, ~150 HuRI, ~100 hu.MAP, ~50 STRING
2. Each record contains 2-5 non-interacting pairs (total ~1,500-2,500 extractable pairs)
3. ~60% of records also mention positive interactions (distractor complexity)
4. Stratify by difficulty within each evidence type
5. Ensure protein diversity: MAX_PER_PROTEIN = 20 across the full dataset
6. Template deduplication: no two records share the same pair set

### Gold Standard

The extraction gold standard is derived directly from database fields:
1. **non_interacting_pairs** -- from protein_protein_pairs + proteins table JOINs
2. **evidence_type** -- from ppi_negative_results.evidence_type
3. **detection_method** -- from ppi_negative_results.detection_method (IntAct only)
4. **total_negative_count** -- counted during context construction
5. **positive_interactions_mentioned** -- set during context construction

No manual annotation is needed because the constructed text is generated from known ground truth fields.

### Prompt Design

```python
PPI_L2_QUESTION = (
    "Extract structured information about protein non-interactions from the "
    "following evidence summary. Return a JSON object with the fields "
    "specified below.\n\n"
    "{context_text}\n\n"
    "Required JSON fields:\n"
    "- non_interacting_pairs: array of objects, each with:\n"
    "    - protein_a: gene symbol or protein name\n"
    "    - protein_b: gene symbol or protein name\n"
    "    - uniprot_a: UniProt accession if mentioned (null otherwise)\n"
    "    - uniprot_b: UniProt accession if mentioned (null otherwise)\n"
    "    - evidence_type: one of [experimental_non_interaction, "
    "systematic_screen, ml_predicted, database_absence]\n"
    "    - detection_method: specific method if mentioned (null otherwise)\n"
    "    - confidence: high, medium, or low\n"
    "- total_negative_count: integer count of non-interacting pairs\n"
    "- positive_interactions_mentioned: true or false\n"
    "- evidence_source_type: one of [targeted_experiment, systematic_screen, "
    "computational, database_mining]\n\n"
    "Return ONLY valid JSON, no additional text."
)
```

### Evaluation Metrics

| Metric | Primary? | Description |
|--------|----------|-------------|
| Schema compliance | Yes | % of responses that parse as valid JSON with required fields |
| Entity F1 | Yes | F1 for correctly extracted (protein_a, protein_b) pairs (order-independent) |
| Field accuracy | Yes | Exact match on evidence_type, detection_method, confidence per pair |
| Total count accuracy | Supplementary | |predicted_count - gold_count| / gold_count |
| UniProt accuracy | Supplementary | Exact match on uniprot_a, uniprot_b when present |
| Parse rate | Supplementary | % of responses parseable as JSON |

---

## 5. PPI-L3: Non-Interaction Reasoning

### Task Definition

Given two proteins with rich annotation data (function, subcellular location, domains, GO terms), provide a scientific explanation for why these two proteins do not physically interact.

### Data Source

**Gold tier only** -- requires high-confidence non-interaction evidence for meaningful reasoning. Both IntAct (curated) and HuRI (systematic screen) gold-tier records are eligible.

**Source composition:**
- IntAct gold (69 records): Experimentally validated with specific detection methods
- IntAct silver (710 records): Experimental but lower-confidence methods (two-hybrid)
- HuRI gold (500,000 records): Systematic Y2H screen negatives

**For L3, we select from gold + silver tiers** with preference for records where both proteins have rich annotations:

**Annotation requirements (both proteins must have):**
1. `function_description IS NOT NULL` (biological function context)
2. `subcellular_location IS NOT NULL` (compartmentalization reasoning)
3. `domain_annotations IS NOT NULL` (structural reasoning)
4. `go_terms IS NOT NULL` (functional category information)

**Compartment balance:** Target ~50% same-compartment pairs and ~50% different-compartment pairs. Same-compartment cases are harder (must reason beyond co-localization). Different-compartment cases test basic biological knowledge about subcellular separation.

### Dataset Construction

**Total: 200 records**

| Split | Size | Purpose |
|-------|------|---------|
| Fewshot pool | 20 | 3-shot example selection |
| Validation | 20 | Judge calibration |
| Test | 160 | Final evaluation |

**Diversity requirements:**
- Subcellular compartment balance: ~50% same-compartment, ~50% different-compartment
- Functional diversity: no more than 20% from any single GO biological process category
- Source balance: ~60 IntAct (gold+silver) + ~140 HuRI (gold)
- Protein diversity: MAX_PER_PROTEIN = 5 across the dataset
- Sequence length diversity: include short (<300 AA), medium (300-600 AA), and long (>600 AA) proteins

### Context Text Generation

```
Protein A: {gene_symbol_1} ({uniprot_accession_1})
  Full name: {protein_name_1}
  Function: {function_description_1}
  Subcellular location: {subcellular_location_1}
  Domains: {domain_annotations_1}
  GO terms: {go_terms_1}
  Sequence length: {sequence_length_1} amino acids

Protein B: {gene_symbol_2} ({uniprot_accession_2})
  Full name: {protein_name_2}
  Function: {function_description_2}
  Subcellular location: {subcellular_location_2}
  Domains: {domain_annotations_2}
  GO terms: {go_terms_2}
  Sequence length: {sequence_length_2} amino acids

Organism: Homo sapiens

These two proteins have been experimentally confirmed to NOT physically
interact. Provide a scientific explanation for this non-interaction.
```

### Prompt Design

```python
PPI_L3_QUESTION = (
    "The two proteins described below have been experimentally confirmed to "
    "NOT physically interact. Based on the protein information provided, "
    "explain the biological and structural reasons for this non-interaction.\n\n"
    "{context_text}\n\n"
    "Your explanation should address:\n"
    "1. Biological plausibility -- Are these proteins involved in the same "
    "or different biological pathways? Would an interaction be expected?\n"
    "2. Structural reasoning -- Do the domain architectures suggest "
    "compatible binding interfaces?\n"
    "3. Mechanistic factors -- Consider subcellular localization, "
    "expression timing, post-translational modifications, or complex "
    "membership that would prevent interaction.\n"
    "4. Specificity -- What specific features of THESE two proteins "
    "(not generic statements) explain the non-interaction?\n\n"
    "Provide a thorough explanation in 3-5 paragraphs."
)
```

### Evaluation: LLM-as-Judge

**4 dimensions, each scored 1-5:**

| Dimension | Description | Scoring Guide |
|-----------|-------------|---------------|
| biological_plausibility | Does the explanation correctly assess pathway relevance? | 5 = accurate pathway analysis, 1 = wrong pathway claims |
| structural_reasoning | Does it address domain/binding interface compatibility? | 5 = specific domain analysis, 1 = no structural discussion |
| mechanistic_completeness | Coverage of localization, timing, modifications, complex membership | 5 = all relevant factors addressed, 1 = only 1 factor |
| specificity | Analysis specific to this protein pair vs generic statements | 5 = pair-specific throughout, 1 = entirely generic |

**Judge model:** GPT-4o-mini

**Rationale for GPT-4o-mini over Gemini Flash:** PPI reasoning requires protein biology expertise. GPT-4o-mini has demonstrated stronger performance on biological reasoning in the DTI L3 judge evaluation, and its cost is manageable for 200 records x 5 models = 1,000 judge calls.

**Judge prompt:**
```python
PPI_L3_JUDGE_PROMPT = (
    "You are evaluating a scientific explanation for why two proteins do "
    "not physically interact.\n\n"
    "PROTEIN PAIR CONTEXT:\n{context_text}\n\n"
    "EVIDENCE TIER: {confidence_tier}\n"
    "COMPARTMENT RELATIONSHIP: {compartment_relationship}\n\n"
    "RESPONSE TO EVALUATE:\n{response_text}\n\n"
    "Score the response on 4 dimensions (1-5 each):\n"
    "1. biological_plausibility: Does the explanation correctly assess "
    "whether these proteins participate in related or unrelated pathways?\n"
    "2. structural_reasoning: Does it discuss domain architectures, binding "
    "interfaces, or structural compatibility?\n"
    "3. mechanistic_completeness: Does it consider localization, expression "
    "patterns, modifications, and complex membership?\n"
    "4. specificity: Is the analysis specific to this protein pair, or could "
    "it apply to any two random proteins?\n\n"
    "Return ONLY a JSON object: "
    "{\"biological_plausibility\": N, \"structural_reasoning\": N, "
    "\"mechanistic_completeness\": N, \"specificity\": N}"
)
```

### Judge Validation Protocol

1. Human-score 30 responses (6 per model) as calibration set
2. Compute Pearson correlation between judge scores and human scores per dimension
3. Target: r > 0.6 per dimension (substantial agreement)
4. If any dimension r < 0.4, revise the judge prompt with additional rubric detail
5. Report inter-judge agreement (3 judge runs with temperature=0; measure coefficient of variation)

---

## 6. PPI-L4: Interaction Testing Discrimination

### Task Definition

Given two protein identifiers with minimal context, determine whether this protein pair has been experimentally tested for interaction (and found non-interacting) versus simply never tested.

### Design Rationale

PPI databases have extreme sparsity: with ~20,000 human proteins, there are ~200M possible pairs, but only ~2.2M are in NegBioDB and ~62K are confirmed interactors. The vast majority of pairs are untested. This task evaluates whether LLMs can distinguish experimentally validated non-interactions from the untested background, testing knowledge of the PPI experimental landscape.

**Comparison with DTI-L4 and CT-L4:**
- DTI-L4 showed MCC <= 0.18 (near random) -- LLMs cannot distinguish tested from untested compound-target pairs
- CT-L4 showed meaningful performance (MCC 0.60-0.76) -- clinical trial registry data is well-represented in training corpora
- PPI-L4 hypothesis: performance between DTI and CT, since major PPI studies (HuRI, BioPlex) are well-known but individual pair-level data is less accessible than trial registries

### Data Source

**Tested pairs (250):**

| Subset | Count | Source | Temporal Window |
|--------|-------|--------|-----------------|
| Pre-2015 tested | 125 | IntAct records with publication_year <= 2015 | High contamination risk |
| Post-2020 tested | 125 | HuRI (2020 publication) + IntAct 2020+ | Lower contamination risk |

**Temporal split rationale:**
- IntAct records span 2000-2024, with ~50 unique PMIDs pre-2015 providing sufficient coverage for 125 pairs
- HuRI was published in May 2020 (PMID 32296183). Its systematic screen data may or may not be in LLM training corpora depending on cutoff dates
- The pre-2015/post-2020 split creates a 5-year gap that cleanly separates likely-memorized from possibly-unseen data

**Selection criteria for tested pairs:**
- Both proteins must have gene_symbol (recognizable name)
- Both proteins must be well-studied (>10 publications in UniProt)
- Prefer pairs where both proteins are individually well-known but their non-interaction is the informative signal

**Untested pairs (250):**

| Subset | Count | Construction Method |
|--------|-------|---------------------|
| Trick pairs | 125 | Same pathway/compartment, plausible but untested |
| Obvious pairs | 125 | Different compartments/organisms, clearly unrelated |

**Trick pair construction:**
- Select two proteins from the SAME GO biological process or SAME complex family
- Verify the pair does NOT appear in: (a) protein_protein_pairs, (b) HuRI positive interactions, (c) STRING with score > 0, (d) BioGRID (spot-check)
- Example: "CDK2 and CDK7" -- both are cyclin-dependent kinases but CDK2-CDK7 direct binding is not established as a simple binary interaction (CDK7 acts through CAK complex)

**Obvious pair construction:**
- Select proteins from clearly different subcellular compartments (e.g., mitochondrial matrix protein + extracellular matrix protein)
- Or proteins from different functional categories with no known pathway connection
- Example: "Insulin (P01308) and Histone H3 (P68431)" -- secreted hormone vs nuclear chromatin protein

### Verification Protocol

1. All "untested" pairs verified against:
   - `protein_protein_pairs` table (no match)
   - HuRI positive interactions file (no match)
   - STRING combined_score > 0 check (must be zero or absent)
   - BioGRID API spot-check (20% sample)
   - PubMed search for `"{gene_A}" AND "{gene_B}" AND ("interact" OR "binding" OR "complex")` (10% sample)
   - Manual expert review of all 250 untested pairs
2. All "tested" pairs verified:
   - Exist in `protein_protein_pairs` with best_confidence IN ('gold', 'silver')
   - Both protein gene_symbols resolve correctly in UniProt
   - The specific experiment/screen is identifiable

### Dataset Construction

**Total: 500 records**

| Split | Size | Per Class | Purpose |
|-------|------|-----------|---------|
| Fewshot pool | 50 | 25 tested + 25 untested | 3-shot example selection |
| Validation | 50 | 25 tested + 25 untested | Threshold tuning |
| Test | 400 | 200 tested + 200 untested | Final evaluation |

### Context Text Generation

```
Protein A: {gene_symbol_1} ({uniprot_accession_1})
Organism: Homo sapiens

Protein B: {gene_symbol_2} ({uniprot_accession_2})
Organism: Homo sapiens

Question: Has this protein pair been experimentally tested for
physical interaction, and found to NOT interact?
```

**Note:** No function descriptions, no subcellular locations, no domains -- the model must rely on its knowledge of the PPI experimental landscape. This parallels DTI-L4 (no SMILES, no assay data) and CT-L4 (no phase, no outcome data).

### Prompt Design

```python
PPI_L4_QUESTION = (
    "Based on your knowledge of protein-protein interaction studies, "
    "determine whether the following protein pair has been experimentally "
    "tested for physical interaction and confirmed as non-interacting.\n\n"
    "{context_text}\n\n"
    "On the first line, respond with ONLY 'tested' or 'untested'.\n"
    "On the second line, provide brief evidence for your answer (e.g., "
    "study name, screening method, or reasoning for why it was/wasn't "
    "tested)."
)
```

### Anti-Contamination Analysis

PPI data has heterogeneous contamination risk:
- **IntAct:** Individual PMIDs are indexed in PubMed; specific non-interaction data may be in abstracts or supplementary tables
- **HuRI:** Single landmark paper (2020), widely cited (~1,500 citations). The paper itself is likely in training data, but individual pair-level results require reading supplementary tables
- **hu.MAP / STRING:** Computational predictions, not individually published

| Metric | Description |
|--------|-------------|
| accuracy_pre_2015 | Accuracy on IntAct pairs from publications before 2015 |
| accuracy_post_2020 | Accuracy on HuRI/IntAct pairs from 2020+ |
| accuracy_gap | pre_2015 - post_2020 |
| contamination_flag | accuracy_gap > 15% suggests memorization |
| trick_pair_accuracy | Accuracy specifically on trick (plausible-but-untested) pairs |
| obvious_pair_accuracy | Accuracy on clearly-untested pairs (baseline) |

**Additional contamination signals:**
- PMID citation: model cites the correct publication for tested pairs -> memorization
- Screen name citation: model mentions "HuRI" or "Human Reference Interactome" -> training data leakage
- Evidence specificity: model describes Y2H protocol details not in prompt -> memorization

### Evaluation Metrics

| Metric | Primary? | Description |
|--------|----------|-------------|
| Accuracy | Yes | Overall correct classification rate |
| F1 (tested) | Yes | Binary F1 with pos_label="tested" |
| MCC | Yes | Matthews correlation coefficient |
| Parse rate | Supplementary | % of responses parseable as tested/untested |
| Evidence citation rate | Supplementary | % with substantive evidence (>50 chars or contains PPI keywords) |
| Temporal breakdown | Supplementary | accuracy_pre_2015 vs accuracy_post_2020 |
| Trick vs obvious | Supplementary | accuracy on trick pairs vs obvious untested pairs |

**PPI evidence keywords:**
`{yeast two-hybrid, Y2H, co-immunoprecipitation, co-IP, pulldown, BiFC, FRET, AP-MS, BioGRID, IntAct, HuRI, STRING, complex, interactome, bait, prey}`

---

## 7. Cross-Domain Comparison Framework

### 7.1 Task-Level Alignment Across Domains

| Level | DTI | CT | PPI | Common Pattern |
|-------|-----|-----|-----|----------------|
| L1 | 4-way MCQ (activity classification) | 5-way MCQ (failure category) | 4-way MCQ (evidence quality) | Multi-class classification from context |
| L2 | Extraction from PubMed abstracts | Extraction from why_stopped text | Extraction from constructed summaries | Structured JSON output |
| L3 | Inactivity reasoning | Failure reasoning | Non-interaction reasoning | Free-text scientific explanation |
| L4 | Tested/untested DTI pairs | Tested/untested trial pairs | Tested/untested PPI pairs | Binary discrimination |

### 7.2 Difficulty Gradient Hypothesis

Based on DTI (complete) and CT (in progress) results:

| Level | DTI Result | CT Result (partial) | PPI Prediction | Rationale |
|-------|------------|---------------------|----------------|-----------|
| L1 | Accuracy 0.63-0.66 | Accuracy 0.64-0.66 | Accuracy 0.55-0.70 | PPI evidence descriptions may be less familiar to LLMs than drug/disease terms |
| L2 | Extract F1 0.72-0.76 | Category acc 0.70-0.90 | Entity F1 0.65-0.80 | Constructed text is cleaner than abstracts, but protein name disambiguation is harder |
| L3 | Judge mean 2.5-3.0/5 | Judge mean 2.5-3.5/5 | Judge mean 2.0-3.0/5 | Protein structural reasoning is more specialized than clinical reasoning |
| L4 | MCC <= 0.18 | MCC 0.60-0.76 | MCC 0.10-0.40 | PPI data less contaminated than CT registries but more than DTI assay data |

**Key predictions:**
1. PPI-L1 may be slightly harder than DTI-L1/CT-L1 because evidence quality assessment requires understanding experimental methodology (Y2H, co-IP, ML prediction quality)
2. PPI-L2 may be easier than DTI-L2 because context is constructed (no parsing ambiguity) but harder because protein names have more variants
3. PPI-L3 will likely score lowest across domains because protein structural reasoning is the most specialized knowledge domain
4. PPI-L4 will be between DTI-L4 (near random) and CT-L4 (above random): HuRI is a landmark paper likely in training data, but individual pair data requires memorizing supplementary tables

### 7.3 Cross-Domain Comparison Table (Paper Figure)

The paper will include a multi-panel figure showing:
1. L1 accuracy across DTI/CT/PPI for each of the 5 models (grouped bar chart)
2. L4 MCC across domains (radar chart or parallel coordinates)
3. L4 contamination gap (pre/post temporal split) across domains
4. L3 judge score distribution (violin plot, 4 dimensions x 3 domains)

---

## 8. Prompt Architecture

### System Prompt

```python
PPI_SYSTEM_PROMPT = (
    "You are a molecular biologist with expertise in protein-protein "
    "interactions, structural biology, and proteomics. Provide precise, "
    "evidence-based answers about protein interactions and non-interactions."
)
```

### Prompt Variants

| Config | Description | Runs |
|--------|-------------|------|
| zero-shot | System + question only | 1 (deterministic at temp=0) |
| 3-shot | System + 3 examples + question | 3 (fewshot sets 0/1/2 for variance) |

**Few-shot selection:** From the fewshot pool, select 3 examples per class (PPI-L1: 12 total, PPI-L4: 6 total) using different random seeds per set (42, 43, 44). Examples formatted as:

```
--- Example {i} ---
{context_text}

Answer: {gold_answer}
---
```

### Context Pre-rendering

Following DTI/CT pattern, `context_text` is pre-rendered at dataset build time and stored in the JSONL record. This decouples data construction from inference and ensures reproducibility.

```python
def generate_ppi_context_text(record: dict, task: str) -> str:
    """Generate context text for a PPI LLM benchmark record."""
    if task == "PPI-L1":
        return _format_l1_context(record)
    elif task == "PPI-L2":
        return _format_l2_context(record)
    elif task == "PPI-L3":
        return _format_l3_context(record)
    elif task == "PPI-L4":
        return _format_l4_context(record)
    raise ValueError(f"Unknown task: {task}")
```

---

## 9. Dataset File Format

### JSONL Schema (per record)

```json
{
  "question_id": "PPIL1-0042",
  "task": "PPI-L1",
  "split": "test",
  "difficulty": "medium",
  "context_text": "Protein A: BRCA1 (P38398)\n  Function: ...\n\nProtein B: TP53 ...",
  "gold_answer": "A",
  "gold_category": "direct_experimental",
  "metadata": {
    "pair_id": 12345,
    "protein1_id": 789,
    "protein2_id": 456,
    "gene_symbol_1": "BRCA1",
    "gene_symbol_2": "TP53",
    "uniprot_1": "P38398",
    "uniprot_2": "P04637",
    "source_db": "intact",
    "confidence_tier": "gold",
    "evidence_type": "experimental_non_interaction",
    "publication_year": 2012,
    "compartment_relationship": "same"
  }
}
```

### Output Directory Structure

```
exports/ppi_llm/
+-- ppi_l1_dataset.jsonl      # 1,200 records
+-- ppi_l2_dataset.jsonl      # 500 records
+-- ppi_l2_gold.jsonl         # 500 gold annotations (extraction targets)
+-- ppi_l3_dataset.jsonl      # 200 records
+-- ppi_l4_dataset.jsonl      # 500 records
+-- metadata.json             # Dataset statistics, creation date, DB version
```

### Results Directory Structure

```
results/ppi_llm/
+-- {task}_{model}_{config}_fs{set}/
|   +-- predictions.jsonl
|   +-- results.json
|   +-- run_meta.json
+-- ppi_llm_summary.csv        # Aggregated results across all runs
```

---

## 10. Model Selection

### Target Models

| Model | Provider | Access | Notes |
|-------|----------|--------|-------|
| Llama-3.3-70B | vLLM (HPC) | SLURM | Same as DTI/CT benchmark |
| Qwen2.5-32B-AWQ | vLLM (HPC) | SLURM | Same as DTI/CT benchmark |
| GPT-4o-mini | OpenAI API | Paid | Commercial baseline |
| Gemini-2.5-Flash | Google API | Free tier (250 RPD) | Rate-limited but free |
| Claude-Haiku-4.5 | Anthropic API | Paid | Switched from Sonnet for tier matching + budget |

**Rationale for 5-model consistency:**
- All 5 models are used across DTI, CT, and PPI domains
- Enables direct cross-domain comparison (same model, different tasks)
- Mix of open-source (Llama, Qwen), commercial (GPT, Claude), and free tier (Gemini)
- The DTI benchmark originally used Gemini 2.5 Flash, Llama 3.3 70B, and Mistral 7B; CT expanded to the current 5-model set which PPI inherits

### Inference Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature | 0.0 | Deterministic for reproducibility |
| Max tokens (L1, L4) | 256 | Short structured answers |
| Max tokens (L2) | 1024 | JSON extraction |
| Max tokens (L3) | 2048 | Free-text reasoning |
| Retry | 3 attempts | Exponential backoff |
| Resume | Read existing predictions.jsonl | Skip completed items |
| Rate limit | Model-specific | Gemini: 250 RPD, others: as available |

---

## 11. Experimental Design

### Exp PPI-LLM-1: Cross-Level Performance Profile

Run all 5 models x 4 tasks x 2 configs = 80 total runs.

| Config | Runs per Task | Total |
|--------|--------------|-------|
| zero-shot | 5 models x 1 = 5 | 20 |
| 3-shot (set 0) | 5 models x 1 = 5 | 20 |
| 3-shot (set 1) | 5 models x 1 = 5 | 20 |
| 3-shot (set 2) | 5 models x 1 = 5 | 20 |
| **Total** | | **80** |

**Hypothesis:** Performance will decrease from L1 -> L4, following the DTI/CT pattern. PPI-L1 accuracy ~55-70% (evidence quality classification is tractable but requires domain knowledge), PPI-L4 MCC 0.10-0.40 (between DTI's near-random and CT's above-random).

### Exp PPI-LLM-2: Contamination Analysis (PPI-L4)

Compare pre-2015 (IntAct) vs post-2020 (HuRI + recent IntAct) accuracy.

**Hypothesis:** Pre-2015 accuracy will exceed post-2020 by 10-20% for models whose training data likely includes IntAct publications. The gap may be smaller than CT-L4 because PPI non-interaction data is less prominently reported (supplementary tables vs public registries).

**Additional analysis:**
- Subset by source: IntAct-derived tested pairs vs HuRI-derived tested pairs
- If HuRI pairs show high accuracy despite 2020 publication, this suggests the HuRI dataset was incorporated into training data (landmark paper effect)

### Exp PPI-LLM-3: Evidence Type Difficulty Gradient (PPI-L1)

Stratify PPI-L1 results by evidence category (A/B/C/D) and difficulty level (easy/medium/hard).

**Hypothesis:**
- Category A (experimental) will be easiest to identify (most distinctive language)
- Category D (database absence) will be hardest (most generic-sounding evidence)
- Category B vs C discrimination will be the key differentiator between models (both are large-scale but require understanding experimental vs computational methodology)

### Exp PPI-LLM-4: Cross-Domain Comparison

Compare DTI L1-L4, CT L1-L4, and PPI L1-L4 results on the same 5 models.

**All 5 models overlap across domains**, enabling clean comparison without model-identity confounds.

**Hypotheses:**
1. L1 accuracy ordering: CT >= DTI >= PPI (clinical failure categories most intuitive, PPI evidence quality most specialized)
2. L2 extraction: DTI ~ PPI > CT (DTI has real abstracts, PPI has clean constructed text, CT has short why_stopped text)
3. L3 reasoning: CT >= DTI > PPI (clinical reasoning most accessible, protein structural reasoning most specialized)
4. L4 discrimination: CT >> PPI > DTI (trial registries most contaminated, PPI landmark papers partially contaminated, DTI assay data least contaminated)

### Exp PPI-LLM-5: Compartment Reasoning Analysis (PPI-L3)

Stratify PPI-L3 judge scores by compartment relationship (same vs different compartment).

**Hypothesis:** Models will score higher on different-compartment pairs (easy to reason about spatial separation) than same-compartment pairs (requires deeper mechanistic understanding). The gap quantifies how much LLMs rely on localization heuristics vs genuine structural/functional reasoning.

---

## 12. Contamination Controls

### Source-Level Contamination Risk

| Source | Data Type | Public Access | Contamination Risk |
|--------|-----------|---------------|-------------------|
| IntAct | Curated DB | EMBL-EBI, freely downloadable | Medium -- data available but individual non-interaction records are niche |
| HuRI | Publication | Science (2020), supplementary tables | High -- landmark paper, likely in training data |
| hu.MAP | Web resource | humap3.proteincomplexes.org | Low -- computational predictions, less cited |
| STRING | Web DB | string-db.org | Medium -- widely used but zero-score pairs are the "absence" data, rarely discussed |

### Mitigation Strategies

1. **Source-blind context:** Evidence descriptions never name IntAct, HuRI, hu.MAP, or STRING
2. **Temporal split in L4:** Pre-2015 vs post-2020 quantifies memorization
3. **Trick pairs in L4:** Plausible-but-untested pairs test reasoning vs recall
4. **L3 reasoning requirement:** Even if a model memorized that "BRCA1 and insulin receptor don't interact," it must explain WHY -- memorization alone is insufficient
5. **Paraphrase robustness (future):** Generate 3 semantically equivalent prompts per L1 question; inconsistent answers suggest memorization-based shortcutting

### Contamination Detection Metrics

| Signal | Detection Method | Threshold |
|--------|-----------------|-----------|
| Temporal gap | accuracy_pre_2015 - accuracy_post_2020 | > 15% = flag |
| Source naming | Model mentions IntAct/HuRI/STRING in L1 responses | > 5% = flag |
| PMID citation | Model cites specific PMIDs for tested pairs in L4 | > 10% = flag |
| Screen protocol | Model describes Y2H details for HuRI pairs in L4 | Any = flag |
| Consistency | Same question with paraphrased context yields different answers | > 20% inconsistency = flag |

---

## 13. Implementation Plan

### Priority Order

| Priority | Task | Prerequisite | Script |
|----------|------|-------------|--------|
| P0 | Protein annotation fetch (function, GO, domains) | Migration 002 applied | `scripts_ppi/fetch_protein_annotations.py` |
| P1 | PPI-L1 dataset construction | P0 complete | `scripts_ppi/build_ppi_l1_dataset.py` |
| P2 | PPI-L4 dataset construction + verification | Pair aggregation complete | `scripts_ppi/build_ppi_l4_dataset.py` |
| P3 | PPI-L2 dataset + evidence text construction | P0 complete | `scripts_ppi/build_ppi_l2_dataset.py` |
| P4 | PPI-L3 dataset construction | P0 + annotation quality check | `scripts_ppi/build_ppi_l3_dataset.py` |
| P5 | Inference harness (PPI adapter) | P1-P4 | `scripts_ppi/run_ppi_llm_benchmark.py` |
| P6 | L3 judge pipeline | P5 + GPT-4o-mini API | `scripts_ppi/run_ppi_l3_judge.py` |
| P7 | Results aggregation + cross-domain comparison | P6 + DTI/CT results | `scripts_ppi/collect_ppi_llm_results.py` |

### Code Architecture

```
src/negbiodb_ppi/
+-- llm_prompts.py     # PPI_SYSTEM_PROMPT, format_ppi_prompt(), context generators
+-- llm_eval.py        # evaluate_ppi_l1/l2/l3/l4, parse functions, judge prompt
+-- llm_dataset.py     # Dataset builder utilities (sampling, dedup, split, annotation)
```

Mirrors `src/negbiodb/llm_prompts.py`, `src/negbiodb/llm_eval.py`, and `src/negbiodb_ct/llm_prompts.py` structure.

### Shared Infrastructure (from DTI domain)

The following modules are reused across all three domains:

| Module | File | Reuse |
|--------|------|-------|
| LLM Client | `src/negbiodb/llm_client.py` | HTTP/API calls to all 5 model providers |
| Metrics | `src/negbiodb/metrics.py` | accuracy, F1, MCC computation |
| Judge Framework | `src/negbiodb/llm_eval.py` | L3 judge dispatch + score parsing |

### Annotation Data Requirements

Before building L1/L2/L3 datasets, the following protein annotations must be populated:

| Field | Source | Coverage Target | Fetch Script |
|-------|--------|-----------------|-------------|
| function_description | UniProt API | >= 90% | `fetch_protein_annotations.py` |
| go_terms | UniProt API (GO cross-refs) | >= 85% | Same script |
| domain_annotations | UniProt API (Pfam/InterPro) | >= 80% | Same script |
| subcellular_location | UniProt API | >= 85% | Already in schema, may need backfill |

**Estimated effort:** UniProt batch API supports 100 accessions per request. At 18,412 proteins, ~185 requests needed. With rate limiting, ~30 minutes total.

---

## 14. Expected Outcomes

Based on DTI domain results (complete), CT domain results (in progress), and PPI data characteristics:

| Task | Expected Performance | Rationale |
|------|---------------------|-----------|
| PPI-L1 (MCQ) | Accuracy 55-70% | Evidence quality classification requires understanding experimental methods; less intuitive than clinical failure categories |
| PPI-L2 (Extraction) | Schema compliance 75-90%, Entity F1 0.65-0.80 | Constructed text is cleaner than abstracts but protein name disambiguation adds difficulty |
| PPI-L3 (Reasoning) | Judge mean 2.0-3.0/5 | Protein structural reasoning is the most specialized domain across all three benchmarks |
| PPI-L4 (Discrimination) | MCC 0.10-0.40 | Between DTI (near random) and CT (above random); HuRI as a landmark paper may boost pre-2020 accuracy |

### Key Predictions for Paper

1. **PPI-L3 will show the lowest judge scores across all three domains** -- protein structural reasoning requires specialized knowledge that general-purpose LLMs lack, unlike clinical trial reasoning (CT-L3) which draws on broadly accessible medical knowledge
2. **PPI-L4 contamination gap will be intermediate** -- smaller than CT (public registry) but larger than DTI (specialized assay databases), reflecting the mixed accessibility of PPI data
3. **PPI-L1 category D (database absence) will be the hardest class** -- "absence of evidence" is inherently harder to classify than explicit experimental evidence
4. **Same-compartment L3 pairs will be scored ~0.5-1.0 points lower** than different-compartment pairs, revealing LLMs' reliance on localization heuristics

---

## 15. Open Design Questions

1. **PPI-L1 Category A sample size:** Only 779 IntAct records exist. If annotation filtering (function_description, go_terms, etc.) reduces the eligible pool below 300, options are: (a) relax annotation requirements for L1 only, (b) reduce per-class size to 200, (c) augment with IntAct records from non-human organisms mapped to human orthologs.

2. **PPI-L2 constructed text realism:** The fallback to constructed evidence summaries means L2 tests extraction from idealized text, not real publication language. This reduces ecological validity but ensures sufficient dataset size. Should we include a small validation subset (n=50) of real abstracts from the 65 IntAct PMIDs for comparison?

3. **PPI-L3 protein annotation quality:** function_description and domain_annotations from UniProt vary in detail. Well-studied proteins (TP53, BRCA1) have paragraphs; poorly characterized proteins have single sentences. Should we normalize annotation length, or let the natural variation serve as implicit difficulty?

4. **PPI-L4 HuRI contamination:** HuRI's entire dataset was published as supplementary files. Some LLMs may have ingested the full supplementary tables. If HuRI-derived tested pairs show >80% accuracy, this would suggest wholesale memorization of the dataset. Should we include a "HuRI-specific" contamination sub-analysis?

5. **Judge model for L3:** GPT-4o-mini is specified, but the DTI domain used Gemini Flash (free tier) and CT proposed either Gemini or Claude Sonnet. Should all three domains use the same judge model for consistent cross-domain scoring, or is domain-specific judge selection acceptable?

---

## 16. Summary of Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| L1 task type | Evidence quality (not interaction prediction) | Mirrors DTI-L1/CT-L1 pattern; tests understanding of evidence methodology |
| L1 class count | 4-way (not 5 or 7) | IntAct gold/silver too small to separate; 4 represents distinct quality tiers |
| L2 input source | Constructed evidence text (not PubMed abstracts) | Only 65 unique PMIDs insufficient for 500-record extraction benchmark |
| L3 compartment balance | ~50/50 same vs different | Tests localization heuristic reliance vs genuine structural reasoning |
| L3 judge model | GPT-4o-mini | Protein biology reasoning quality; manageable cost |
| L4 temporal split | Pre-2015 / post-2020 | IntAct pre-2015 has ~50 PMIDs, HuRI published 2020; 5-year gap for clean separation |
| L4 trick pairs | Same pathway/compartment | Tests whether LLMs mistake pathway co-membership for direct interaction evidence |
| Evidence descriptions | Source-blind (no DB names) | Prevents trivial shortcutting via database name pattern matching |
| Model set | Same 5 as DTI/CT | Enables direct cross-domain comparison |
| Context pre-rendering | At dataset build time | Reproducibility; decouples construction from inference |
