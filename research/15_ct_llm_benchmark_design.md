# 15. CT Domain LLM Benchmark Design

## 1. Overview

The NegBioDB-CT LLM benchmark evaluates large language models on clinical trial failure understanding tasks across four difficulty levels. It mirrors the DTI domain's L1–L4 structure but adapts to clinical trial characteristics: textual failure descriptions, quantitative outcome data, categorical failure taxonomy, and public trial registry contamination risk.

### Database State (as of 2026-03-18)
- **132,925 failure results** across 28,135 unique trials
- **Tier distribution:** gold 23,570 / silver 28,505 / bronze 60,223 / copper 20,627
- **Category distribution:** efficacy 42.6% / enrollment 23.0% / other 21.0% / strategic 7.1% / safety 3.9% / design 1.8% / regulatory 0.7% / PK 0.0%
- **failure_detail content by tier:**
  - Gold/Silver: structured p-value strings (e.g., `"p=0.4900 (t-test, 2 sided)"`)
  - Bronze: natural language from `why_stopped` (e.g., `"Difficult to recruit patients"`)
  - Copper: placeholder `"CTO binary failure label"` (excluded from L1/L2/L3)
- **why_stopped availability:** bronze 100%, silver 21.8%, gold 8.2%, copper variable

---

## 2. Level Definitions

| Level | Task | Size | Input | Output | Eval | Difficulty |
|-------|------|------|-------|--------|------|------------|
| CT-L1 | Failure Category MCQ (5-way) | 1,500 | Trial context + failure evidence | A/B/C/D/E letter | Accuracy, macro_F1, MCC | Low |
| CT-L2 | Failure Report Extraction | 500 | why_stopped text + trial metadata | Structured JSON | Schema compliance, field F1 | Medium |
| CT-L3 | Failure Reasoning | 200 | Drug + condition + quantitative evidence | Free text explanation | LLM-as-Judge (4 dim) | High |
| CT-L4 | Trial Existence Discrimination | 500 | Drug + condition description | "tested" / "untested" | Accuracy, F1, MCC | Highest |

### Category Mapping

**CT-L1 uses 5-way classification** (collapsed from 7 active categories):

| MCQ Option | Source Categories | Records Available* |
|------------|-------------------|-------------------|
| A) Safety | safety | 5,164 |
| B) Efficacy | efficacy | 56,588 |
| C) Enrollment | enrollment | 30,567 |
| D) Strategic | strategic | 9,400 |
| E) Other | design + regulatory + other (non-copper) | 10,579 |

*Pre-filter counts (before excluding Placebo, non-drug interventions, and copper tier). The dataset build script must verify post-filter counts per class; if any class drops below 300, adjust dataset size target.

**Rationale for 5-way over 7-way:**
- Regulatory (925) and design (2,377) are too small for reliable per-class evaluation at 300 samples/class
- 7-way MCQ creates high cognitive load; 5-way matches DTI L1's 4-way complexity
- CT-L2 extraction uses full 7-way for fine-grained evaluation (automated scoring, no per-class sample floor)

**Copper tier (20,627) excluded from CT-L1/L2/L3:**
- `failure_detail = "CTO binary failure label"` — no meaningful text for classification or extraction
- `result_interpretation = NULL` — no interpretive signal
- `failure_category = "other"` — always maps to E, creating trivial examples
- Copper is only used in CT-L4 where binary tested/untested is the task

---

## 3. CT-L1: Failure Category Classification (MCQ)

### Task Definition

Given clinical trial metadata and failure evidence, classify the trial's failure into one of 5 categories.

### Data Source Selection

| Tier | Available | Usable for L1 | Context Type |
|------|-----------|----------------|--------------|
| Gold | 23,570 | Yes | Drug + condition + phase + p-value + endpoint + effect size |
| Silver | 28,505 | Yes | Drug + condition + phase + p-value + endpoint |
| Bronze | 60,223 | Yes | Drug + condition + phase + why_stopped text |
| Copper | 20,627 | **No** | Excluded (generic placeholder) |

### Difficulty Stratification

| Difficulty | Tier | Criteria | % of Dataset |
|------------|------|----------|--------------|
| Easy | Gold | Phase III + quantitative p-value + endpoint met/not met | 40% |
| Medium | Silver | Phase II/III + p-value but less definitive | 35% |
| Hard | Bronze | Only why_stopped text, often short/ambiguous | 25% |

**Difficulty operationalization (tier-based, independent of class label):**
- Easy: gold tier, any category — quantitative p-value + endpoint data provides clear signal
- Medium: silver tier, any category — p-value available but less definitive evidence
- Hard: bronze tier, any category — only `why_stopped` text, often short/ambiguous

### Dataset Construction

**Total: 1,500 records** (300 per class × 5 classes)

| Split | Size | Per Class | Purpose |
|-------|------|-----------|---------|
| Fewshot pool | 300 | 60 | 3-shot example selection (3 per class = 15 total) |
| Validation | 300 | 60 | Hyperparameter tuning |
| Test | 900 | 180 | Final evaluation |

**Sampling strategy per class:**
1. Exclude copper tier entirely
2. Stratify by difficulty: target 40% easy (gold) / 35% medium (silver) / 25% hard (bronze) within each class. **Note:** Gold tier is 100% efficacy and silver is 91% efficacy (per tier×category cross-tab). For non-efficacy classes (safety, enrollment, strategic, other), shift allocation to available tiers (e.g., medium + hard only). Report actual per-class difficulty distribution in dataset metadata.
3. For class E (other): sample proportionally from design, regulatory, and non-copper other
4. Deduplicate: MAX_PER_DRUG = 10 (prevent single drug dominating a class)
5. Exclude Placebo and non-drug interventions (procedure, device, behavioral, radiation, dietary, genetic)
6. For multi-arm trials: include the specific drug arm in context text to disambiguate (same trial may have safety failure for drug A and efficacy failure for drug B)

### Context Text Generation

**Gold/Silver records:**
```
Trial: {drug_name} for {condition_name}
Phase: {trial_phase}
Design: {blinding}, {control_type}
Enrollment: {enrollment_actual} participants
Primary endpoint: {endpoint_description}
Result: p = {p_value_primary} ({statistical_method})
Effect size: {effect_size} [{ci_lower}, {ci_upper}]
Serious adverse events: {serious_adverse_events}
```

**Bronze records:**
```
Trial: {drug_name} for {condition_name}
Phase: {trial_phase}
Design: {blinding}, {control_type}
Enrollment: {enrollment_actual} participants
Termination reason: "{why_stopped}"
```

### Prompt Design

```python
# Uses CT_SYSTEM_PROMPT (defined in §7)

CT_L1_QUESTION = (
    "Based on the clinical trial information below, classify the primary "
    "reason for this trial's failure.\n\n"
    "{context_text}\n\n"
    "Categories:\n"
    "A) Safety — Trial failed due to drug toxicity, adverse events, or safety signals\n"
    "B) Efficacy — Trial failed to demonstrate therapeutic benefit vs control\n"
    "C) Enrollment — Trial failed to recruit sufficient participants\n"
    "D) Strategic — Trial was discontinued for business, strategic, or portfolio reasons\n"
    "E) Other — Trial failed due to study design flaws, regulatory issues, or other reasons\n"
)

CT_L1_ANSWER_FORMAT = "Respond with ONLY a single letter (A, B, C, D, or E)."
```

### Evaluation Metrics

| Metric | Primary? | Description |
|--------|----------|-------------|
| Accuracy | Yes | Overall correct classification rate |
| Macro F1 | Yes | Unweighted average F1 across 5 classes |
| MCC | Yes | Matthews correlation coefficient (multiclass) |
| Per-class accuracy | Supplementary | Breakdown by A/B/C/D/E |
| Parse rate | Supplementary | % of responses successfully parsed as A-E |
| Difficulty breakdown | Supplementary | Accuracy by easy/medium/hard |

---

## 4. CT-L2: Failure Report Extraction

### Task Definition

Given a trial's termination text (`why_stopped`) and basic metadata, extract structured failure information as JSON.

### Data Source

**Bronze tier only (60,223 records)** — this is the only tier with natural language `why_stopped` text at near-100% availability. Gold/silver have structured p-value strings in `failure_detail` which are not suitable for extraction tasks.

### Target Schema

```json
{
  "failure_category": "efficacy | safety | enrollment | strategic | design | regulatory | other",
  "failure_subcategory": "string (free text, e.g., 'futility', 'hepatotoxicity')",
  "affected_system": "string or null (e.g., 'hepatic', 'cardiovascular')",
  "severity_indicator": "mild | moderate | severe | fatal | null",
  "quantitative_evidence": "boolean (does text mention numbers/statistics?)",
  "decision_maker": "sponsor | dsmb | regulatory | investigator | null",
  "patient_impact": "string or null (e.g., 'no safety concerns', '3 deaths reported')"
}
```

### Difficulty Stratification

| Difficulty | Criteria | Example why_stopped | % |
|------------|----------|---------------------|---|
| Easy | Clear single-reason, explicit category keywords | "Study terminated due to hepatotoxicity in 3 patients" | 40% |
| Medium | Multi-reason or implicit category | "Sponsor decided to terminate due to competitive landscape and slow enrollment" | 35% |
| Hard | Vague, short, or ambiguous | "Business reasons" / "Study was stopped" | 25% |

### Dataset Construction

**Total: 500 records**

| Split | Size | Purpose |
|-------|------|---------|
| Fewshot pool | 50 | 3-shot example selection |
| Validation | 50 | Schema validation tuning |
| Test | 400 | Final evaluation |

**Sampling strategy:**
1. Bronze tier only (why_stopped required, non-empty)
2. **Pre-flight:** Query bronze-tier category distribution (may differ from overall DB distribution — bronze is NLP-classified and may underrepresent safety/efficacy vs gold/silver)
3. 7-way category distribution: proportional to bronze-specific frequency (capped at 40% for efficacy)
4. Stratify by difficulty within each category
5. Min 20 records per category in test set (regulatory ~20, design ~30) — verify post-filter
6. Exclude records where `why_stopped` is identical to another record (template dedup)

### Gold Standard

The extraction gold standard is derived from two sources:
1. **failure_category** — from NLP classification (etl_classify.py), serves as ground truth for the primary field
2. **Manual annotation** — 500 records hand-annotated for subcategory, affected_system, severity, decision_maker, patient_impact fields

Manual annotation is required because the database only stores `failure_category` at the top level. The subcategory and detail fields require human judgment on the `why_stopped` text.

**Recommended approach (phased):** Phase 1 uses `failure_category` as the sole gold field (automated evaluation, no annotation needed). Phase 2 annotates 100 records for calibration + uses high-agreement LLM consensus (3 models agree) for the remaining 400. This reduces manual effort from ~50h to ~10h while maintaining quality.

### Prompt Design

```python
CT_L2_QUESTION = (
    "Extract structured failure information from the following clinical trial "
    "termination report. Return a JSON object with the fields specified below.\n\n"
    "Trial: {drug_name} for {condition_name}\n"
    "Phase: {trial_phase}\n"
    "Termination text: \"{why_stopped}\"\n\n"
    "Required JSON fields:\n"
    "- failure_category: one of [efficacy, safety, enrollment, strategic, design, regulatory, other]\n"
    "- failure_subcategory: specific reason (e.g., 'futility', 'hepatotoxicity', 'slow accrual')\n"
    "- affected_system: organ system affected (null if not applicable)\n"
    "- severity_indicator: one of [mild, moderate, severe, fatal, null]\n"
    "- quantitative_evidence: true if text mentions specific numbers or statistics\n"
    "- decision_maker: who terminated [sponsor, dsmb, regulatory, investigator, null]\n"
    "- patient_impact: brief description of patient safety impact (null if not mentioned)\n\n"
    "Return ONLY valid JSON, no additional text."
)
```

### Evaluation Metrics

| Metric | Primary? | Description |
|--------|----------|-------------|
| Schema compliance | Yes | % of responses that parse as valid JSON with all 7 fields |
| Category accuracy | Yes | Exact match on failure_category field |
| Field F1 (micro) | Yes | Token-level F1 across all string fields |
| Subcategory F1 | Supplementary | Fuzzy match on failure_subcategory (Jaro-Winkler > 0.85) |
| Severity accuracy | Supplementary | Exact match on severity_indicator |
| Decision-maker accuracy | Supplementary | Exact match on decision_maker |

---

## 5. CT-L3: Failure Reasoning

### Task Definition

Given a drug, condition, trial phase, and quantitative outcome data, provide a scientific explanation for why this clinical trial failed.

### Data Source

**Gold tier only (23,570 records)** — requires quantitative evidence (p-value, effect size, SAE data) for meaningful reasoning. LLM must go beyond restating the numbers to explain the biological/pharmacological/clinical reasons.

**Additional filters:**
1. Drug must have ChEMBL resolution (SMILES available for molecular context)
2. Failure category in {safety, efficacy} (clearest scientific reasoning)
3. Phase II or III (most clinically informative)
4. `has_results = 1` (posted results on ClinicalTrials.gov)

**Estimated eligible pool:** At current ~13% ChEMBL resolution: gold (23,570) × 13% ChEMBL × 46.5% safety+efficacy × 56.6% Phase II-III ≈ ~800 records. At target 35-45% resolution after fuzzy matching: ~2,200–2,800 records. The 200-record dataset is feasible from either pool, but CT-L3 dataset construction should wait for drug resolution to reach ≥ 25% coverage.

### Dataset Construction

**Total: 200 records**

| Split | Size | Purpose |
|-------|------|---------|
| Fewshot pool | 20 | 3-shot example selection |
| Validation | 20 | Judge calibration |
| Test | 160 | Final evaluation |

**Diversity requirements:**
- Therapeutic area balance: oncology ≤ 40%, cardiology ≤ 15%, neurology ≤ 15%, other ≥ 30%
- Drug type balance: small molecule ≥ 60%, biologic ≥ 20%
- Safety vs efficacy: ~50/50 split
- Phase II vs III: ~50/50 split

> **Deployment note (2026-03-23):** The safety/efficacy balance was unachievable at time of construction. Gold-tier records with ChEMBL resolution, Phase II/III, and `failure_category='safety'` yielded zero eligible records after all filters. The deployed L3 dataset is 200/200 efficacy-only. The build script logs a warning and falls back to all-efficacy. This must be disclosed as a limitation in the paper: CT-L3 evaluates reasoning about efficacy failures only, not the full safety/efficacy spectrum originally intended.

### Context Text Generation

```
Drug: {drug_name} ({molecular_type})
SMILES: {canonical_smiles}
Known targets: {target_list from intervention_targets}
Condition: {condition_name}
Therapeutic area: {therapeutic_area}
Phase: {trial_phase}
Design: {randomized}, {blinding}, {control_type}
Enrollment: {enrollment_actual} participants

Primary endpoint: {endpoint_description}
Result: p = {p_value_primary} ({statistical_method})
Effect size: {effect_size}
Confidence interval: [{ci_lower}, {ci_upper}]
Serious adverse events: {serious_adverse_events}
Primary endpoint met: {endpoint_met}
Interpretation: {result_interpretation}
```

### Prompt Design

```python
CT_L3_QUESTION = (
    "The following clinical trial was confirmed as a FAILURE. Based on the trial "
    "data below, provide a scientific explanation for why this drug failed in this "
    "clinical trial.\n\n"
    "{context_text}\n\n"
    "Your explanation should address:\n"
    "1. Mechanism — What is the drug's mechanism of action and why might it be "
    "insufficient for this condition?\n"
    "2. Evidence interpretation — What do the statistical results tell us about "
    "the magnitude and confidence of the failure?\n"
    "3. Clinical factors — What trial design, patient population, or disease "
    "biology factors may have contributed?\n"
    "4. Broader context — How does this failure relate to the known challenges "
    "of treating this condition or drug class?\n\n"
    "Provide a thorough explanation in 3-5 paragraphs."
)
```

### Evaluation: LLM-as-Judge

**4 dimensions, each scored 1–5:**

| Dimension | Description | Scoring Guide |
|-----------|-------------|---------------|
| Accuracy | Factual correctness of claims about the drug, target, and condition | 5 = all claims verifiable, 1 = major factual errors |
| Reasoning | Logical coherence of the causal explanation | 5 = clear causal chain, 1 = non-sequiturs |
| Completeness | Coverage of mechanism, evidence, clinical factors, and context | 5 = all 4 aspects addressed, 1 = only 1 aspect |
| Specificity | Trial-specific analysis vs generic statements | 5 = cites specific p-values, endpoints, patient factors, 1 = could apply to any trial |

**Judge prompt:**
```python
CT_L3_JUDGE_PROMPT = (
    "You are evaluating a scientific explanation for a clinical trial failure.\n\n"
    "TRIAL CONTEXT:\n{context_text}\n\n"
    "GROUND TRUTH CATEGORY: {failure_category}\n\n"
    "RESPONSE TO EVALUATE:\n{response_text}\n\n"
    "Score the response on 4 dimensions (1-5 each):\n"
    "1. accuracy: Are factual claims about the drug, target, and disease correct?\n"
    "2. reasoning: Is the causal explanation logically coherent?\n"
    "3. completeness: Does it address mechanism, evidence, clinical factors, and context?\n"
    "4. specificity: Does it reference specific trial data (p-values, endpoints) "
    "rather than making generic statements?\n\n"
    "Return ONLY a JSON object: {\"accuracy\": N, \"reasoning\": N, \"completeness\": N, \"specificity\": N}"
)
```

---

## 6. CT-L4: Trial Existence Discrimination

### Task Definition

Given a drug and a disease condition, determine whether the pair was ever tested in a registered clinical trial.

### Design Rationale

ClinicalTrials.gov is publicly accessible and heavily represented in LLM training data. This task tests whether LLMs have memorized the trial registry or can genuinely reason about drug-condition plausibility. The DTI-L4 analog showed MCC ≤ 0.18 (near random), suggesting LLMs cannot reliably distinguish tested from untested pairs even in well-studied domains.

**Design note:** The original CT domain design (research/13) proposed "Phase Transition Judgment" (Phase II → III prediction) as CT-L4. This was replaced with tested/untested discrimination to: (1) enable direct cross-domain comparison with DTI-L4 (Exp CT-LLM-4), and (2) test the contamination hypothesis which is especially relevant for public trial registries. The phase transition task may be revisited as a future CT-L5.

### Data Source

**Tested pairs (250):**
- Source: `intervention_condition_pairs` table
- 125 pre-2020 pairs: high contamination risk (trials well-documented before LLM training cutoff)
- 125 post-2023 pairs: low contamination risk (recent trials, less likely in training data)
- Drug must have a recognizable name (not codes like "BMS-123456")
- Condition must be a standard disease name

**Untested pairs (250):**
- 125 trick pairs: well-known drug × plausible but never-tested condition
  - e.g., "Pembrolizumab for Alzheimer's Disease" (immunotherapy × neurodegeneration)
  - Must pass plausibility check: drug mechanism is tangentially relevant to condition
- 125 obvious untested: drug × clearly unrelated condition
  - e.g., "Metformin for Hearing Loss" (diabetes drug × ENT condition)
  - Serves as baseline — models should get these right

### Verification Protocol

1. All "untested" pairs verified against:
   - `intervention_condition_pairs` table (no match)
   - `trial_interventions` × `trial_conditions` JOIN (no shared trial)
   - ClinicalTrials.gov API spot-check (10% sample)
   - PubMed search for `"{drug_name}" AND "{condition_name}" AND "clinical trial"` (catch unregistered studies)
   - Manual review of all 250 untested pairs by domain expert
2. All "tested" pairs verified:
   - Exist in `intervention_condition_pairs` with ≥ 1 trial
   - Drug name resolves to a real intervention in the database

### Dataset Construction

**Total: 500 records**

| Split | Size | Per Class | Purpose |
|-------|------|-----------|---------|
| Fewshot pool | 50 | 25 tested + 25 untested | 3-shot example selection |
| Validation | 50 | 25 tested + 25 untested | Threshold tuning |
| Test | 400 | 200 tested + 200 untested | Final evaluation |

### Context Text Generation

```
Drug: {drug_name}
Drug type: {molecular_type}
Condition: {condition_name}
Therapeutic area: {therapeutic_area}

Question: Has this drug-condition combination ever been tested in a registered
clinical trial?
```

**Note:** No SMILES, no phase, no outcome data — the model must rely on its knowledge of the clinical trial landscape.

### Prompt Design

```python
CT_L4_QUESTION = (
    "Based on your knowledge of clinical trials and drug development, determine "
    "whether the following drug-condition combination has ever been tested in a "
    "registered clinical trial (e.g., on ClinicalTrials.gov).\n\n"
    "{context_text}\n\n"
    "On the first line, respond with ONLY 'tested' or 'untested'.\n"
    "On the second line, provide brief evidence for your answer (e.g., trial "
    "identifiers, known results, or reasoning for why it was/wasn't tested)."
)
```

### Anti-Contamination Analysis

ClinicalTrials.gov data is public → high contamination risk for pre-cutoff trials.

**Temporal cutoff rationale:** CT uses pre-2020/post-2023 (vs DTI's pre-2023/post-2024) because: (1) CT data volume drops sharply after 2022 (2023: 2,567, 2024: 975 records), requiring a wider post-cutoff window to reach 125 pairs; (2) the 2020-2022 gap includes the COVID-19 spike (12,600 results in 2020 alone), which confounds contamination with distribution shift. The 3-year gap is wider than DTI's 1-year gap, so contamination flag thresholds may need adjustment.

| Metric | Description |
|--------|-------------|
| accuracy_pre_2020 | Accuracy on trials registered before 2020 |
| accuracy_post_2023 | Accuracy on trials registered 2023+ |
| accuracy_gap | pre_2020 − post_2023 |
| contamination_flag | accuracy_gap > 15% suggests memorization |

**COVID confound:** The 2020 spike may inflate pre-2020 vs post-2023 gap due to distribution shift (not just memorization). Sensitivity analysis: report accuracy excluding COVID-condition trials separately.

**Additional contamination signals:**
- NCT ID citation: if model outputs correct NCT ID for tested pairs → memorization
- Evidence specificity: model cites specific results (p-values, dates) not in prompt → memorization
- Paraphrase detection: response similarity to `why_stopped` text (cosine > 0.8 → memorization)

### Evaluation Metrics

| Metric | Primary? | Description |
|--------|----------|-------------|
| Accuracy | Yes | Overall correct classification rate |
| F1 (tested) | Yes | Binary F1 with pos_label="tested" |
| MCC | Yes | Matthews correlation coefficient |
| Parse rate | Supplementary | % of responses parseable as tested/untested |
| Evidence citation rate | Supplementary | % with substantive evidence (> 50 chars or contains keywords) |
| Temporal breakdown | Supplementary | accuracy_pre_2020 vs accuracy_post_2023 |

**Evidence keywords (CT-specific):**
`{clinicaltrials, nct, pubmed, doi, pmid, p-value, hazard, aact, eudract, fda, endpoint}`

---

## 7. Prompt Architecture

### System Prompt

```python
CT_SYSTEM_PROMPT = (
    "You are a clinical trial expert with deep knowledge of drug development, "
    "regulatory science, and clinical pharmacology."
)
```

### Prompt Variants

| Config | Description | Runs |
|--------|-------------|------|
| zero-shot | System + question only | 1 (deterministic at temp=0) |
| 3-shot | System + 3 examples + question | 3 (fewshot sets 0/1/2 for variance) |

**Few-shot selection:** From the fewshot pool, select 3 examples per class (CT-L1: 15 total, CT-L4: 6 total) using different random seeds per set (42, 43, 44). Examples formatted as:

```
--- Example {i} ---
{context_text}

Answer: {gold_answer}
---
```

### Context Pre-rendering

Following DTI pattern, `context_text` is pre-rendered at dataset build time and stored in the JSONL record. This decouples data construction from inference, ensuring reproducibility.

```python
def generate_ct_context_text(record: dict, task: str) -> str:
    """Generate context text for a CT LLM benchmark record."""
    # Task-specific formatting
    if task == "CT-L1":
        return _format_l1_context(record)
    elif task == "CT-L2":
        return _format_l2_context(record)
    # ...
```

---

## 8. Dataset File Format

### JSONL Schema (per record)

```json
{
  "question_id": "CTL1-0042",
  "task": "CT-L1",
  "split": "test",
  "difficulty": "medium",
  "context_text": "Trial: Pembrolizumab for Breast Cancer\nPhase: phase_3\n...",
  "gold_answer": "B",
  "gold_category": "efficacy",
  "metadata": {
    "trial_failure_result_id": 12345,
    "source_trial_id": "NCT02345678",
    "intervention_id": 789,
    "condition_id": 456,
    "confidence_tier": "silver",
    "trial_phase": "phase_3",
    "drug_name": "Pembrolizumab",
    "condition_name": "Breast Cancer"
  }
}
```

### Output Directory Structure

```
exports/ct_llm/
├── ct_l1_dataset.jsonl      # 1,500 records
├── ct_l2_dataset.jsonl      # 500 records
├── ct_l2_gold.jsonl         # 500 gold annotations (extraction targets)
├── ct_l3_dataset.jsonl      # 200 records
├── ct_l4_dataset.jsonl      # 500 records
└── metadata.json            # Dataset statistics, creation date, DB version
```

### Results Directory Structure

```
results/ct_llm/
├── {task}_{model}_{config}_fs{set}/
│   ├── predictions.jsonl
│   ├── results.json
│   └── run_meta.json
└── ct_llm_summary.csv        # Aggregated results across all runs
```

---

## 9. Model Selection

### Target Models

| Model | Provider | Access | Notes |
|-------|----------|--------|-------|
| Llama-3.1-70B | vLLM (local) | HPC | Same as DTI benchmark |
| Llama-3.1-8B | vLLM (local) | HPC | Smaller model baseline |
| Gemini 2.0 Flash | Google API | Free tier | Judge model for L3 |
| GPT-4o-mini | OpenAI API | Paid | Commercial baseline |
| Claude Sonnet 4.6 | Anthropic API | Paid | High-capability baseline |

### Inference Configuration

- Temperature: 0.0 (deterministic)
- Max tokens: 256 (L1, L4), 1024 (L2), 2048 (L3)
- Retry: 3 attempts with exponential backoff
- Resume: read existing `predictions.jsonl` to skip completed items

---

## 10. Experimental Design

### Exp CT-LLM-1: Cross-Level Performance Profile

Run all 5 models × 4 tasks × 2 configs = 80 total runs (20 zero-shot + 60 three-shot with 3 fewshot sets each).

**Hypothesis:** Performance will decrease from L1 → L4 (same pattern as DTI). L1 accuracy ~60-75% (classification is tractable), L4 MCC near random (~0.1-0.2, same as DTI-L4).

### Exp CT-LLM-2: Contamination Analysis (CT-L4)

Compare pre-2020 vs post-2023 accuracy to quantify training data memorization.

**Hypothesis:** pre-2020 accuracy will exceed post-2023 by >15% for larger models (Llama-70B, GPT-4o-mini, Claude Sonnet 4.6) that likely ingested ClinicalTrials.gov during pretraining.

### Exp CT-LLM-3: Tier Difficulty Gradient (CT-L1)

Stratify CT-L1 results by difficulty (easy/medium/hard) and tier (gold/silver/bronze).

**Hypothesis:** Easy (gold) accuracy > medium (silver) > hard (bronze). The gap quantifies how much quantitative evidence helps LLMs vs pure text understanding.

### Exp CT-LLM-4: Cross-Domain Comparison (DTI vs CT)

Compare DTI L1-L4 results with CT L1-L4 results on overlapping models. **Note:** DTI used Gemini 2.5 Flash, Llama 3.3 70B, Mistral 7B; CT proposes Llama-3.1-70B, Gemini 2.0 Flash, GPT-4o-mini. For direct comparison, at least 2 models must be shared — either run CT tasks on DTI models (Llama 3.3, Gemini 2.5 Flash) or add DTI tasks to CT models retroactively.

**Hypothesis:** CT tasks may show higher absolute accuracy than DTI (clinical trials are better represented in training data than molecular pharmacology), but similar relative difficulty ordering across levels.

---

## 11. Implementation Priority

| Priority | Task | Prerequisite | Script |
|----------|------|-------------|--------|
| P0 | CT-L1 dataset construction | Drug resolution complete | `scripts_ct/build_ct_l1_dataset.py` |
| P1 | CT-L4 dataset construction | Pair aggregation complete | `scripts_ct/build_ct_l4_dataset.py` |
| P2 | CT-L2 dataset + manual annotation | None (bronze tier ready) | `scripts_ct/build_ct_l2_dataset.py` |
| P3 | CT-L3 dataset construction | Drug resolution complete | `scripts_ct/build_ct_l3_dataset.py` |
| P4 | Inference harness (CT adapter) | P0-P3 | `scripts_ct/run_ct_llm_benchmark.py` |
| P5 | L3 judge pipeline | P4 + Gemini API | `scripts_ct/run_ct_l3_judge.py` |
| P6 | Results aggregation | P4-P5 | `scripts_ct/collect_ct_llm_results.py` |

### Code Architecture

```
src/negbiodb_ct/
├── llm_prompts.py     # CT_SYSTEM_PROMPT, format_ct_prompt(), context generators
├── llm_eval.py        # evaluate_ct_l1/l2/l3/l4, parse functions, judge prompt
└── llm_dataset.py     # Dataset builder utilities (sampling, dedup, split)
```

Mirrors `src/negbiodb/llm_prompts.py` and `src/negbiodb/llm_eval.py` structure from DTI domain.

---

## 12. Expected Outcomes

Based on DTI domain experience and clinical trial data characteristics:

| Task | Expected Performance | Rationale |
|------|---------------------|-----------|
| CT-L1 (MCQ) | Accuracy 55-75% | Classification from rich context; easier than DTI-L1 due to textual cues |
| CT-L2 (Extraction) | Schema compliance 70-90% | JSON extraction well-suited to instruction-following models |
| CT-L3 (Reasoning) | Judge mean 2.5-3.5/5 | Similar to DTI-L3; specificity score will be low |
| CT-L4 (Discrimination) | MCC 0.1-0.3 | Near random for post-2023; slight above-chance for pre-2020 (memorization) |

**Key prediction:** CT-L1 will outperform DTI-L1 because clinical trial failure descriptions contain direct textual cues ("terminated due to safety concerns") while DTI requires molecular reasoning. The **gap** between L1 and L4 should be wider in CT than DTI.

---

## 13. Open Design Questions

1. **CT-L2 gold standard creation:** Manual annotation of 500 records is a bottleneck. Options: (a) annotate all 500 manually, (b) annotate 100 as calibration + use high-agreement LLM consensus for remaining 400, (c) use only the `failure_category` field as gold and defer detailed extraction scoring.

2. **CT-L3 judge model:** DTI uses Gemini free tier. For CT-L3, the judge needs clinical trial expertise — should we use Claude Sonnet 4.6 instead of Gemini for higher-quality judging?

3. **CT-L4 trick pair construction:** Requires clinical expertise to create plausible-but-untested drug-condition pairs. Options: (a) manual curation by domain expert, (b) LLM-assisted generation with human verification, (c) systematic generation from drug mechanism × disease ontology mapping.

4. **Biologics in CT-L3:** Biologics lack SMILES but have mechanism data. Include them with molecular_type + target info only, or restrict to small molecules with full structural context?
