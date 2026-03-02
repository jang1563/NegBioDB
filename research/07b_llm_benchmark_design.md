# LLM Benchmark Design for NegBioBench

> Comprehensive design for LLM evaluation tasks on negative DTI data (2026-03-02)

---

## Executive Summary

No existing benchmark evaluates LLMs on any negative DTI task — not predicting inactivity, not extracting negative results from papers, not explaining why compounds don't bind targets. This represents a **completely unoccupied niche** in the LLM benchmark landscape.

NegBioBench will be a **dual-track benchmark**: traditional ML tasks (sequence/graph-based DTI prediction) + LLM tasks (text-based reasoning, extraction, and prediction). This dual design maximizes the benchmark's reach across both the cheminformatics and the LLM research communities.

### Cost Strategy for LLM Evaluation

**Phase 1 (MVP/NeurIPS submission): $0**
- Evaluation models: Gemini 2.5 Flash free tier, Llama 3.3 (ollama local), Mistral 7B (ollama local)
- LLM-as-Judge: Gemini 2.5 Flash free tier (for automated scoring of open-ended responses)
- Human evaluation: Manual review on sampled outputs

**Phase 2 (Post-stabilization): Flagship models**
- GPT-4/4.1, Claude Opus/Sonnet, Gemini Pro — added to leaderboard once benchmark design is stable
- Paid API costs justified by publication/impact value at that stage

---

## 1. Research Landscape Summary

### 1.1 Key Findings from Literature Survey

**Existing bio/chem LLM benchmarks (none address negative results):**

| Benchmark | Domain | Tasks | Limitation for Us |
|-----------|--------|-------|-------------------|
| ChemBench (Nature Chemistry 2025) | Chemistry QA | 2,700 MCQ + open-ended | No negative result tasks |
| Mol-Instructions (ICLR 2024) | Molecular tasks | 148K molecule instructions | Positive-outcome oriented |
| LlaSMol/SMolInstruct (COLM 2024) | Chemistry | 14 tasks, 3M+ samples | No inactivity prediction |
| MedQA | Medical | USMLE-style MCQ | Saturating; not drug discovery |
| SciBench (ICML 2024) | Science reasoning | College-level open problems | General science, not DTI |
| SciKnowEval | Science knowledge | 70K problems, 5 levels | No DTI-specific tasks |
| DrugChat (bioRxiv 2024) | Drug properties | Free-form prediction | Positive interactions focus |
| LAB-Bench | Lab capabilities | Literature, protocols | General biology |

**LLM capabilities for DTI (current state):**
- Fine-tuned small models (Phi-3.5, 2.7B) match GPT-4 on DDI tasks after fine-tuning
- Multi-agent systems (DrugAgent, ICLR 2025) improve DTI F1 by ~45% via CoT reasoning
- LLMs fundamentally struggle with SMILES interpretation (cannot count rings, not robust to SMILES variants)
- **No papers benchmark LLMs on predicting or explaining compound INACTIVITY** — this is our gap

**Evaluation methodology (state of the art):**
- G-Eval (GPT-4 as judge): ~80% agreement with human raters
- Prometheus 2: Open-source judge, Pearson 0.6-0.7 with GPT-4
- STED (NeurIPS 2025): Semantic Tree Edit Distance for structured output evaluation
- HealthBench/LiveMedBench: Multi-dimensional rubric-based evaluation (gold standard)
- Anti-contamination: Rolling question refresh + temporal metadata essential

### 1.2 The Gap NegBioBench Fills

| Capability | Existing Benchmarks | NegBioBench |
|------------|-------------------|-------------|
| Predicting drug will NOT bind target | Absent | **Core task** |
| Extracting "no binding observed" from papers | Absent | **Core task** |
| Explaining WHY a drug doesn't bind | Absent | **Core task** |
| Distinguishing "tested negative" from "untested" | Absent | **Core task** |
| Assessing reliability of negative evidence | Absent | **Core task** |
| Reasoning about assay-context-dependent inactivity | Absent | **Core task** |

---

## 2. LLM Task Design (6 Tasks)

### Task L1: Negative DTI Classification (MCQ)

**Description:** Given a text description of a compound-target pair and experimental context, classify whether the interaction is positive, negative, or inconclusive.

**Format:**
```
Input: "Compound imatinib (a tyrosine kinase inhibitor) was tested against
        EGFR kinase in a biochemical binding assay at concentrations up to
        10 uM. The IC50 was determined to be >10 uM with no significant
        inhibition observed at any concentration tested."

Question: What is the interaction outcome?
A) Active (the compound inhibits the target)
B) Inactive (confirmed negative interaction)
C) Inconclusive (insufficient evidence)
D) Conditionally active (active only under specific conditions)

Answer: B
```

**Data Construction:**
- Source: NegBioDB entries converted to natural language descriptions
- Include positive examples from ChEMBL for balance
- Include ambiguous/conditional cases for difficulty
- ~2,000 examples (800 negative, 400 positive, 400 conditional, 400 inconclusive)

**Metrics:** Accuracy, Weighted F1, MCC (per-class and macro)

**Difficulty Levels:**
- Easy: Clear-cut active (IC50 < 100 nM) or inactive (no response at 100 uM)
- Medium: Borderline cases (IC50 = 5-15 uM around the 10 uM threshold)
- Hard: Conditional negatives (inactive in one assay, active in another)

**Cost:** $0 — MCQ format, automated scoring

---

### Task L2: Negative Result Extraction (Structured Output)

**Description:** Given a PubMed abstract or results section, extract all negative DTI findings into structured JSON format.

**Format:**
```
Input: "We screened a panel of 47 kinase inhibitors against CDK2 using a
        fluorescence polarization binding assay. Of these, 41 compounds
        showed no significant binding at 10 uM (IC50 > 10 uM), including
        sorafenib, lapatinib, and erlotinib. Six compounds showed moderate
        to strong binding, with palbociclib (IC50 = 5.2 nM) being the
        most potent."

Expected Output:
{
  "negative_results": [
    {
      "compound": "sorafenib",
      "target": "CDK2",
      "activity_type": "IC50",
      "activity_value": ">10000",
      "activity_unit": "nM",
      "assay_type": "fluorescence polarization binding",
      "assay_format": "biochemical",
      "max_concentration_tested": "10 uM",
      "outcome": "inactive"
    },
    {
      "compound": "lapatinib",
      "target": "CDK2",
      ...
    },
    {
      "compound": "erlotinib",
      "target": "CDK2",
      ...
    }
  ],
  "total_inactive_count": 41,
  "explicitly_named_inactive": 3,
  "positive_results_mentioned": true
}
```

**Data Construction:**
- Manually annotate 200 abstracts with gold-standard JSON extractions
- Include challenging cases: hedged language, implicit negatives, multi-target panels
- Include supplementary table descriptions for harder examples

**Metrics:**
- **Schema compliance rate**: Does the output parse as valid JSON with required fields?
- **Field-level F1**: Per-field precision/recall (compound name, target, activity value, etc.)
- **Entity-level F1**: Correct compound-target-outcome tuples extracted
- **STED score**: Semantic Tree Edit Distance vs. gold standard JSON

**Difficulty Levels:**
- Easy: Explicit "compound X did not inhibit target Y at concentration Z"
- Medium: Hedged language ("appeared to have minimal effect"), implicit negatives in panels
- Hard: Negation scope ambiguity, conditional negatives, supplementary table references

**Cost:** $0 — automated JSON comparison scoring

---

### Task L3: Inactivity Reasoning (Open-Ended)

**Description:** Given a confirmed negative DTI result, explain plausible biological/chemical reasons for the inactivity.

**Format:**
```
Input: "Sorafenib (a multi-kinase inhibitor approved for hepatocellular
        carcinoma) showed no binding to CDK2 (IC50 > 10 uM) in a biochemical
        binding assay. Sorafenib is known to primarily target RAF kinases,
        VEGFR, and PDGFR.

        Question: Provide a scientific explanation for why sorafenib is
        inactive against CDK2. Consider the structural and pharmacological
        properties of both the compound and the target."

Expected response should address:
- Binding site complementarity (RAF vs CDK2 ATP binding site differences)
- Kinase selectivity (sorafenib's Type II binding mode vs CDK2 structure)
- DFG-out conformation preference
- Key amino acid differences in the binding pocket
```

**Data Construction:**
- Select 200 well-characterized negative DTI pairs where structural rationale exists
- Create gold-standard reasoning rubric with required scientific concepts per example
- Include cases at varying difficulty (well-studied targets to Tdark targets)

**Evaluation (Multi-Dimensional Rubric):**

| Dimension | Weight | Scoring |
|-----------|--------|---------|
| **Scientific Accuracy** | 35% | Are stated facts correct? No hallucinated mechanisms? |
| **Reasoning Quality** | 30% | Logical chain from compound/target properties to inactivity? |
| **Completeness** | 20% | Key factors addressed (structure, selectivity, pharmacology)? |
| **Specificity** | 15% | Specific to this compound-target pair, not generic? |

**Scoring Method:**
- Phase 1: LLM-as-Judge (Gemini Flash free tier) with detailed rubric prompts
- Phase 2: Expert human evaluation on 30% sample to validate LLM judge
- Phase 3: Flagship LLM-as-Judge (GPT-4/Claude) for full evaluation

**Cost:** $0 in Phase 1 (Gemini free tier as judge); expert validation is manual labor

---

### Task L4: Tested-vs-Untested Discrimination

**Description:** Given a list of compound-target pairs, determine which have been experimentally tested (and found inactive) vs. which have simply never been tested.

**Format:**
```
Input: "For the following compound-target pairs, determine whether the
        inactivity is experimentally confirmed or assumed from lack of data.

        1. Aspirin — EGFR kinase
        2. Metformin — CDK4/6
        3. Imatinib — BRAF V600E
        4. Acetaminophen — mTOR
        5. Sorafenib — ALK"

Expected Output:
[
  {"pair": "Aspirin-EGFR", "status": "experimentally_tested",
   "evidence": "PubChem AID 1234, IC50 > 100 uM", "confidence": "high"},
  {"pair": "Metformin-CDK4/6", "status": "untested",
   "evidence": "No published binding data found", "confidence": "medium"},
  ...
]
```

**Data Construction:**
- 500 compound-target pairs: 50% experimentally confirmed inactive (from NegBioDB), 50% genuinely untested
- Include "trick" pairs where the compound is well-known but the specific target has never been tested
- Require evidence citation (assay ID, paper DOI, database reference)

**Metrics:**
- Binary classification: Accuracy, F1, MCC for tested/untested discrimination
- Evidence quality: Are cited references real and relevant?
- Calibration: Does stated confidence match actual accuracy?

**Why This Task Matters:**
- This is the fundamental problem in DTI prediction: conflating "untested" with "inactive"
- Tests whether LLMs can distinguish absence of evidence from evidence of absence
- Directly relevant to the core thesis of NegBioDB

**Cost:** $0 — automated scoring of classification + manual spot-check of evidence citations

---

### Task L5: Assay Context Reasoning (Conditional Negatives)

**Description:** Given a negative DTI result under specific assay conditions, predict whether the result might change under different conditions and explain why.

**Format:**
```
Input: "Compound X was tested against Target Y in a cell-based assay using
        HEK293 cells at pH 7.4 and found inactive (EC50 > 50 uM).

        However, Target Y is known to be a pH-sensitive enzyme with
        increased activity at acidic pH, and Compound X has been reported
        to have poor cell permeability.

        Question: Would you expect different results if:
        a) The assay was run at pH 5.5?
        b) A biochemical (cell-free) assay was used instead?
        c) The compound was tested at 100 uM instead of 50 uM?

        For each scenario, predict the likely outcome and explain your
        reasoning."

Expected: Scientifically grounded predictions with reasoning about:
- pH effect on target conformation and compound ionization
- Cell permeability vs. biochemical assay accessibility
- Concentration-dependent effects and the arbitrary nature of thresholds
```

**Data Construction:**
- 150 conditional negative examples from NegBioDB where assay context clearly matters
- Gold-standard annotations by domain experts on expected condition-change outcomes
- Include cases where: pH matters, cell type matters, concentration matters, species matters

**Evaluation Rubric:**

| Dimension | Weight | Scoring |
|-----------|--------|---------|
| **Prediction Accuracy** | 30% | Correct directional prediction per scenario? |
| **Scientific Reasoning** | 35% | Mechanistically sound explanation? |
| **Uncertainty Calibration** | 20% | Appropriately hedged when uncertain? |
| **Practical Insight** | 15% | Actionable experimental recommendations? |

**Cost:** $0 (LLM-as-Judge via Gemini free tier)

---

### Task L6: Negative Evidence Quality Assessment

**Description:** Given a negative DTI result with metadata, assess the reliability and confidence level of the evidence.

**Format:**
```
Input: "The following negative result was reported:
        - Compound: Dasatinib
        - Target: JAK2
        - Result: IC50 > 10 uM
        - Assay: Primary HTS screen, single-point at 10 uM
        - Cell line: N/A (biochemical)
        - Z-factor: 0.72
        - Replicates: 1
        - Source: PubChem AID 624297

        Assess the quality and reliability of this negative result.
        Assign a confidence tier (Gold/Silver/Bronze/Copper) and justify."

Expected Output:
{
  "confidence_tier": "Bronze",
  "justification": "Single-point primary screen without dose-response
    confirmation. While the Z-factor (0.72) indicates excellent assay
    quality, the lack of dose-response data and single replicate limit
    confidence. Dasatinib is a known multi-kinase inhibitor, and JAK2
    is a kinase target, so the negative result is biologically plausible
    but should be confirmed with dose-response testing.",
  "recommended_followup": [
    "Confirmatory dose-response assay (8-point, 3 replicates)",
    "Counter-screen for compound interference",
    "Test at higher concentrations (up to 100 uM)"
  ],
  "risk_factors": [
    "Single-point data only",
    "No replication",
    "Compound known to inhibit related kinases"
  ]
}
```

**Data Construction:**
- 300 negative results spanning all four confidence tiers with gold-standard annotations
- Include edge cases: high-quality assay but surprising result, low-quality assay confirming expected result
- Test whether LLMs can correctly identify the factors that determine evidence quality

**Metrics:**
- Tier classification accuracy (Gold/Silver/Bronze/Copper): Weighted F1
- Justification quality: LLM-as-Judge rubric score (accuracy, completeness, specificity)
- Follow-up recommendation relevance: Expert-validated scoring

**Cost:** $0 (structured output scoring + Gemini free tier for justification)

---

## 3. LLM Evaluation Infrastructure

### 3.1 Model Tiers for Evaluation

**Tier 1: Free / Local (MVP — always available)**

| Model | Access | Type | Purpose |
|-------|--------|------|---------|
| Gemini 2.5 Flash | Free tier (250 RPD) | API | Primary free evaluation model |
| Gemini 2.5 Flash-Lite | Free tier (1,000 RPD) | API | High-throughput MCQ evaluation |
| Llama 3.3 70B | Ollama local | Local | Open-source baseline (if RAM ≥ 32GB) |
| Llama 3.1 8B | Ollama local | Local | Small model baseline |
| Mistral 7B | Ollama local | Local | Small model baseline |
| Phi-3.5 3.8B | Ollama local | Local | Small model (DDI-competitive after fine-tune) |
| Qwen2.5 7B/72B | Ollama local | Local | Open-source alternative |

**Tier 2: Flagship (Post-Stabilization)**

| Model | Access | Purpose |
|-------|--------|---------|
| GPT-4.1 / GPT-4o | Paid API | Commercial flagship |
| Claude Sonnet/Opus | Paid API | Commercial flagship |
| Gemini Pro | Paid API | Commercial flagship |
| Med-PaLM 2 (if available) | Paid API | Medical specialist |

**Tier 3: Specialized Bio-LLMs (Research)**

| Model | Access | Purpose |
|-------|--------|---------|
| LlaSMol | HuggingFace | Chemistry-specialized |
| BioMedGPT | HuggingFace | Biomedical-specialized |
| Galactica | HuggingFace | Scientific knowledge |
| DrugChat | Custom | Drug property specialist |

### 3.2 LLM-as-Judge Configuration (Free)

For Tasks L3, L5, L6 (open-ended evaluation):

```python
# LLM-as-Judge using Gemini Free Tier
JUDGE_CONFIG = {
    "model": "gemini-2.5-flash",  # Free tier
    "temperature": 0,
    "max_tokens": 1024,
    "runs_per_judgment": 3,  # Majority vote for stability
    "rubric_format": "dimension_scores",  # Score each dimension separately
    "score_range": [1, 5],  # 5-point Likert scale
    "require_justification": True,  # Judge must explain scores
}

# Fallback: Local Llama 3.3 as judge (if Gemini rate-limited)
JUDGE_FALLBACK = {
    "model": "llama3.3:70b",  # Ollama local
    "temperature": 0,
}
```

**Validation Protocol:**
1. Human-annotate 50 examples per task as gold standard
2. Compute Cohen's kappa between LLM judge and human scores
3. Report kappa in paper (target: kappa > 0.6 = substantial agreement)
4. If kappa < 0.5, fall back to human evaluation for that task

### 3.3 Reproducibility Protocol

| Requirement | Implementation |
|-------------|---------------|
| Multiple runs | Each evaluation: 3 runs, report mean ± std |
| Temperature | temperature=0 for all evaluations |
| Prompt versioning | All prompts version-controlled in GitHub |
| Model versioning | Record exact model ID, date, API version |
| Seed parameter | Set where available (e.g., OpenAI seed parameter) |
| Cost reporting | Log tokens used per model per task |

### 3.4 Anti-Contamination Strategy

Since NegBioDB data is derived from public databases (PubChem, ChEMBL), LLMs may have seen the underlying data during training. Mitigation:

1. **Novel combinations:** Create test examples by pairing compounds and targets that have been individually present in training data but whose specific interaction (or non-interaction) requires reasoning beyond memorization
2. **Reasoning-required tasks:** Tasks L3, L5, L6 require explanation, not just recall — memorization alone is insufficient
3. **Temporal holdout:** Include post-2024 data that postdates most LLM training cutoffs
4. **Paraphrased variants:** Generate 3 semantically equivalent prompts per question; performance should be consistent across paraphrases
5. **Contamination detection:** Compare performance on pre-2023 vs. post-2024 entries; large gap suggests contamination

---

## 4. Dual-Track Benchmark Architecture

### 4.1 Complete NegBioBench Task Overview

```
NegBioBench
├── Track A: Traditional ML Tasks
│   ├── M1: DTI Binary Prediction
│   │     Input: (SMILES, protein sequence)
│   │     Output: Active / Inactive
│   │     Metrics: LogAUC, BEDROC, EF@1%, AUPRC, MCC, AUROC
│   │     Splits: Random, Cold-Compound, Cold-Target, Cold-Both,
│   │             Temporal, Scaffold, DDB
│   │
│   ├── M2: Negative Confidence Prediction
│   │     Input: (SMILES, protein sequence, assay features)
│   │     Output: Gold / Silver / Bronze / Copper
│   │     Metrics: Weighted F1, MCC
│   │
│   └── M3: Activity Value Regression
│         Input: (SMILES, protein sequence)
│         Output: pIC50 / pKd value
│         Metrics: RMSE, R², Spearman ρ
│
├── Track B: LLM Tasks
│   ├── L1: Negative DTI Classification (MCQ)
│   │     Input: Natural language description
│   │     Output: Active / Inactive / Inconclusive / Conditional
│   │     Metrics: Accuracy, Weighted F1, MCC
│   │     Evaluation: Automated
│   │
│   ├── L2: Negative Result Extraction (Structured Output)
│   │     Input: Paper abstract / results section
│   │     Output: Structured JSON
│   │     Metrics: Schema compliance, Field F1, Entity F1, STED
│   │     Evaluation: Automated
│   │
│   ├── L3: Inactivity Reasoning (Open-Ended)
│   │     Input: Confirmed negative DTI + context
│   │     Output: Scientific explanation
│   │     Metrics: Accuracy, Reasoning, Completeness, Specificity
│   │     Evaluation: LLM-as-Judge (Gemini free) + human sample
│   │
│   ├── L4: Tested-vs-Untested Discrimination
│   │     Input: List of compound-target pairs
│   │     Output: Tested/Untested classification + evidence
│   │     Metrics: Accuracy, F1, evidence quality
│   │     Evaluation: Automated + evidence spot-check
│   │
│   ├── L5: Assay Context Reasoning (Conditional Negatives)
│   │     Input: Negative result + condition change scenarios
│   │     Output: Prediction + reasoning per scenario
│   │     Metrics: Prediction accuracy, reasoning quality
│   │     Evaluation: LLM-as-Judge + expert sample
│   │
│   └── L6: Negative Evidence Quality Assessment
│         Input: Negative result + assay metadata
│         Output: Confidence tier + justification + recommendations
│         Metrics: Tier accuracy (F1), justification quality
│         Evaluation: Automated tier scoring + LLM-judge for text
│
└── Track C: Cross-Track Tasks (ML + LLM)
    └── C1: Ensemble Prediction
          Combine ML model scores with LLM reasoning
          Metrics: Does LLM reasoning improve ML predictions?
```

### 4.2 NeurIPS Submission Prioritization

**Must have for NeurIPS (11-week sprint):**
- Track A: M1 (DTI Binary Prediction) — core ML task
- Track B: L1 (Classification), L2 (Extraction) — automated evaluation
- Track B: L4 (Tested-vs-Untested) — unique selling point, automated eval
- 2-3 free LLM baselines (Gemini Flash, Llama 3.3, Mistral 7B)
- 3+ ML baselines (DeepDTA, GraphDTA, DrugBAN)

**Should have (strengthens paper):**
- Track A: M2 (Confidence Prediction)
- Track B: L3 (Reasoning) with LLM-as-Judge validation
- Track B: L6 (Evidence Assessment)
- More LLM baselines (Phi-3.5, Qwen2.5)

**Nice to have (defer to camera-ready/v2):**
- Track A: M3 (Regression)
- Track B: L5 (Conditional reasoning)
- Track C: C1 (Ensemble)
- Flagship model evaluations (GPT-4, Claude)
- Specialized bio-LLM evaluations (LlaSMol, DrugChat)

### 4.3 Data Size Requirements

| Task | Train | Validation | Test | Total | Construction Method |
|------|-------|-----------|------|-------|-------------------|
| **M1** | 70% of DB | 10% | 20% | Full NegBioDB | Automated from DB |
| **M2** | 70% | 10% | 20% | Full NegBioDB | Automated from DB |
| **M3** | 70% | 10% | 20% | Subset with quantitative values | Automated from DB |
| **L1** | 200 (few-shot examples) | 200 | 1,600 | 2,000 | Semi-automated + human review |
| **L2** | 20 (few-shot examples) | 30 | 150 | 200 | Manual annotation (abstracts) |
| **L3** | 10 (few-shot examples) | 20 | 170 | 200 | Expert-created rubrics |
| **L4** | 50 (few-shot examples) | 50 | 400 | 500 | Semi-automated + validation |
| **L5** | 10 (few-shot examples) | 15 | 125 | 150 | Expert-created scenarios |
| **L6** | 20 (few-shot examples) | 30 | 250 | 300 | Automated from DB + expert rubrics |

**Note:** LLM tasks use few-shot examples (not training sets) since LLMs are evaluated zero-shot or few-shot, not fine-tuned (unless specifically studying fine-tuning effects).

---

## 5. Evaluation Metrics Deep Dive

### 5.1 Automated Metrics (Tasks L1, L2, L4)

```python
# Task L1: MCQ Classification
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

metrics_l1 = {
    "accuracy": accuracy_score(y_true, y_pred),
    "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
    "macro_f1": f1_score(y_true, y_pred, average='macro'),
    "mcc": matthews_corrcoef(y_true, y_pred),
    "per_class_f1": f1_score(y_true, y_pred, average=None),
}

# Task L2: Structured Output
import json

def evaluate_extraction(predicted_json: str, gold_json: dict) -> dict:
    # 1. Schema compliance
    try:
        pred = json.loads(predicted_json)
        schema_valid = validate_schema(pred)  # Pydantic validation
    except:
        return {"schema_compliance": 0, "field_f1": 0, "entity_f1": 0}

    # 2. Field-level F1
    field_scores = []
    for field in REQUIRED_FIELDS:
        field_scores.append(fuzzy_match(pred.get(field), gold.get(field)))

    # 3. Entity-level F1 (compound-target-outcome tuples)
    pred_tuples = extract_tuples(pred)
    gold_tuples = extract_tuples(gold)
    entity_f1 = compute_f1(pred_tuples, gold_tuples)

    return {
        "schema_compliance": 1.0 if schema_valid else 0.0,
        "field_f1": np.mean(field_scores),
        "entity_f1": entity_f1,
    }

# Task L4: Classification + Evidence Quality
metrics_l4 = {
    "classification_accuracy": accuracy_score(y_true, y_pred),
    "classification_f1": f1_score(y_true, y_pred),
    "classification_mcc": matthews_corrcoef(y_true, y_pred),
    "evidence_citation_rate": n_cited / n_total,  # Did it cite evidence?
    # Evidence validity checked by human spot-check on 20% sample
}
```

### 5.2 LLM-as-Judge Rubric (Tasks L3, L5, L6)

```python
JUDGE_PROMPT_TEMPLATE = """
You are evaluating an LLM's response to a scientific reasoning task about
drug-target interaction inactivity.

## Task Description
{task_description}

## Input Given to the Model
{input_text}

## Model's Response
{model_response}

## Gold Standard Key Points
{gold_standard_points}

## Evaluation Rubric
Score each dimension on a 1-5 scale:

### Scientific Accuracy (1-5)
1 = Major factual errors or hallucinated mechanisms
2 = Some factual errors that affect the conclusion
3 = Mostly accurate with minor errors
4 = Accurate with only trivial imprecisions
5 = Fully accurate, all stated facts verifiable

### Reasoning Quality (1-5)
1 = No logical connection between evidence and conclusion
2 = Weak reasoning with major logical gaps
3 = Adequate reasoning but missing key steps
4 = Strong reasoning with clear logical chain
5 = Excellent reasoning, all steps clearly connected

### Completeness (1-5)
1 = Misses most key factors
2 = Addresses only 1-2 of the key factors
3 = Addresses roughly half of the key factors
4 = Addresses most key factors
5 = Addresses all key factors comprehensively

### Specificity (1-5)
1 = Entirely generic (could apply to any compound-target pair)
2 = Mostly generic with some specific details
3 = Mix of generic and specific reasoning
4 = Mostly specific to this compound-target pair
5 = Highly specific, demonstrates deep understanding of this pair

## Output Format
Provide your scores as JSON:
{
  "scientific_accuracy": <1-5>,
  "reasoning_quality": <1-5>,
  "completeness": <1-5>,
  "specificity": <1-5>,
  "justification": "<brief explanation of scores>"
}
"""
```

### 5.3 Domain-Adapted BERTScore (Supplementary Metric)

For open-ended tasks, use PubMedBERT or SciBERT as the backbone for BERTScore instead of generic BERT:

```python
from bert_score import score

# Use domain-adapted model for biomedical text
P, R, F1 = score(
    predictions,
    references,
    model_type="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    lang="en",
)
```

---

## 6. Prompt Templates

### 6.1 Standard Prompt Structure

All LLM tasks follow a consistent template:

```
[System prompt — sets the domain context]
You are a pharmaceutical scientist with expertise in drug-target
interactions, assay development, and medicinal chemistry.

[Task instruction — specific to each task]
{task_specific_instruction}

[Input data]
{input_data}

[Output format specification]
Respond in the following format:
{output_format}

[Few-shot examples (if applicable)]
Example 1:
Input: {example_input}
Output: {example_output}
```

### 6.2 Evaluation Configurations

| Configuration | Description | Purpose |
|---------------|-------------|---------|
| Zero-shot | No examples provided | Baseline capability |
| 3-shot | 3 diverse examples | Standard few-shot |
| 5-shot | 5 examples including edge cases | Enhanced few-shot |
| CoT (Chain-of-Thought) | "Think step by step" added | Reasoning assessment |
| CoT + 3-shot | Few-shot with CoT examples | Best expected performance |

**Report all configurations** for each model to capture the effect of prompting strategy.

---

## 7. Comparison with Existing Benchmarks

### 7.1 NegBioBench LLM Track vs. Related Benchmarks

| Feature | ChemBench | Mol-Inst. | MedQA | LAB-Bench | **NegBioBench** |
|---------|-----------|-----------|-------|-----------|----------------|
| Negative result tasks | No | No | No | No | **Yes (all 6 tasks)** |
| Structured extraction | No | Limited | No | No | **Yes (L2)** |
| Scientific reasoning | Partial | No | No | Partial | **Yes (L3, L5)** |
| Evidence assessment | No | No | No | No | **Yes (L6)** |
| DTI-specific | No | Partial | No | No | **Yes** |
| Free to evaluate | Yes | Yes | Yes | Yes | **Yes** |
| ML + LLM dual track | No | No | No | No | **Yes** |
| Confidence tiers | No | No | No | No | **Yes** |
| Anti-contamination | No | No | Partial | No | **Yes** |

### 7.2 Novelty Claims for Paper

1. **First benchmark for negative DTI reasoning** — no existing benchmark tests any of these capabilities
2. **Dual ML + LLM track** — bridges cheminformatics and NLP/LLM communities in a single benchmark
3. **Tested-vs-Untested discrimination** (L4) — directly addresses the "untested = negative" assumption that plagues all DTI prediction
4. **Evidence quality assessment** (L6) — teaches LLMs to evaluate scientific evidence reliability
5. **Free-to-evaluate design** — all tasks evaluable with free/open-source models, lowering barrier to participation

---

## 8. Implementation Timeline

### For NeurIPS Sprint (Weeks 1-11)

| Week | LLM Benchmark Tasks |
|------|-------------------|
| 1-2 | Design prompt templates for L1, L2, L4 |
| 2-3 | Construct L1 dataset (2,000 MCQ from NegBioDB entries) |
| 3-4 | Construct L2 dataset (annotate 200 abstracts) |
| 3-4 | Construct L4 dataset (500 tested/untested pairs) |
| 4-5 | Implement automated evaluation scripts (L1, L2, L4) |
| 5-6 | Set up LLM-as-Judge pipeline (Gemini free tier) |
| 5-7 | Run baselines: Gemini Flash, Llama 3.3, Mistral 7B on L1, L2, L4 |
| 7-8 | Construct L3 dataset (200 reasoning examples) if time permits |
| 7-8 | Run L3 baselines with LLM-as-Judge evaluation |
| 8-10 | Analyze results, create figures, write paper sections |

### Post-Submission Expansion

- Complete L5, L6 datasets
- Add flagship model evaluations (GPT-4, Claude)
- Add specialized bio-LLM evaluations (LlaSMol, BioMedGPT)
- Build leaderboard with separate ML and LLM tracks
- Human validation study on LLM-as-Judge reliability

---

## Sources

### Existing Benchmarks
- ChemBench: https://www.nature.com/articles/s41557-025-01815-x
- Mol-Instructions (ICLR 2024): https://github.com/zjunlp/Mol-Instructions
- LlaSMol (COLM 2024): https://arxiv.org/html/2402.09391v4
- SciBench (ICML 2024): https://arxiv.org/abs/2307.10635
- SciKnowEval: https://arxiv.org/html/2406.09098v1
- LAB-Bench: https://github.com/Future-House/LAB-Bench
- HealthBench (OpenAI 2025): https://openai.com/index/healthbench/
- LiveMedBench: https://arxiv.org/abs/2602.10367

### LLMs for Drug Discovery
- DrugAgent (ICLR 2025): https://arxiv.org/abs/2408.13378
- DrugChat: https://www.biorxiv.org/content/10.1101/2024.09.29.615524v1
- MolecularGPT: https://arxiv.org/abs/2406.12950
- MolRAG (ACL 2025): https://aclanthology.org/2025.acl-long.755/
- TwinBooster: https://arxiv.org/abs/2401.04478
- LLMDTA (ISBRA 2024): https://link.springer.com/chapter/10.1007/978-981-97-5131-0_14
- LLMs for DDI comparison: https://arxiv.org/abs/2502.06890
- "Can LLMs Understand Molecules?": https://link.springer.com/article/10.1186/s12859-024-05847-x

### Evaluation Methods
- G-Eval: https://arxiv.org/abs/2303.16634
- FActScore: https://arxiv.org/abs/2305.14251
- STED (NeurIPS 2025): https://arxiv.org/abs/2512.23712
- Prometheus 2: https://github.com/prometheus-eval/prometheus-eval
- StructEval: https://tiger-ai-lab.github.io/StructEval/
- LLM-as-Judge survey: https://arxiv.org/html/2411.15594v2
- MT-Bench: https://arxiv.org/abs/2306.05685

### Contamination & Reproducibility
- LiveBench: https://github.com/LiveBench/LiveBench
- AntiLeakBench (ACL 2025): https://aclanthology.org/2025.acl-long.901/
- LLM Non-Determinism: https://arxiv.org/html/2408.04667v5
- Reproducible Evaluation: https://arxiv.org/html/2410.03492

### Prompt Engineering for Chemistry
- Domain-knowledge prompting: https://pmc.ncbi.nlm.nih.gov/articles/PMC11350497/
- Chemistry prompt engineering (ACS 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12022906/
- SMILES robustness: https://pmc.ncbi.nlm.nih.gov/articles/PMC12574305/
- CLEANMOL (EMNLP 2025): https://arxiv.org/abs/2505.16340
