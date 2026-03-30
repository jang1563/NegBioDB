# Appendix A: LLM Benchmark Prompt Templates

This appendix documents all prompt templates used in the NegBioDB LLM benchmark (tasks L1--L4). Templates are reproduced verbatim from `src/negbiodb/llm_prompts.py` and `src/negbiodb/llm_eval.py`.

## A.1 System Prompt (Shared Across All Tasks)

```
You are a pharmaceutical scientist with expertise in drug-target interactions, assay development, and medicinal chemistry. Provide precise, evidence-based answers.
```

## A.2 L1: Activity Classification (Multiple Choice)

### A.2.1 Zero-Shot Template

```
{context}
```

### A.2.2 Few-Shot Template

```
Here are some examples of drug-target interaction classification:

{examples}

Now classify the following:

{context}
```

### A.2.3 Answer Format Instruction

```
Respond with ONLY the letter of the correct answer: A, B, C, or D.
```

The answer format instruction is appended after both zero-shot and few-shot templates.

## A.3 L2: Structured Extraction

### A.3.1 Zero-Shot Template

```
Extract all negative drug-target interaction results from the following abstract.

Abstract:
{abstract_text}

For each negative result found, extract:
- compound: compound/drug name
- target: target protein/gene name
- target_uniprot: UniProt accession (if determinable)
- activity_type: type of measurement (IC50, Ki, Kd, EC50, etc.)
- activity_value: reported value with units
- activity_relation: relation (=, >, <, ~)
- assay_format: biochemical, cell-based, or in vivo
- outcome: inactive, weak, or inconclusive

Also report:
- total_inactive_count: total number of inactive results mentioned
- positive_results_mentioned: true/false

Respond in JSON format.
```

### A.3.2 Few-Shot Template

```
Extract negative drug-target interaction results from abstracts.

{examples}

Now extract from this abstract:

Abstract:
{abstract_text}

Respond in JSON format.
```

Few-shot examples include the abstract text and the corresponding gold extraction in JSON format, separated by `---` delimiters.

## A.4 L3: Scientific Reasoning

### A.4.1 Zero-Shot Template

```
{context}

Provide a detailed scientific explanation (3-5 paragraphs) covering:
1. Structural compatibility between compound and target binding site
2. Known selectivity profile and mechanism of action
3. Relevant SAR (structure-activity relationship) data
4. Pharmacological context and therapeutic implications
```

### A.4.2 Few-Shot Template

```
Here are examples of scientific reasoning about inactive drug-target interactions:

{examples}

Now explain the following:

{context}
```

### A.4.3 LLM-as-Judge Rubric

Responses are evaluated by a judge model (Gemini 2.5 Flash) using the following rubric:

```
Rate the following scientific explanation of why a compound is inactive against a target.

Compound: {compound_name}
Target: {target_gene} ({target_uniprot})

Explanation to evaluate:
{response}

Rate on these 4 dimensions (1-5 each):
1. Accuracy: Are the scientific claims factually correct?
2. Reasoning: Is the logical chain from structure to inactivity sound?
3. Completeness: Are all relevant factors considered (binding, selectivity, SAR)?
4. Specificity: Does the explanation use specific molecular details, not generalities?

Respond in JSON: {{"accuracy": X, "reasoning": X, "completeness": X, "specificity": X}}
```

The judge returns scores as JSON with four dimensions (accuracy, reasoning, completeness, specificity), each rated 1--5.

## A.5 L4: Tested vs Untested Discrimination

### A.5.1 Zero-Shot Template

```
{context}
```

### A.5.2 Few-Shot Template

```
Here are examples of tested/untested compound-target pair determination:

{examples}

Now determine:

{context}
```

### A.5.3 Answer Format Instruction

```
Respond with 'tested' or 'untested' on the first line. If tested, provide the evidence source (database, assay ID, or DOI) on the next line.
```

The answer format instruction is appended after both zero-shot and few-shot templates.

## A.6 Model Configuration

| Parameter | Value |
|-----------|-------|
| Temperature | 0.0 (deterministic) |
| Max output tokens | 1024 (L1/L4), 2048 (L2/L3) |
| Few-shot sets | 3 independent sets (fs0, fs1, fs2) |
| Retry policy | Exponential backoff, max 8 retries |

### Models

| Model | Provider | Inference |
|-------|----------|-----------|
| Claude Haiku-4.5 | Anthropic API | Cloud |
| Gemini 2.5 Flash | Google Gemini API | Cloud |
| GPT-4o-mini | OpenAI API | Cloud |
| Qwen2.5-7B-Instruct | vLLM | Local (A100 GPU) |
| Llama-3.1-8B-Instruct | vLLM | Local (A100 GPU) |

Gemini 2.5 Flash uses `thinkingConfig: {thinkingBudget: 0}` to disable internal reasoning tokens and ensure the full output budget is available for the response.
