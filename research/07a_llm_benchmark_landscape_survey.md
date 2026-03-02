# LLM Benchmark Landscape: Biomedical, Chemistry, and Drug Discovery Domains

> Comprehensive survey of existing benchmarks, evaluation methods, and gaps relevant to building an LLM benchmark around negative drug-target interaction data (2024-2026)

---

## 1. Existing Biomedical/Chemistry LLM Benchmarks

### 1.1 ChemBench (Lamalab, Nature Chemistry 2025)

**Paper:** "A framework for evaluating the chemical knowledge and reasoning abilities of large language models against the expertise of chemists"
**URL:** https://www.nature.com/articles/s41557-025-01815-x | https://github.com/lamalab-org/chembench

| Attribute | Detail |
|-----------|--------|
| **Tasks** | Knowledge recall, chemical intuition, multi-step reasoning across undergraduate and postgraduate chemistry |
| **Scale** | 2,700+ question-answer pairs; 6,202 MCQ + 857 open-ended questions |
| **Input/Output** | Text-based questions -> MCQ selection or free-form answers |
| **Scoring** | Accuracy for MCQ; automated + human evaluation for open-ended; comparison against human chemist performance |
| **Key Finding** | Best LLMs outperform best human chemists on average, but struggle with basic tasks and are overconfident |
| **Limitations** | Primarily knowledge-oriented (not task-oriented like synthesis planning); does not evaluate negative/null result interpretation; no drug-target interaction tasks |

**Relevance to NegBioDB:** ChemBench evaluates chemical reasoning but not the ability to interpret failed experiments or negative binding data. A benchmark testing whether LLMs can correctly identify and reason about non-interactions would fill a clear gap.

---

### 1.2 ChemLLMBench (NeurIPS 2023 Datasets & Benchmarks)

**Paper:** "What can Large Language Models do in chemistry? A comprehensive benchmark on eight tasks"
**URL:** https://github.com/ChemFoundationModels/ChemLLMBench | https://proceedings.neurips.cc/paper_files/paper/2023/hash/bbb330189ce02be00cf7346167028ab1-Abstract-Datasets_and_Benchmarks.html

| Attribute | Detail |
|-----------|--------|
| **Eight Tasks** | (1) Name prediction, (2) Property prediction, (3) Yield prediction, (4) Reaction prediction, (5) Retrosynthesis, (6) Text-based molecule design, (7) Molecule captioning, (8) Reagent selection |
| **Models Tested** | GPT-4, GPT-3.5, Davinci-003, Llama-2, Galactica |
| **Input/Output** | SMILES strings, natural language descriptions -> SMILES, text, numerical values |
| **Scoring** | Task-specific: exact match, BLEU, validity of generated SMILES, Tanimoto similarity, numerical accuracy |
| **Evaluation Settings** | Zero-shot, few-shot in-context learning |
| **Limitations** | No negative result tasks; no drug-target interaction prediction; no evaluation of reasoning about why a reaction fails or why a compound does not bind |

**Relevance to NegBioDB:** The property prediction task could be extended to include "predicting non-binding" as an explicit task. Currently, all tasks assume positive outcomes (predict what DOES happen, not what does NOT).

---

### 1.3 Mol-Instructions (ICLR 2024)

**Paper:** "Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models"
**URL:** https://github.com/zjunlp/Mol-Instructions | https://openreview.net/forum?id=Tlsdsb6l9n

| Attribute | Detail |
|-----------|--------|
| **Three Components** | (1) Molecule-oriented: 148.4K instructions across 6 tasks, (2) Protein-oriented: 505K instructions across 5 task categories, (3) Biomolecular text: 53K instructions |
| **Molecule Tasks** | Property prediction, molecule description, molecule design from description, forward reaction prediction, retrosynthesis, reagent prediction |
| **Input/Output** | SMILES + natural language -> SMILES, text descriptions, property values |
| **Scoring** | Task-specific metrics (exact match, BLEU, validity, fingerprint similarity) |
| **Limitations** | Instruction-tuning dataset, not a standalone evaluation benchmark; no negative interaction data; all tasks are positive-outcome oriented |

**Relevance to NegBioDB:** The instruction-tuning paradigm could be adapted: "Given this drug and target, the following experiment was conducted and NO binding was observed. Explain why." This task format is entirely absent from current instruction datasets.

---

### 1.4 ScienceQA (Lu et al., NeurIPS 2022)

**URL:** https://scienceqa.github.io/

| Attribute | Detail |
|-----------|--------|
| **Scale** | 21,208 multiple-choice questions from US grade-school science curricula (grades 1-12) |
| **Modalities** | Text-only, image+text, or text passage+question |
| **Annotation** | 83.9% have grounded lectures; 90.5% have detailed explanations |
| **Scoring** | Accuracy on MCQ; BLEU-n, ROUGE-L, Sentence-BERT for generated explanations |
| **SOTA** | Multimodal-CoT (T5-Large + ViT): 90.45%; Chameleon + GPT-4: 86.54% |
| **Limitations** | K-12 level, not research-grade science; no drug discovery or negative result content |

**Relevance to NegBioDB:** Minimal direct relevance due to K-12 scope, but the annotation methodology (lectures + explanations) provides a template for how reasoning chains should be annotated in a negative-results benchmark.

---

### 1.5 BioASQ (Ongoing since 2013, Task 13b in 2025)

**URL:** http://bioasq.org/ | https://ceur-ws.org/Vol-4038/paper_10.pdf

| Attribute | Detail |
|-----------|--------|
| **Tasks** | Biomedical semantic indexing (Task A), question answering from biomedical literature (Task B) |
| **QA Types** | Yes/No, Factoid, List, Summary questions |
| **Input/Output** | Natural language questions + PubMed context -> answers in various formats |
| **Scoring** | F1 for yes/no; strict/lenient accuracy for factoid; F1 for list; ROUGE for summary |
| **Scale** | Continuously updated; thousands of expert-curated QA pairs |
| **SOTA (2025)** | MedBioLM: 96% accuracy on BioASQ tasks |
| **Limitations** | General biomedical knowledge; no drug-target interaction specificity; no explicit negative result handling |

---

### 1.6 MedQA (USMLE-style)

**URL:** https://www.vals.ai/benchmarks/medqa

| Attribute | Detail |
|-----------|--------|
| **Tasks** | Medical reasoning via USMLE-style multiple-choice questions |
| **Scale** | ~12,000+ questions across Steps 1-3 |
| **Scoring** | Accuracy on MCQ |
| **SOTA (2025)** | GPT-5: 95.84%; Med-Gemini: 91.1% |
| **Key Critique (ACL 2025)** | MedQA benchmark scores show only moderate correlation with clinical performance (Spearman's rho = 0.59); fails to capture patient communication, longitudinal care, and clinical information extraction. Performance drops dramatically (below 1/10th) when same questions are embedded in interactive diagnostic scenarios. |
| **Limitations** | Clinical medicine focus, not drug discovery; factual recall bias; no negative result interpretation |

**Paper on limitations:** "Questioning Our Questions: How Well Do Medical QA Benchmarks Evaluate Clinical Capabilities of Language Models?" - https://aclanthology.org/2025.bionlp-1.24/

---

### 1.7 Rx-LLM (Medication Benchmark, 2025)

**Paper:** "Rx-LLM: a benchmarking suite to evaluate safe LLM performance for medication-related tasks"
**URL:** https://www.medrxiv.org/content/10.64898/2025.12.01.25341004v2

| Attribute | Detail |
|-----------|--------|
| **Six Tasks** | (1) Drug formulation matching, (2) Drug order (sig) generation, (3) Drug route matching, (4) Drug-drug interaction identification, (5) Renal dose identification, (6) Drug-indication matching |
| **Scale** | 250 standardized input-output pairs per task |
| **Scoring** | F1 score and accuracy per task |
| **Best Models** | LLaMA3-70B best on 4/6 tasks (drug-formulation F1: 54.0%, drug-indication accuracy: 97.6%); GPT-4o-mini best on DDI (accuracy: 70.4%) |
| **Limitations** | Clinical pharmacy focus; does not evaluate DTI prediction or negative experimental outcomes |

**Relevance to NegBioDB:** The drug-drug interaction identification task is closest to our domain. However, it evaluates known DDIs, not experimentally confirmed non-interactions.

---

### 1.8 DrugChat (bioRxiv 2024, Multi-modal Drug LLM)

**Paper:** "Multi-Modal Large Language Model Enables All-Purpose Prediction of Drug Mechanisms and Properties"
**URL:** https://www.biorxiv.org/content/10.1101/2024.09.29.615524v1

| Attribute | Detail |
|-----------|--------|
| **Architecture** | Graph Neural Network (molecular graph) + LLM (text) multimodal system |
| **Training Data** | 10,834 drug compounds, 143,517 QA pairs; 14,000+ molecules, 91,365 molecule-prompt-answer triplets |
| **Tasks** | Drug indications, pharmacodynamics, mechanism of action, structure-activity relationships |
| **Input/Output** | Molecular graph + natural language query -> free-form text predictions |
| **Scoring** | Expert human evaluation + automated metrics |
| **Key Result** | 25% higher mechanism accuracy than GPT-4 |
| **Limitations** | Free-form generation makes systematic evaluation difficult; "molecular hallucinations" are a major concern; no negative interaction prediction tasks |

---

### 1.9 GPQA (Graduate-Level Google-Proof Q&A)

**Paper:** "GPQA: A Graduate-Level Google-Proof Q&A Benchmark"
**URL:** https://arxiv.org/abs/2311.12022 | https://github.com/idavidrein/gpqa

| Attribute | Detail |
|-----------|--------|
| **Scale** | 448 MCQ questions; Diamond subset: 198 questions (highest quality) |
| **Domains** | Biology, physics, chemistry |
| **Design** | Written by PhD-level domain experts; designed to be unanswerable by non-experts even with internet access |
| **Scoring** | Accuracy on MCQ |
| **Human Baselines** | Domain experts: 65% (74% corrected); Non-expert validators: 34% (with 30+ min internet) |
| **Limitations** | Graduate-level knowledge, but still MCQ format; no evaluation of experimental design or negative result interpretation |

---

### 1.10 MMLU-Pro (NeurIPS 2024)

**Paper:** "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark"
**URL:** https://arxiv.org/abs/2406.01574 | https://github.com/TIGER-AI-Lab/MMLU-Pro

| Attribute | Detail |
|-----------|--------|
| **Scale** | 12,000+ questions across 14 domains including Biology, Chemistry, Physics |
| **Format** | MCQ with 10 answer choices (vs. 4 in original MMLU) |
| **Sources** | Original MMLU questions + STEM websites + TheoremQA + SciBench |
| **Key Feature** | CoT reasoning significantly improves performance (unlike original MMLU), indicating more complex reasoning questions |
| **Scoring** | Accuracy; CoT vs. direct answer comparison |
| **Limitations** | Broad coverage but shallow depth in any single scientific domain; no drug discovery or negative result tasks |

---

### 1.11 SciBench (ICML 2024)

**Paper:** "SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models"
**URL:** https://arxiv.org/abs/2307.10635 | https://github.com/mandyyyyii/scibench

| Attribute | Detail |
|-----------|--------|
| **Tasks** | Open-ended, free-response scientific problems requiring multi-step reasoning + complex arithmetic |
| **Domains** | Mathematics, chemistry, physics (college-level textbooks + exams) |
| **Scoring** | Correctness of final numerical answers; error attribution via LLM verifier categorizing failures into 10 problem-solving abilities |
| **Multimodal** | 94 problems with graphs/figures |
| **SOTA** | Best overall: 43.22% — showing these are genuinely difficult |
| **Limitations** | Textbook problems, not research problems; no drug discovery content; no negative result interpretation |

---

### 1.12 SciEval (AAAI 2024)

**Paper:** "SciEval: A Multi-Level Large Language Model Evaluation Benchmark for Scientific Research"
**URL:** https://arxiv.org/abs/2308.13149 | https://github.com/OpenDFM/SciEval

| Attribute | Detail |
|-----------|--------|
| **Scale** | ~18,000 objective + subjective evaluation questions |
| **Domains** | Chemistry, physics, biology |
| **Four Dimensions** | (1) Basic knowledge, (2) Knowledge application, (3) Scientific calculation, (4) Research ability |
| **Taxonomy** | Based on Bloom's taxonomy |
| **Anti-Leakage** | Includes "dynamic" subset generated from scientific principles to prevent data contamination |
| **Scoring** | Accuracy for objective; rubric-based for subjective |
| **Limitations** | Broad coverage but no drug-target interaction tasks; no negative result evaluation |

---

### 1.13 SciKnowEval (2024, Updated V2 July 2025)

**URL:** https://arxiv.org/html/2406.09098v1 | https://github.com/HICAI-ZJU/SciKnowEval

| Attribute | Detail |
|-----------|--------|
| **Scale** | V1: 70K problems; V2: 28,392 samples across 58 tasks |
| **Domains** | Biology, chemistry, physics, materials science |
| **Five Progressive Levels** | L1: Memory/retrieval, L2: Deep inquiry/analysis, L3: Critical thinking/reasoning, L4: Discernment, L5: Application |
| **Key Finding** | GPT-4o >85% on L1/L2 but struggles on L3/L5, especially tasks involving SMILES and protein sequences |
| **Limitations** | Broad scientific knowledge, not focused on drug discovery or negative results |

---

### 1.14 LLaSMol / SMolInstruct (COLM 2024)

**Paper:** "LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset"
**URL:** https://arxiv.org/html/2402.09391v3 | https://osu-nlp-group.github.io/LLM4Chem/

| Attribute | Detail |
|-----------|--------|
| **Dataset** | SMolInstruct: 14 chemistry tasks, 3M+ samples |
| **Models** | LlaSMol (Galactica, Llama-2, Code Llama, Mistral bases) |
| **Key Result** | LlaSMol-Mistral: 93.2% EM on SMILES->Formula vs. GPT-4's 4.8% and Claude 3 Opus's 9.2% |
| **Tasks** | Name conversion, property prediction, molecule generation, reaction prediction, retrosynthesis |
| **Limitations** | Instruction-tuning focused; no negative data evaluation; all tasks oriented toward predicting positive outcomes |

---

### 1.15 LAB-Bench (Future House, 2024)

**URL:** https://github.com/Future-House/LAB-Bench

| Attribute | Detail |
|-----------|--------|
| **Scale** | 2,400+ MCQ questions |
| **Focus** | Practical biology research capabilities |
| **Tasks** | Literature recall/reasoning, figure interpretation, database navigation, DNA/protein sequence manipulation |
| **Key Finding** | Frontier LLMs have emergent capabilities in table interpretation but lag human experts in complex reasoning and tool use |
| **Limitations** | Biology research general; no drug discovery-specific tasks; no negative result tasks |

---

### 1.16 ChemSafetyBench (2024)

**Paper:** "ChemSafetyBench: Benchmarking LLM Safety on Chemistry Domain"
**URL:** https://arxiv.org/html/2411.16736v1

| Attribute | Detail |
|-----------|--------|
| **Focus** | Safety evaluation of LLMs in chemistry (preventing harmful chemical synthesis instructions) |
| **Relevance** | Tangential -- evaluates what LLMs should NOT do rather than negative scientific results |

---

## 2. Drug Discovery-Specific LLM Evaluation

### 2.1 LLMs for Drug-Drug Interaction Prediction (2025)

**Paper:** "LLMs for Drug-Drug Interaction Prediction: A Comprehensive Comparison"
**URL:** https://arxiv.org/abs/2502.06890

- Evaluated **18 LLMs** including GPT-4, Claude, Gemini, Llama-3, DeepSeek V3
- Fine-tuned Phi-3.5 2.7B achieved sensitivity 0.978, accuracy 0.919 on balanced datasets (50% positive/50% negative)
- **Key finding:** LLM-based methods are the most robust against distribution changes in DDI prediction
- **Negative data:** Generated from comprehensive databases; 50/50 positive-negative balance used
- **Limitation:** Balanced datasets are unrealistic; real negative rates are much higher

### 2.2 LLM3-DTI: Drug-Target Interaction with LLMs (2025)

**Paper:** "LLM3-DTI: A Large Language Model and Multi-modal data co-powered framework for Drug-Target Interaction prediction"
**URL:** https://arxiv.org/html/2511.06269

- Multi-modal framework combining LLM text understanding with molecular/protein structural data
- Represents emerging trend of using LLMs as components in DTI prediction pipelines

### 2.3 Drug Mechanism Understanding Benchmark (2025)

**Paper:** "How Well Do LLMs Understand Drug Mechanisms? A Knowledge + Reasoning Evaluation Dataset"
**URL:** https://arxiv.org/html/2511.06418

- Evaluates both factual knowledge of known drug mechanisms AND reasoning about novel situations
- Most relevant to NegBioDB: tests whether LLMs can reason about WHY drugs work (or don't work)

### 2.4 DrugGPT and Pharmacology Evaluation (2024)

**Paper:** "Aligning Large Language Models with Humans: A Comprehensive Survey of ChatGPT's Aptitude in Pharmacology"
**URL:** https://link.springer.com/article/10.1007/s40265-024-02124-2

- DrugGPT consistently outperforms GPT-4 across all pharmaceutical categories
- Shows specialized systems exceed general-purpose LLMs for drug-specific tasks
- Highlights hallucination as a fundamental limitation

### 2.5 Drug-Target Binding Prediction Survey (2025)

**Paper:** "A survey on deep learning for drug-target binding prediction: models, benchmarks, evaluation, and case studies"
**URL:** https://academic.oup.com/bib/article/26/5/bbaf491/8260789

- Comprehensive metrics: Accuracy, AUROC, AUPR, MCC, F1 Score
- Highlights that negative data handling is a critical challenge across all DTI models
- Notes that increasing training set imbalance correlates with declining model performance

### 2.6 Collaborative LLM for Drug Analysis (Nature Biomedical Engineering, 2025)

**URL:** https://www.nature.com/articles/s41551-025-01471-z

- Nature BME publication demonstrating collaborative LLM approaches for drug analysis
- Represents growing mainstream acceptance of LLMs in drug discovery

---

## 3. Scientific Reasoning Benchmarks

### 3.1 Chain-of-Thought Evaluation

| Benchmark | CoT Impact | Notes |
|-----------|-----------|-------|
| MMLU-Pro | CoT significantly improves performance (unlike original MMLU) | Indicates genuine reasoning, not pattern matching |
| SciBench | Error profiles analyzed via 10 problem-solving ability categories | CoT helps some skills but hurts others |
| ScienceQA | Multimodal-CoT achieves 90.45% with sub-1B models | Two-stage: rationale generation then answer |
| GPQA | CoT essential for expert-level questions | |

### 3.2 Multi-Step Reasoning Evaluation

**Key benchmarks testing multi-step scientific reasoning:**

1. **SciBench** - All problems require multiple reasoning steps + complex arithmetic
2. **MMLU-Pro** - 10 answer choices force more deliberate reasoning
3. **SciKnowEval L3** - "Thinking profoundly" level: critical thinking, logical deduction, numerical calculation
4. **GPQA Diamond** - Questions designed to require expert-level multi-step reasoning

### 3.3 Counterfactual Reasoning in Science

**Key papers and benchmarks (2024-2025):**

1. **CounterBench** - 1,000 questions with varying causal structures testing hypothetical scenario reasoning
   - Tests how models reason about "what if" scenarios

2. **CausalProbe-2024** - Two variants:
   - CausalProbe-H (Hard): one-choice with counterfactual distractors
   - CausalProbe-M (Multiple): multiple correct answers requiring nuanced discrimination

3. **CausalLink** - Interactive evaluation framework for causal reasoning
   - URL: https://aclanthology.org/2025.findings-acl.1147.pdf

4. **Executable Counterfactuals Framework** - Operates on code/math domains
   - Uses SFT and RLVR to enhance counterfactual reasoning

5. **"On the Eligibility of LLMs for Counterfactual Reasoning" (2025)**
   - URL: https://arxiv.org/abs/2505.11839
   - Finding: Current LLMs remain weak at counterfactual and causal reasoning
   - RL elicits stronger generalization than SFT for counterfactual tasks

6. **Medical LLM Counterfactual Study (eLife 2025)**
   - "Critique of impure reason: Unveiling the reasoning behaviour of medical large language models"
   - URL: https://elifesciences.org/articles/106187
   - Tests whether medical LLMs reason or merely pattern-match

**Relevance to NegBioDB:** Evaluating LLMs on negative DTI data is inherently a counterfactual reasoning task: "Given that this drug does NOT bind this target, can the LLM reason about WHY?" This is a completely unaddressed evaluation dimension.

---

## 4. Information Extraction Benchmarks

### 4.1 BioNLP Shared Tasks (2024-2025)

**BioNLP 2025 Workshop at ACL 2025:**
- URL: https://aclanthology.org/2025.bionlp-1.0.pdf

**Four Shared Tasks in 2025:**
1. SMAFIRA - Annotating literature for finding methods alternative to animal experiments
2. ClinIQLink 2025 - Evaluating generative models on factually accurate information

**Key benchmarking results:**
- GPT-5 on BC5CDR-Chemical NER: 0.886 F1 (zero-shot), 0.874 (five-shot) vs. SOTA 0.950
- Source: https://arxiv.org/pdf/2509.04462

**Comprehensive BioNLP Evaluation (Nature Communications 2025):**
- URL: https://www.nature.com/articles/s41467-025-56989-2
- Evaluated GPT-3.5, GPT-4, LLaMA 2, PMC LLaMA
- Finding: Fine-tuned domain models still outperform LLMs on information extraction tasks

### 4.2 Chemical NER/RE Benchmarks

| Benchmark | Task | Best LLM Performance | SOTA (fine-tuned) |
|-----------|------|---------------------|-------------------|
| BC5CDR-Chemical | Chemical NER | 0.886 F1 (GPT-5 zero-shot) | 0.950 F1 |
| ChemProt | Chemical-Protein RE | Varies | Fine-tuned models dominate |
| DDIExtraction | Drug-Drug Interaction RE | LLMs competitive | Traditional NER+RE better |

### 4.3 Structured Data Extraction from Scientific Literature

**Key papers:**

1. **"Structured information extraction from scientific text with large language models" (Nature Communications 2024)**
   - URL: https://www.nature.com/articles/s41467-024-45563-x
   - Fine-tuned GPT-3, Llama-2 can extract complex scientific knowledge
   - Output: simple English sentences or JSON objects

2. **"Benchmarking LLM-based Information Extraction Tools for Medical Documents" (medRxiv 2026)**
   - URL: https://www.medrxiv.org/content/10.64898/2026.01.19.26344287v1
   - 1,000 mock medical documents
   - GPT 4.1-mini: average F1 of 55.6
   - Compared zero-shot and one-shot with various inputs

3. **JSONSchemaBench (2025)** - Evaluating constrained decoding with LLMs on structured JSON output
   - Tests whether LLMs can output valid JSON conforming to a given schema

4. **CaseReportBench (PMC 2025)**
   - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12477612/
   - Dense information extraction from clinical case reports
   - **Key weakness: LLMs struggle with recognizing negative findings for differential diagnosis**

### 4.4 Benchmarks for Extracting Negative/Null Results

**THIS IS THE CRITICAL GAP.** My research found no dedicated benchmark for extracting negative or null scientific results from literature. The closest work:

1. **"Diagnosing Structural Failures in LLM-Based Evidence Extraction for Meta-Analysis" (2026)**
   - URL: https://arxiv.org/abs/2602.10881
   - Evaluated LLMs on extracting meta-analytic evidence across 5 scientific domains
   - **Key finding:** Full meta-analytic association tuples extracted with near-zero reliability
   - Failures stem from: role reversals, cross-analysis binding drift, instance compression, numeric misattribution
   - Performance moderate for single-property queries but degrades sharply for variable binding

2. **"Towards Fine-Grained Extraction of Scientific Claims from Heterogeneous Tables Using Large Language Models" (VLDB 2025 Workshop)**
   - URL: https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/TaDA/TaDA25_16.pdf
   - Attempts fine-grained claim extraction from scientific tables
   - Smaller LLMs (Llama3-8B) face significant challenges even with in-context examples

3. **CaseReportBench (2025)** - Notes that LLMs specifically struggle with negative clinical findings

**Gap Analysis:** No benchmark exists that specifically evaluates:
- Extracting "no binding observed" results from DTI literature
- Distinguishing "tested and negative" from "not tested"
- Classifying confidence levels of negative results
- Extracting the experimental conditions under which negative results were obtained

---

## 5. LLM-as-Judge Approaches

### 5.1 G-Eval (Liu et al., 2023)

**Paper:** "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"
**URL:** https://deepeval.com/docs/metrics-llm-evals | https://www.confident-ai.com/blog/g-eval-the-definitive-guide

| Attribute | Detail |
|-----------|--------|
| **Method** | GPT-4 prompted with task-specific rubrics to score outputs |
| **Axes** | Relevance, coherence, factual consistency |
| **Output** | Continuous score 0-1 (not a fixed rubric scale) |
| **Advantage** | Higher correlation with human ratings than BLEU, ROUGE, sentence embeddings |
| **Limitation** | No formal structured rubric; all evaluation steps treated equally; rubric can be provided but not natively structured |

### 5.2 MT-Bench / LLM-as-a-Judge (Zheng et al., NeurIPS 2023)

**Paper:** "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
**URL:** https://arxiv.org/abs/2306.05685

| Attribute | Detail |
|-----------|--------|
| **Method** | Pairwise comparison: LLM judge sees question + two answers, picks better one or declares tie |
| **Scale** | 80 multi-turn questions across 8 categories |
| **Agreement** | GPT-4 as judge: >80% agreement with human preferences (matches inter-human agreement) |
| **Biases** | Position bias, verbosity bias -- mitigated via prompt engineering, order randomization, ensemble aggregation |

### 5.3 Expert Rubric Evaluation

**Best practices from recent literature (2024-2025):**

1. **Explicit rubric specification** - Define exact criteria and score ranges in the prompt
2. **Explanation-then-score** - Prompting LLMs to explain ratings significantly improves human alignment (Chiang & Lee 2023)
3. **Multi-axis evaluation** - Score separate dimensions independently (accuracy, completeness, reasoning quality)
4. **Ensemble/voting** - Aggregate across model families for robustness
5. **Post-hoc calibration** - Quantitative calibration or uncertainty estimation to adjust scores

### 5.4 RevisEval (2025)

**Paper:** "RevisEval: Improving LLM-as-a-Judge via Response-Adapted References"
**URL:** https://openreview.net/pdf?id=1tBvzOYTLF

- Generates response-adapted references to improve evaluation quality
- Addresses the challenge of evaluating when there's no single correct answer

### 5.5 Automated Metrics for Open-Ended Scientific Text

| Metric Type | Examples | Best For |
|-------------|----------|----------|
| **Lexical** | BLEU, ROUGE | Surface-level text similarity |
| **Semantic** | BERTScore, Sentence-BERT | Meaning preservation |
| **Factual** | G-Eval, FActScore | Factual accuracy |
| **LLM-Judge** | GPT-4 pairwise, rubric-based | Nuanced quality assessment |
| **Domain-specific** | Chemical validity (SMILES parse), Tanimoto similarity | Chemistry/drug tasks |

---

## 6. Agentic / Discovery Benchmarks

### 6.1 ScienceAgentBench (ICLR 2025)

**Paper:** "ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery"
**URL:** https://arxiv.org/abs/2410.05080 | https://github.com/OSU-NLP-Group/ScienceAgentBench

| Attribute | Detail |
|-----------|--------|
| **Scale** | 102 tasks from 44 peer-reviewed papers |
| **Domains** | Bioinformatics, computational chemistry, GIS, psychology/cognitive neuroscience |
| **Output** | Self-contained Python program files (not JSON/workflow descriptions) |
| **Scoring** | Program correctness, execution results, cost |
| **Best Result** | 32.4% tasks solved independently; 34.3% with expert knowledge |

### 6.2 ResearchBench (2025)

**URL:** https://arxiv.org/abs/2503.21248
- Benchmarks LLMs on scientific discovery via inspiration-based task decomposition

### 6.3 Unified Scientific Benchmark Suite (2025)

**From OpenReview paper:**
Integrates 10 prominent scientific benchmarks: GPQA, MMLU-Pro, SuperGPQA, LabBench, OlympiadBench, SciBench, SciRIFF, UGPhysics, SciEval, and SciKnow-Eval.

---

## 7. SciRIFF: Scientific Literature Understanding

**URL:** https://github.com/allenai/SciRIFF

| Attribute | Detail |
|-----------|--------|
| **Focus** | LLM instruction-following for scientific literature understanding |
| **Evaluation** | 9 held-out tasks covering representative task categories and scientific domains |
| **Developer** | Allen AI |
| **Relevance** | Directly applicable: could extend to include negative result extraction tasks |

---

## 8. Gap Analysis and Implications for NegBioDB Benchmark

### 8.1 What Exists (Covered Territories)

| Capability | Benchmark(s) | Status |
|------------|-------------|--------|
| Chemical knowledge recall | ChemBench, MMLU-Pro, GPQA | Well-covered |
| Molecule property prediction | ChemLLMBench, Mol-Instructions, LLaSMol | Well-covered |
| Reaction/retrosynthesis | ChemLLMBench, Mol-Instructions | Well-covered |
| Medical QA | MedQA, BioASQ | Saturating (>95% accuracy) |
| Drug-drug interaction | Rx-LLM, DDI benchmarks | Partially covered |
| Chemical NER/RE | BC5CDR, ChemProt | Covered by traditional NLP |
| Scientific reasoning | SciBench, GPQA, MMLU-Pro | Covered but not drug-specific |
| Structured extraction | CaseReportBench, JSONSchemaBench | Emerging |

### 8.2 What Does NOT Exist (The Gap NegBioDB Fills)

| Missing Capability | Description | NegBioDB Opportunity |
|--------------------|-------------|---------------------|
| **Negative DTI prediction** | Can LLMs predict that a drug will NOT bind a target? | Primary benchmark task |
| **Negative result extraction** | Can LLMs extract "no binding observed" from papers? | Information extraction task |
| **Negative result reasoning** | Can LLMs explain WHY a drug doesn't bind? | Reasoning/explanation task |
| **Confidence classification** | Can LLMs distinguish "tested negative" from "untested"? | Classification task |
| **Assay context extraction** | Can LLMs extract conditions under which negatives were measured? | Structured extraction task |
| **Counterfactual drug reasoning** | "If we modify this functional group, would binding change?" | Reasoning task |
| **Negative data quality assessment** | Can LLMs assess reliability of negative DTI evidence? | Meta-evaluation task |

### 8.3 Recommended Benchmark Task Design for NegBioDB

Based on this landscape analysis, we recommend the following task categories:

**Task 1: Negative DTI Classification (MCQ)**
- Input: Drug SMILES + Target UniProt ID + Experimental context
- Output: Binary (binding/non-binding) + confidence level
- Scoring: Accuracy, F1, AUROC, AUPR
- Inspiration: DDI prediction benchmarks, ChemLLMBench property prediction

**Task 2: Negative Result Extraction (Structured IE)**
- Input: Scientific paper abstract/full text containing negative DTI results
- Output: Structured JSON with drug, target, assay type, result, confidence, conditions
- Scoring: F1 per field, exact match, schema validity
- Inspiration: CaseReportBench, SciRIFF, structured extraction benchmarks

**Task 3: Negative Result Reasoning (Open-ended)**
- Input: Drug + Target + "No binding observed" + assay conditions
- Question: "Why might this drug fail to bind this target?"
- Output: Free-form scientific explanation
- Scoring: LLM-as-judge with expert rubric (G-Eval style) + expert validation subset
- Inspiration: ChemBench open-ended, DrugChat free-form, drug mechanism benchmark

**Task 4: Tested-vs-Untested Discrimination**
- Input: Drug-target pair + literature context
- Output: Classification (experimentally confirmed negative / assumed negative / untested / ambiguous)
- Scoring: Multi-class accuracy, confusion matrix analysis
- Inspiration: Unique -- no existing benchmark addresses this

**Task 5: Negative Evidence Quality Assessment**
- Input: Description of negative DTI experiment
- Output: Quality tier (high confidence / moderate / low / unreliable) + justification
- Scoring: Agreement with expert panel + LLM-judge on justification quality
- Inspiration: Meta-analysis extraction benchmarks, CausalProbe reasoning

---

## 9. Key Papers Reference List

### Benchmarks
1. ChemBench - Nature Chemistry 2025 - https://www.nature.com/articles/s41557-025-01815-x
2. ChemLLMBench - NeurIPS 2023 - https://github.com/ChemFoundationModels/ChemLLMBench
3. Mol-Instructions - ICLR 2024 - https://github.com/zjunlp/Mol-Instructions
4. ScienceQA - NeurIPS 2022 - https://scienceqa.github.io/
5. MedQA - https://www.vals.ai/benchmarks/medqa
6. GPQA - https://arxiv.org/abs/2311.12022
7. MMLU-Pro - NeurIPS 2024 - https://arxiv.org/abs/2406.01574
8. SciBench - ICML 2024 - https://arxiv.org/abs/2307.10635
9. SciEval - AAAI 2024 - https://arxiv.org/abs/2308.13149
10. SciKnowEval - https://arxiv.org/html/2406.09098v1
11. LLaSMol - COLM 2024 - https://arxiv.org/html/2402.09391v3
12. LAB-Bench - https://github.com/Future-House/LAB-Bench
13. ScienceAgentBench - ICLR 2025 - https://arxiv.org/abs/2410.05080
14. Rx-LLM - 2025 - https://www.medrxiv.org/content/10.64898/2025.12.01.25341004v2
15. SciRIFF - https://github.com/allenai/SciRIFF

### Drug Discovery LLM Evaluation
16. LLMs for DDI Prediction - https://arxiv.org/abs/2502.06890
17. DrugChat - https://www.biorxiv.org/content/10.1101/2024.09.29.615524v1
18. Drug Mechanism Understanding - https://arxiv.org/html/2511.06418
19. LLM3-DTI - https://arxiv.org/html/2511.06269
20. DTI Binding Prediction Survey - https://academic.oup.com/bib/article/26/5/bbaf491/8260789
21. Pharmacology Survey - https://link.springer.com/article/10.1007/s40265-024-02124-2

### Reasoning and Counterfactual
22. CounterBench / CausalProbe-2024 - see emergentmind.com/topics/causalprobe-2024
23. CausalLink - https://aclanthology.org/2025.findings-acl.1147.pdf
24. LLM Counterfactual Eligibility - https://arxiv.org/abs/2505.11839
25. Medical LLM Reasoning - https://elifesciences.org/articles/106187

### Information Extraction
26. BioNLP 2025 - https://aclanthology.org/2025.bionlp-1.0.pdf
27. GPT-5 BioNLP Benchmark - https://arxiv.org/pdf/2509.04462
28. BioNLP Comprehensive - Nature Comms 2025 - https://www.nature.com/articles/s41467-025-56989-2
29. Structured IE from Science - Nature Comms 2024 - https://www.nature.com/articles/s41467-024-45563-x
30. Meta-Analysis Extraction Failures - https://arxiv.org/abs/2602.10881
31. Fine-Grained Scientific Claims - VLDB 2025 - https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/TaDA/TaDA25_16.pdf
32. MedQA Limitations - ACL 2025 - https://aclanthology.org/2025.bionlp-1.24/

### LLM-as-Judge
33. G-Eval - https://www.confident-ai.com/blog/g-eval-the-definitive-guide
34. MT-Bench / LLM-as-Judge - https://arxiv.org/abs/2306.05685
35. RevisEval - https://openreview.net/pdf?id=1tBvzOYTLF
36. LLM-as-Judge Survey - https://arxiv.org/html/2411.15594v1
37. LLM Evaluator Effectiveness - https://eugeneyan.com/writing/llm-evaluators/

---

*Last updated: 2026-03-02*
*Research conducted for NegBioDB project targeting NeurIPS 2026 D&B Track*
