# Bias Detection & Neutralization in News Text

A machine learning pipeline for detecting and neutralizing political bias in news articles. This project combines a BERT-based bias classifier with an LLM-powered rewrite module to identify biased sentences and generate neutral alternatives while preserving semantic meaning.

## Overview

**Course:** DATA 641 NLP  
**Team:** Group 11 - Daniel (121033345), Alex (121166357)  
**Duration:** 3 weeks

### Key Components

1. **Bias Classifier** – Detects whether text is biased and identifies bias direction (left/center/right)
2. **LLM Rewrite Module** – Generates neutral alternatives to biased sentences
3. **LangGraph Pipeline** – Orchestrates classification, rewriting, and safety checks
4. **Evaluation Framework** – Measures bias reduction, semantic preservation, and output quality

## Features

- **Binary Classification:** Biased vs. non-biased sentence detection (BABE expert labels)
- **Multi-class Classification:** Predicts the source outlet's political leaning (left / center / right, via AllSides) as a weak signal for routing the rewrite prompt
- **Semantic Safety Gate:** SBERT similarity check with retry logic (threshold ≥ 0.80)
- **Progressive Neutralization:** Automatic retry up to 3 times if similarity drops below threshold
- **Comprehensive Metrics:** Tracks bias reduction, semantic similarity, perplexity, and more
- **Experiment Tracking:** Full MLflow integration for reproducibility

## Installation

### Prerequisites
- Python 3.8+
- GPU (recommended for faster inference)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd Bias-Detection-Neutralization-in-News

# Install dependencies
pip install -r requirements.txt

# Initialize MLflow (optional but recommended)
mlflow server --host 0.0.0.0 --port 5000
```

### Dataset

We use **BABE** (*Bias Annotations By Experts*, Spinde et al. 2022), the
expert-annotated successor to MBIC, hosted on HuggingFace as
[`mediabiasgroup/BABE`](https://huggingface.co/datasets/mediabiasgroup/BABE).

```bash
python data_preprocessing.py \
    --raw_dir data/raw \
    --processed_dir data/processed
```

This downloads BABE (~4.1k expert-annotated sentences), cleans the text,
imputes the `type` column (left/center/right) from each row's `outlet` using
the AllSides leaning map, performs a stratified 70/15/15 split on
`(label, outlet)`, and writes:

- `data/raw/babe.csv`
- `data/processed/{train,val,test}.csv`
- `data/processed/split_summary.json`

> The `type` (left/center/right) field is **outlet-derived**, not
> sentence-derived. We use it as a weak signal for selecting the rewrite
> prompt template, not as a sentence-level political-content classifier.

## Project Structure

```
.
├── data/
│   ├── raw/                    # Full BABE CSV (downloaded from HuggingFace)
│   ├── processed/              # Cleaned, stratified train/val/test splits
│   └── test_sentences/         # 50-sentence evaluation set with reference rewrites
├── models/
│   ├── tfidf_baseline/         # TF-IDF + Logistic Regression checkpoint
│   ├── bert_classifier/        # Fine-tuned BERT/RoBERTa checkpoint
│   ├── sbert_similarity/       # (Optional) SBERT cache or reference
│   └── llm_rewrite/            # Mistral 7B artifacts (if cached locally)
├── src/
│   ├── preprocessing.py        # Data cleaning and dataset preparation
│   ├── classifier.py           # TF-IDF and BERT classifier implementations
│   ├── rewrite.py              # LLM rewrite prompts and logic
│   ├── similarity.py           # SBERT semantic similarity scorer
│   ├── pipeline.py             # LangGraph pipeline definition
│   └── evaluation.py           # Evaluation and metrics computation
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_classifier_training.ipynb
│   └── 03_pipeline_evaluation.ipynb
├── tests/
│   ├── test_classifier.py
│   ├── test_rewrite.py
│   └── test_pipeline.py
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── CLAUDE.md                   # Project documentation for Claude Code
└── project_plan.md             # Detailed 3-week breakdown
```

## Usage

### Quick Start

```python
from src.pipeline import build_pipeline

# Initialize the pipeline
pipeline = build_pipeline(
    classifier_path="models/bert_classifier",
    llm_model="mistralai/Mistral-7B-Instruct-v0.2"
)

# Run on a sentence
result = pipeline.invoke({
    "input_sentence": "The reckless senator pushed through the controversial bill"
})

print(result)
# Output:
# {
#   "original": "...",
#   "rewritten": "...",
#   "bias_label": "right",
#   "bias_score_before": 0.91,
#   "bias_score_after": 0.15,
#   "similarity": 0.84,
#   "retries": 1,
#   "warning": null
# }
```

### Running the Full Evaluation

```bash
python src/evaluation.py \
    --test_set data/test_sentences/eval_set.json \
    --output_dir results/
```

### Training from Scratch

```bash
# Download + preprocess BABE (writes to data/raw/ and data/processed/)
python data_preprocessing.py

# Train TF-IDF baseline
python src/classifier.py --method tfidf --output models/tfidf_baseline/

# Fine-tune BERT
python src/classifier.py --method bert --epochs 3 --batch_size 16 --output models/bert_classifier/

# Evaluate
python src/evaluation.py --test_set data/processed/test.csv
```

## Results

### Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| BERT F1 > TF-IDF baseline | F1 macro improvement | In Progress |
| Minority class detection | Left/right F1 > 0.50 | In Progress |
| Bias reduction | ≥ 50% score reduction | In Progress |
| Semantic preservation | SBERT similarity ≥ 0.80 | In Progress |
| Pipeline stability | 100% of inputs complete | In Progress |

### Key Metrics

Results will be updated upon project completion (Week 3).

**Classifier Performance (Test Set)**
- TF-IDF F1 macro: TBD
- BERT F1 macro: TBD
- Left-leaning detection F1: TBD
- Right-leaning detection F1: TBD

**Rewrite Module**
- Mean bias score reduction: TBD
- Mean semantic similarity: TBD
- First-pass success rate: TBD
- Max-retry triggers: TBD

## Pipeline Architecture

The LangGraph pipeline consists of five nodes:

1. **Input Node** – Validates and formats input
2. **Bias Classifier Node** – Detects bias presence and type
3. **Rewrite Node** – Generates neutral alternatives (triggered if bias_score > 0.5)
4. **Safety Check Node** – Validates semantic preservation with retry logic
5. **Output Node** – Assembles final response

**Retry Logic:** If similarity < 0.80 and retries < 3, loop back to rewrite. Otherwise, output best attempt with warning flag if applicable.

## Development Notes

- **Class Imbalance:** BABE is roughly balanced for the binary biased/non-biased label, but the outlet-derived `type` skews away from `center`. Handled with stratified `(label, outlet)` splits and (optional) weighted cross-entropy on the 3-class head.
- **LLM Inference:** Uses `mistralai/Mistral-7B-Instruct-v0.2`. For faster iterations, consider quantization or smaller models like `Mistral-7B`.
- **Semantic Similarity:** SBERT `all-MiniLM-L6-v2` chosen for speed. For higher accuracy, use `all-mpnet-base-v2` (slower).
- **Retry Threshold:** Set conservatively at 0.80 to ensure meaning preservation. Can be tuned based on use case.

## Experiment Tracking

All model runs are logged to MLflow with:
- Hyperparameters (learning rate, batch size, etc.)
- Metrics (F1, precision, recall, similarity scores)
- Model artifacts (checkpoints, tokenizers)
- Evaluation plots and JSON results

Access the MLflow UI at `http://localhost:5000` (when server is running).

## Team Responsibilities

| Member | Primary Contributions |
|--------|----------------------|
| **Alex** | Bias classifier (TF-IDF + BERT), LLM rewrite module, MLflow logging |
| **Daniel** | LangGraph pipeline, evaluation scripts, dataset exploration, report writing |
| **Joint** | Integration testing, final report assembly, presentation |

## References

- **Dataset:** BABE — *Bias Annotations By Experts* ([HuggingFace `mediabiasgroup/BABE`](https://huggingface.co/datasets/mediabiasgroup/BABE)); Spinde et al., "Neural Media Bias Detection Using Distant Supervision With BABE" (2022)
- **Models:** 
  - BERT classifier: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019)
  - LLM rewrite: Mistral 7B Instruct
  - Similarity: Sentence Transformers (Reimers & Gupta, 2019)
- **Pipeline Framework:** LangGraph (LangChain)

## License

Course project for DATA 641 NLP.

---

For detailed implementation notes and architecture decisions, see [CLAUDE.md](CLAUDE.md).
