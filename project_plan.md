cla# Bias Detection & Neutralization in News Text — 3-Week Project Plan

**Course:** DATA 641 NLP  
**Group:** Group 11  
**Members:** Alex (121166357) · Daniel (121033345)  

---

## Overview

**Goal:** Build an end-to-end pipeline to detect and neutralize political bias in news sentences.

**Architecture:** 

- Bias classifier (BERT/RoBERTa) → identifies if text is biased
- LLM rewriter (Mistral 7B) → generates neutral alternative
- Safety gate (SBERT similarity) → validates meaning preserved
- LangGraph orchestration → handles retries and routing

**Team Roles:**

- **Alex:** Classifier training, LLM rewrite prompts, experiment logging
- **Daniel:** Pipeline architecture, evaluation scripts, data exploration

---

## Week 1: Setup, Baseline Models & Prototyping

**Goal:** Establish data pipeline, train baseline classifier, prototype rewrite prompts, design pipeline architecture.

### Alex's Tasks

#### Task 1.1: Prepare Dataset

**What:** Download the BABE dataset from HuggingFace (`mediabiasgroup/BABE`), clean, and analyze.

> **Note:** We use **BABE** (Spinde et al., 2022) instead of MBIC because it is the same group's expert-annotated successor: ~4.1k sentences with higher inter-annotator agreement, and it is the de-facto benchmark for sentence-level media bias.

**Specific Steps:**

1. Load via `datasets.load_dataset("mediabiasgroup/BABE", split="train")`. Inspect columns: `text`, `outlet`, `label` (0/1), `topic`, `news_link`, `biased_words`, `uuid`, `type` (left/center/right, partially populated), `label_opinion`
2. Rename `text` → `sentence`, map `label` `1`→`biased`, `0`→`non-biased`
3. Drop rows with null `sentence` or `label`
4. Lowercase text, remove URLs/non-printable chars/excess whitespace (regex)
5. **Impute missing `type`** from `outlet` using a fixed AllSides outlet→leaning map (BABE leaves `type` null for many rows). Fall back to `center` for unknown outlets.
6. Print class distribution (count each label + type + per-outlet breakdown)

**Done when:**

- BABE downloaded and cleaned (~4.1k sentences after filtering)
- Per-row `type` is fully populated (no nulls)
- Class distribution printed and saved (BABE is closer to ~50/50 biased vs non-biased, much less skewed than MBIC)
- Saved to `data/raw/babe.csv`

---

#### Task 1.2: Train TF-IDF Baseline

**What:** Build a simple baseline classifier for comparison.

**Specific Steps:**

1. Use `TfidfVectorizer(max_features=50000, ngram_range=(1,2))`
2. Train **two separate classifiers:**
  - Task A (binary): biased vs non-biased
  - (optional) Task B (3-class): left vs center vs right (on biased samples only) 
3. Evaluate on validation set
4. Record: F1 (macro), F1 per class, precision, recall

**Done when:** 

- Both classifiers trained and evaluated
- F1 macro scores recorded (this is your **B1 baseline** to beat later)
- Metrics logged to MLflow

---

#### Task 1.3: Prototype Rewrite Prompts

**What:** Draft and test LLM prompts for neutralizing biased sentences.

**Specific Steps:**

1. Load `mistralai/Mistral-7B-Instruct-v0.2` from HuggingFace
2. Write 3 system prompt templates:
  - Left-leaning bias (remove progressive framing, keep facts)
  - Right-leaning bias (remove conservative framing, keep facts)
  - General bias (fallback prompt)
3. Test on 20 hand-selected biased sentences from training data
4. Manually review all 20 outputs:
  - Does rewrite preserve the original meaning?
  - Is the tone genuinely neutral?
  - Does it avoid hallucinations or adding new facts?
5. Iterate prompts at least once based on feedback

**Done when:**

- 3 prompt templates written and tested
- Manual review notes recorded
- Prompts improved based on feedback

---

#### Task 1.4: Set Up MLflow

**What:** Configure experiment tracking for all model runs.

**Specific Steps:**

1. Install MLflow: `pip install mlflow`
2. Start local server: `mlflow server --host 0.0.0.0 --port 5000`
3. Create experiment named `bias-detection`
4. Log TF-IDF baseline runs with:
  - Hyperparameters: `max_features`, `ngram_range`, `class_weight`
  - Metrics: F1 macro, F1 per class
  - Artifact: pickled model file

**Done when:**

- MLflow UI accessible at `localhost:5000`
- At least one TF-IDF run logged and visible in UI

---

### Daniel's Tasks

#### Task 1.5: Data Exploration & Visualization

**What:** Understand dataset structure and patterns.

**Specific Steps:**

1. Create bar charts:
  - Class distribution (biased vs non-biased)
  - Bias type distribution (left, center, right) — note that `type` is outlet-derived in BABE
  - Outlet breakdown (Breitbart, HuffPost, Reuters, Alternet, Fox, Federalist, etc.)
  - Topic breakdown (BABE has a `topic` column: elections, vaccine, white-nationalism, ...)
2. Create histogram: sentence length (word count) by class
3. Identify patterns: do left/right sentences differ in length from neutral ones? Do certain topics correlate with bias label?
4. Write 3–4 bullet-point summary

**Done when:**

- 4 visualizations saved as PNG files
- Summary bullet points written
- Findings recorded for report

---

#### Task 1.6: Build Semantic Similarity Scorer

**What:** Create a function to measure if rewritten sentences preserve meaning.

**Specific Steps:**

1. Install: `pip install sentence-transformers`
2. Load `all-MiniLM-L6-v2` model (lightweight, fast)
3. Write function:
  ```python
   def compute_similarity(original: str, rewritten: str) -> float:
       # Return cosine similarity [0, 1]
  ```
4. Test on 5 sentence pairs:
  - Identical sentences → expect ≈1.0
  - Paraphrases → expect ≥0.85
  - Opposite meaning → expect <0.5

**Done when:**

- Function works correctly on all 5 test pairs
- Results match expected ranges
- Function ready for pipeline integration

---

#### Task 1.7: Design LangGraph Pipeline

**What:** Create architecture blueprint for the pipeline.

**Specific Steps:**

1. Define state schema (TypedDict) with fields:
  - `input_sentence`, `bias_label`, `bias_type`, `bias_score`
  - `rewritten_sentence`, `similarity_score`, `new_bias_score`
  - `retry_count`, `warning`, `final_output`
2. Design 5 nodes (stub functions, return dummy values):
  - **Input Node:** Validate input, initialize state
  - **Bias Classifier Node:** Detect bias presence + type
  - **Rewrite Node:** Generate neutral alternative (if bias_score > 0.5)
  - **Safety Check Node:** Validate similarity (threshold 0.80)
  - **Output Node:** Format final JSON response
3. Define routing conditions:
  - If `bias_score ≤ 0.5` → skip rewrite, go to output
  - If `similarity_score < 0.80` AND `retry_count < 3` → retry rewrite
  - If `retry_count ≥ 3` → force output with warning

**Done when:**

- State schema defined
- All 5 node stubs written (can run without errors)
- Routing conditions documented
- Graph compiles and runs end-to-end

---

#### (Optional) Task 1.8: Create Evaluation Test Set

**What:** Build 50-sentence benchmark with reference neutral rewrites.

**Specific Steps:**

1. Select 50 biased sentences from test split:
  - Aim for ~17 left-leaning, ~17 right-leaning, ~16 general
  - Mix of short and long sentences
2. For each sentence, write a **human-quality neutral rewrite** by hand
3. Save as JSON: `data/test_sentences/eval_set.json`
  ```json
   [
     {"id": 1, "biased": "...", "bias_type": "left", "reference_neutral": "..."},
     ...
   ]
  ```

**Done when:**

- 50 sentences collected with balanced bias types
- Hand-written neutral rewrites completed for all 50
- JSON file created and validated

---

### Joint Tasks

#### Task 1.9: Finalize Data Splits

**What:** Create train/val/test splits and agree on interfaces.

**Specific Steps:**

1. Perform stratified 70/15/15 split on BABE (stratified by `label` AND `outlet`; collapse rare `(label, outlet)` strata into label-only strata to avoid sklearn errors)
2. Save as: `data/processed/train.csv`, `val.csv`, `test.csv` (+ `split_summary.json`)
3. Commit to repo
4. Agree on exact I/O contracts:
  - Classifier input/output format
  - Pipeline state field names
  - Error handling behavior

**Done when:**

- All 3 CSV files created and committed
- Split counts documented (BABE is ~4.1k sentences, so expect roughly train≈2900, val≈600, test≈600)
- Interface contracts written down

---

#### Task 1.10: Week 1 Sync

**What:** Review progress, unblock issues.

**Action:** 30-minute meeting

- Alex presents: baseline F1 scores, rewrite prompt samples (5 good/5 bad examples)
- Daniel presents: dataset analysis, pipeline stubs, evaluation test set preview
- Discuss: any blockers, data quality issues, rewrite prompt direction

**Done when:** Meeting completed, notes recorded

---

### Week 1 Deliverables Checklist

- Dataset cleaned and preprocessed
- TF-IDF baseline trained; F1 scores recorded as B1 baseline
- 3 rewrite prompt templates drafted and tested on 20 sentences
- MLflow experiment set up and first baseline run logged
- Data visualizations created (4 PNGs)
- Semantic similarity scorer working and tested
- LangGraph pipeline stubs complete and compiling
- 50-sentene evaluation test set created with hand-written references
- Train/val/test splits finalized and committed
- Week 1 sync meeting completed

---

## Week 2: BERT Classifier & Full Pipeline Implementation

**Goal:** Replace baseline with BERT, build real LLM rewrite node, wire all pipeline nodes, achieve end-to-end execution.

### Alex's Tasks

#### Task 2.1: Fine-Tune BERT Classifier

**What:** Train a BERT model on bias classification task.

**Specific Steps:**

1. Choose base model: `bert-base-uncased` or `roberta-base`
2. Train **Task A** (binary: biased vs non-biased):
  - Batch size: 16–32
  - Learning rate: 2e-5
  - Epochs: 3 (with early stopping, patience=2)
  - Loss: Cross-entropy with class weighting
3. Train **Task B** (3-class: left/center/right on biased samples only):
  - Apply oversampling (SMOTE/manual duplication) if minority F1 < 0.50
  - Save best validation checkpoint
4. Log all runs to MLflow

**Done when:**

- Both Task A and Task B models trained and saved
- BERT F1 macro ≥ TF-IDF baseline
- Models saved to `models/bert_classifier/`
- MLflow runs logged with hyperparameters and metrics

---

#### Task 2.2: Implement Bias Classifier Node

**What:** Create actual pipeline node for bias detection.

**Specific Steps:**

1. Replace stub with real function:
  - Load BERT checkpoint
  - Run inference on input sentence
  - Return: `bias_label`, `bias_type`, `bias_score`
2. `bias_score` = softmax probability of 'biased' class
3. Write unit test:
  - Input: `"The reckless senator pushed through the controversial bill"`
  - Assert: `bias_score > 0.5` and `bias_type == "right"`

**Done when:**

- Node function complete
- Unit test passes
- Function handles both biased and non-biased inputs correctly

---

#### Task 2.3: Implement Rewrite Node

**What:** Build LLM inference node for neutralizing sentences.

**Specific Steps:**

1. Replace stub with real function:
  - Check if `bias_score > 0.5`; if not, skip rewrite
  - Select prompt template based on `bias_type`
  - Call Mistral 7B with: `max_new_tokens=200, temperature=0.3, do_sample=True`
  - Parse output (strip preamble like "Sure! Here is...")
  - Return clean rewritten sentence
2. Log: input sentence, bias type, rewritten output

**Done when:**

- Node function complete and tested
- Preamble stripping working correctly
- Logging implemented

---

#### Task 2.4: BERT vs TF-IDF Comparison

**What:** Document performance comparison for report.

**Specific Steps:**

1. Create MLflow run: `bert-vs-tfidf-comparison`
2. Log side-by-side metrics on validation set:
  - F1 macro, F1 per class (left/center/right)
  - Precision, Recall per class
  - Both models on same val set for fair comparison

**Done when:**

- Comparison run logged in MLflow
- Comparison table ready for report

---

### Daniel's Tasks

#### Task 2.5: Implement Safety Check Node

**What:** Create gate that validates semantic similarity with retry logic.

**Specific Steps:**

1. Replace stub with real function:
  - Compute similarity: `compute_similarity(original, rewritten)`
  - Return routing decision:
    - `"pass"` if similarity ≥ 0.80
    - `"retry"` if similarity < 0.80 AND `retry_count < 3`
    - `"force_pass"` if `retry_count ≥ 3`
2. If `force_pass`: add warning flag: `{"warning": "similarity below threshold after max retries"}`
3. Write unit tests for all 3 branches

**Done when:**

- Node function complete
- All 3 routing branches tested
- Unit tests passing

---

#### Task 2.6: Add Post-Rewrite Bias Re-evaluation

**What:** Re-score rewritten sentences to measure bias reduction.

**Specific Steps:**

1. After rewrite node, run classifier on rewritten sentence
2. Store `new_bias_score` in pipeline state
3. Use this to measure bias reduction in evaluation

**Done when:**

- Post-rewrite classification working
- `new_bias_score` stored correctly in state

---

#### Task 2.7: Finalize Output Node & Retry Guard

**What:** Assemble final response and guard against infinite loops.

**Specific Steps:**

1. Initialize `retry_count = 0` at pipeline start
2. Increment `retry_count` on each rewrite attempt
3. Output node assembles JSON:
  ```json
   {
     "original": "...",
     "rewritten": "...",
     "bias_label": "left",
     "bias_score_before": 0.91,
     "bias_score_after": 0.15,
     "similarity": 0.84,
     "retries": 1,
     "warning": null
   }
  ```

**Done when:**

- Output node complete
- Retry guard working (max 3 retries enforced)
- JSON output format validated

---

#### Task 2.8: Draft Evaluation Scripts

**What:** Prepare code to run full evaluation in Week 3.

**Specific Steps:**

1. Write `evaluate_rewrite(test_set_path, pipeline)` function that:
  - Iterates 50-sentence test set
  - Computes metrics:
    - Bias score reduction: `(before - after) / before`
    - SBERT similarity scores
    - Perplexity using GPT-2
    - % passing similarity gate first try
  - Saves results to `eval_results.json`

**Done when:**

- Evaluation script complete and tested on 5 sentences
- Output JSON structure defined
- Ready to run full evaluation in Week 3

---

### Joint Tasks

#### Task 2.9: Wire Pipeline & Smoke Test

**What:** Connect all 5 nodes and verify end-to-end execution.

**Specific Steps:**

1. Wire all nodes into `LangGraph StateGraph`
2. Add conditional edges:
  - `bias_score > 0.5` → rewrite
  - `similarity < 0.80` → retry (if retries < 3)
3. Compile: `graph = graph.compile()`
4. Run smoke test on 10 val set sentences
5. Verify:
  - No exceptions
  - State fields populated correctly
  - Retry logic fires on at least 1 sentence
6. Fix any interface mismatches

**Done when:**

- Graph compiles without errors
- Smoke test completes on 10 sentences
- All state fields correct
- Retry logic verified

---

#### Task 2.10: Mid-Project Sync

**What:** Review progress and adjust scope if needed.

**Action:** 30-minute meeting

- Demo: pipeline running on 3 example sentences
- Review: output quality, speed, any bottlenecks
- Discuss: LLM inference speed acceptable? Any prompts needing revision?

**Done when:** Meeting completed, decisions documented

---

### Week 2 Deliverables Checklist

- BERT checkpoint trained and saved to `models/bert_classifier/`
- Bias classifier node working with unit test passing
- Rewrite node implemented with 3 prompt templates
- BERT vs TF-IDF comparison logged to MLflow
- Safety check node with retry logic implemented and tested
- Post-rewrite bias re-evaluation working
- Output node complete with correct JSON format
- Evaluation script drafted and structure validated
- All 5 nodes wired and graph compiling
- Smoke test passing on 10 sentences
- Mid-project sync completed

---

## Week 3: Evaluation, Error Analysis & Final Report

**Goal:** Run full evaluation, analyze results, write report, rehearse presentation.

### Alex's Tasks

#### Task 3.1: Full Classifier Evaluation

**What:** Test both TF-IDF and BERT on held-out test set.

**Specific Steps:**

1. Run classifiers on test set (unseen during training)
2. Compute and record:
  - Accuracy, F1 macro, F1 per class
  - Precision, Recall per class
  - Confusion matrix
3. Log to MLflow run: `final-test-evaluation`
4. Plot confusion matrices (matplotlib/seaborn); save as PNG

**Done when:**

- All metrics computed
- Confusion matrix PNG saved
- Results logged to MLflow

---

#### Task 3.2: Baseline Comparison (B1, B2, B3)

**What:** Document how pipeline compares to simpler approaches.

**Specific Steps:**

1. **B1 (TF-IDF baseline):** Report F1 from Week 1
2. **B2 (rule-based rewriter):** Build simple word-swap rewriter using static charged-word dictionary (5–10 replacements)
  - Test on 50-sentence eval set
  - Measure: bias reduction %, similarity score
3. **B3 (direct LLM, no classifier):** Send sentences directly to Mistral with generic prompt
  - Test on same 50 sentences
  - Measure: bias reduction %, similarity score
4. Create comparison table:
  - Rows: B1, B2, B3, Full Pipeline
  - Columns: F1 (if applicable), Bias Reduction %, Similarity, Perplexity

**Done when:**

- All 3 baselines evaluated
- Comparison table completed
- Results ready for report

---

#### Task 3.3: Write Classifier & Rewrite Sections

**What:** Document classification and rewriting approach for report.

**Specific Steps:**

1. **Section: "Bias Classifier"** (~300 words)
  - Model choice rationale (BERT vs TF-IDF)
  - Training setup (batch size, lr, epochs)
  - How you handled class imbalance
  - Results table: TF-IDF vs BERT F1
  - Error analysis: which sentences were hardest to classify? Why?
2. **Section: "LLM Rewrite Module"** (~300 words)
  - Model choice (Mistral 7B Instruct)
  - Prompt design approach
  - Examples: 2 good rewrites, 1 bad rewrite with explanation
  - Why structured prompting (with classifier) beats direct LLM (B3)

**Done when:**

- Both sections written (~600 words total)
- Sections include tables, examples, and error analysis
- Ready for final report assembly

---

#### Task 3.4: Finalize MLflow & Model Registry

**What:** Organize all experiments and register best model.

**Specific Steps:**

1. Audit all MLflow runs:
  - Each has descriptive name
  - All hyperparameters logged
  - All artifacts (model files, eval JSON, plots) attached
2. Register best BERT model: `bias-classifier-v1` in MLflow Model Registry
3. Export runs summary CSV from MLflow

**Done when:**

- All runs audited and documented
- Best model registered
- Runs CSV ready for report appendix

---

### Daniel's Tasks

#### Task 3.5: Run Full Rewrite Evaluation

**What:** Execute evaluation script on all 50 test sentences.

**Specific Steps:**

1. Run `evaluate_rewrite()` on 50-sentence test set
2. Compute all metrics:
  - Mean bias score reduction %
  - Mean SBERT similarity
  - Mean perplexity
  - % passing similarity gate on first attempt
  - % requiring 1 retry, % requiring 2 retries, % hitting max-retry with warning
3. Save results to `eval_results.json`
4. Check against success criteria:
  - Bias reduction ≥ 50%?
  - Similarity ≥ 0.80?
  - Report actual numbers even if below target

**Done when:**

- All metrics computed and saved
- Results reviewed against success criteria
- Numbers ready for report

---

#### Task 3.6: Error Analysis

**What:** Understand what worked and what failed.

**Specific Steps:**

1. Identify **5 worst rewrites**:
  - Lowest similarity OR highest post-rewrite bias score
  - For each, write 2–3 sentences explaining failure
  - Was it a prompt issue? Hallucination? Ambiguous input?
2. Identify **3 best rewrites**:
  - Highest bias reduction with similarity ≥ 0.80
  - Use as showcase examples for report and slides
3. Save to `error_analysis.json`:
  ```json
   {
     "worst_5": [
       {
         "id": 1,
         "original": "...",
         "rewritten": "...",
         "explanation": "..."
       }
     ],
     "best_3": [...]
   }
  ```

**Done when:**

- 5 worst cases analyzed with explanations
- 3 best cases selected
- JSON file created

---

#### Task 3.7: Write Pipeline & Evaluation Sections

**What:** Document pipeline design and results for report.

**Specific Steps:**

1. **Section: "LangGraph Pipeline Design"** (~350 words)
  - State schema overview
  - Each node's role and logic
  - Edge conditions (routing)
  - Retry logic and max-retry guard
  - Why LangGraph vs simple sequential script
2. **Section: "Evaluation Results"** (~350 words)
  - All metrics from Task 3.5
  - Baseline comparison table (B1/B2/B3 vs Full)
  - Success criteria: which met, which missed
  - Qualitative examples: show 2–3 best/worst rewrite pairs

**Done when:**

- Both sections written (~700 words total)
- Metrics table and comparison table included
- Best/worst examples formatted clearly
- Ready for final report assembly

---

#### Task 3.8: Prepare Qualitative Examples for Report

**What:** Select pipeline examples to illustrate behavior.

**Specific Steps:**

1. Pick 3–5 input/output pairs that best show pipeline behavior
2. For each, create a clean table:
  ```
   | Original | Bias Type | Rewritten | Bias Before | Bias After | Similarity |
   | "..." | right | "..." | 0.92 | 0.15 | 0.84 |
  ```
3. Include: 2 success cases, 1 partial success (low similarity), 1 failure (max retries)

**Done when:**

- 3–5 examples formatted as clean table
- Examples cover range of outcomes
- Ready for report appendix

---

### Joint Tasks

#### Task 3.9: Assemble Final Report

**What:** Combine all sections into 4–6 page report.

**Structure:**

1. Abstract (150 words)
2. Introduction
3. Related Work & Baselines (B1, B2, B3)
4. Dataset (size, class distribution, sources)
5. Classifier Design & Results
6. Rewrite Module Design & Examples
7. Pipeline Architecture & Design Choices
8. Evaluation Results & Success Criteria
9. Discussion (what worked, what didn't, why)
10. Conclusion
11. References
12. Appendix (confusion matrix, qualitative examples, MLflow runs CSV)

**Steps:**

1. Alex writes sections 5–6
2. Daniel writes sections 7–8
3. Both write sections 1, 9–10
4. Proofread for:
  - Consistent terminology (e.g., "bias score" not "bias probability")
  - Consistent figure/table formatting
  - Proper citations
5. Target: **4–6 pages** excluding appendix

**Done when:**

- Report complete and proofread
- All figures/tables formatted consistently
- Ready to submit

---

#### Task 3.10: Finalize & Rehearse Presentation

**What:** Create and practice 10-minute presentation.

**Slides (required):**

1. Problem statement
2. Pipeline diagram (5 nodes with data flow)
3. Dataset stats (size, class distribution)
4. Classifier results table (BERT vs TF-IDF)
5. Rewrite examples (2–3 good, 1 bad with explanation)
6. Evaluation metrics (bias reduction %, similarity, perplexity)
7. Success criteria met/not met
8. Lessons learned
9. Future work

**Steps:**

1. Create slides with real results (replace all preliminary numbers)
2. Rehearse together: time it, cut slides if over 10 minutes
3. Practice: each person speaks 5 minutes
4. Prepare to answer: "What if similarity was still below 0.80?" or "Why BERT over TF-IDF?"

**Done when:**

- Slides finalized with real numbers
- Rehearsal completed and timed
- Presentation ready to deliver

---

#### Task 3.11: Final Submission Package

**What:** Gather all deliverables into single submission.

**Contents:**

- Code repository (CLAUDE.md, README.md, requirements.txt, src/, models/, data/)
- Written report PDF (4–6 pages)
- Presentation slides (PDF or PPTX)
- MLflow artifact directory (all runs, confusion matrix, eval_results.json)
- error_analysis.json with best/worst examples

**Done when:** All files collected and ready to submit

---

### Week 3 Deliverables Checklist

- Full classifier evaluation on test set completed
- B1, B2, B3 baselines evaluated and compared
- Classifier section written (~300 words)
- Rewrite module section written (~300 words)
- MLflow audit complete; best model registered
- Full rewrite evaluation on 50 sentences completed
- Error analysis: 5 worst + 3 best cases documented
- Pipeline design section written (~350 words)
- Evaluation results section written (~350 words)
- Qualitative examples table created (3–5 examples)
- Final report assembled and proofread (4–6 pages)
- Presentation slides finalized and rehearsed
- Submission package collected

---

## Success Criteria

Before submitting, verify:


| Criterion                    | Target                                     | Owner  |
| ---------------------------- | ------------------------------------------ | ------ |
| **BERT F1 > TF-IDF F1**      | BERT macro F1 higher                       | Alex   |
| **Minority class detection** | Left/right F1 > 0.50                       | Alex   |
| **Bias reduction**           | ≥ 50% score drop on average                | Daniel |
| **Semantic preservation**    | SBERT similarity ≥ 0.80 on average         | Daniel |
| **Pipeline stability**       | 100% of 50 test sentences complete         | Daniel |
| **Report quality**           | 4–6 pages, all sections, proper formatting | Joint  |
| **Presentation**             | 10 minutes, covers all required topics     | Joint  |


---

## Key Dates & Checkpoints


| Checkpoint        | Date       | Owner  | What                                     |
| ----------------- | ---------- | ------ | ---------------------------------------- |
| Week 1 Sync       | Day 7 EOD  | Joint  | Review baseline F1, rewrite samples      |
| Week 2 Smoke Test | Day 10 EOD | Joint  | Pipeline runs on 10 sentences end-to-end |
| Mid-Project Sync  | Day 11     | Joint  | Review pipeline quality, adjust scope    |
| Week 3 Evaluation | Day 19 EOD | Daniel | All 50 sentences evaluated               |
| Report Draft      | Day 20 EOD | Joint  | First complete draft ready               |
| Final Submission  | Day 21 EOD | Joint  | Code, report, slides, all artifacts      |


---

*Group 11 · DATA 641 NLP · Daniel (121033345) · Alex (121166357)*