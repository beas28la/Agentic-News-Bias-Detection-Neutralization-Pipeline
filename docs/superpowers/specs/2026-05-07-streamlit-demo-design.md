# Streamlit Demo Frontend — Design

**Date:** 2026-05-07
**Author:** brainstorm session
**Status:** Approved, pending implementation plan

## Goal

A single-page Streamlit app that demos the existing bias detection and neutralization pipeline. A user types a news sentence, clicks Analyze, and sees:

- Whether the sentence is biased and the bias score
- The neutral rewrite (when biased)
- Semantic similarity, bias drop, and retry count

The frontend is a pure consumer of `src/pipeline.py`. **No changes to the existing pipeline, models, or rewrite logic.** BERT stays for classification, Mistral 7B (Ollama) stays for rewriting, SBERT stays for similarity.

## Non-goals

- No backend HTTP layer — `pipeline.run()` is imported directly.
- No Claude API / Opus integration. The pipeline keeps its current models.
- No authentication, persistence, history, or analytics.
- No production deployment concerns — this is a local class demo.

## Architecture

```
Browser ──HTTP──> Streamlit (app.py) ──import──> src/pipeline.py.run(sentence)
                                                       │
                                                       ├─ BERT (cached)
                                                       ├─ Mistral 7B / Ollama
                                                       └─ SBERT (cached)
```

One process. Streamlit serves the page; the same Python process holds the model objects in memory between requests.

## File layout

```
app.py                 # NEW — Streamlit entry point (project root)
src/pipeline.py        # unchanged
src/rewrite.py         # unchanged
src/bert_classifier.py # unchanged
```

`app.py` lives at the project root so it can `from pipeline import run` after prepending `src/` to `sys.path` — matching the convention already used inside `pipeline.py`.

## UI specification

### Header
- Project title only (one line).
- No subtitle, no team or model attribution.

### Input section
- One `st.text_area` labeled "News sentence" (height ~80px, placeholder text).
- Primary `st.button("Analyze")`. Disabled when the text area is empty or whitespace-only.
- A row of three example sentences rendered as small buttons. Clicking one populates the text area via `st.session_state` and triggers analysis. Examples cover one biased-left, one biased-right, and one neutral sentence so the demo can show all three branches.

### Result section (rendered only after Analyze)

**Two panels side-by-side** (Streamlit `st.columns(2)`):

- **Original (red)** — sentence text + badge showing `BIASED` or `NON-BIASED` and the `bias_score_before` rounded to 2 decimals.
- **Rewrite (green)** — rewritten sentence + badge showing `bias_score_after`.

When the input is non-biased (`bias_score_before <= 0.5`):
- Only the original panel renders, styled green with a `NON-BIASED` badge.
- The rewrite panel is replaced by an info message: "No rewrite needed."

**Metric strip** below the panels — three `st.metric` cards. The strip and the warning banner are rendered **only on the biased branch** (see "Why" below):

| Tile | Value | Caption |
|------|-------|---------|
| Semantic similarity | `similarity_score` (2 decimals) | "≥ 0.80 threshold" |
| Bias drop | `before − after` (signed, 2 decimals) | "{before} → {after}" |
| Retries | `retries` | "of 3 max" |

When `warning` is present (max retries hit, similarity below threshold), show a yellow `st.warning` banner above the metric strip with the warning text. The warning can only fire on the biased branch (it's set by `add_max_retry_warning` after `safety_check_node`, both of which are skipped for non-biased input), so co-locating it inside the biased branch matches pipeline reality.

**Why the metrics are biased-only:** the pipeline's `route_after_classifier` routes non-biased inputs directly to `post_rewrite_bias`, skipping `safety_check_node`. So for non-biased input, `similarity_score` is `None`, `retries` is `0`, and `bias_score_before − bias_score_after` is the trivial difference of scoring the same sentence twice. None of these are meaningful, and rendering them would crash on `format_score(None)`.

### Loading state

While `pipeline.run()` is executing, show `st.spinner("Running pipeline...")`. The first call is slow (BERT + SBERT cold load); subsequent calls are fast because of caching.

### Error states

- **Ollama unreachable / Mistral call fails:** caught at the `pipeline.run()` boundary. Show `st.error` with the exception type and a hint: "Make sure `ollama serve` is running and the `mistral:7b-instruct-v0.3-q4_K_M` model is pulled."
- **Any other unexpected exception:** shown via `st.error` with the message.

## Backend integration

```python
# app.py (sketch)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from pipeline import run  # noqa: E402

@st.cache_resource
def warmup():
    """Force BERT + SBERT load once at startup so the first user request
    isn't slow. Calls run() with a short throwaway sentence."""
    run("warmup.")

# ... UI code ...
```

Model caching happens inside `pipeline.py` already (module-level `_tokenizer`, `_model`, `_embedding_model`). The `warmup()` call ensures those caches are populated before the first user request, hiding the cold-start latency from the demo.

## Data flow per request

1. User types or clicks an example → `st.session_state["sentence"]` updates.
2. User clicks Analyze.
3. `result = run(sentence)` — returns the `final_output` dict from `pipeline.py`.
4. `app.py` reads from the dict:
   - `bias_label`, `bias_score_before` → original panel
   - `rewritten`, `bias_score_after` → rewrite panel (skipped if non-biased)
   - `similarity`, `bias_score_before − bias_score_after`, `retries` → metric strip
   - `warning` → optional yellow banner
5. UI re-renders.

## Run command

```
streamlit run app.py
```

Pre-conditions:
- `pip install streamlit` (add to `requirements.txt`)
- BERT checkpoint present at `models/bert_classifier/`
- `ollama serve` running, `mistral:7b-instruct-v0.3-q4_K_M` model pulled

## Out of scope (explicit)

- Multi-sentence batch input
- Comparison view between BERT and another classifier
- Editable rewrite / human feedback collection
- MLflow logging from the UI
- Deployment to a public URL
