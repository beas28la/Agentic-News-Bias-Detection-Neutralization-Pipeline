# Streamlit Demo Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-page Streamlit demo (`app.py`) that wraps the existing `pipeline.run()` so a user can type a news sentence, click Analyze, and see the bias verdict, the neutral rewrite, and the supporting metrics side-by-side.

**Architecture:** One Python process. Streamlit serves the page in the same process that holds the cached BERT, SBERT, and Ollama clients. `app.py` lives at the project root and imports `run` from `src/pipeline.py` after prepending `src/` to `sys.path`. No HTTP layer, no new pipeline code — the frontend is a pure consumer.

**Tech Stack:** Streamlit (new), existing `src/pipeline.py` (BERT + Mistral 7B via Ollama + SBERT), pytest for the small set of pure helpers.

**Reference spec:** `docs/superpowers/specs/2026-05-07-streamlit-demo-design.md`

---

## File structure

```
app.py                     # NEW — Streamlit entry point
app_helpers.py             # NEW — pure-logic helpers (formatters, derived values)
tests/test_app_helpers.py  # NEW — unit tests for helpers
requirements.txt           # MODIFY — add streamlit
src/pipeline.py            # unchanged
```

`app_helpers.py` keeps the testable logic out of `app.py` (which is mostly Streamlit calls that aren't worth unit-testing). Everything in `app.py` is verified by running the app manually.

---

### Task 1: Add Streamlit dependency

**Files:**
- Modify: `requirements.txt` (append after line 30, in the visualization section)

- [ ] **Step 1: Add streamlit to requirements.txt**

Append at the end of the existing file (after `jupyter>=1.0.0`, before the testing section), or in a new section:

```
# ---- Demo frontend ----
streamlit>=1.36.0
```

- [ ] **Step 2: Install it**

Run: `pip install "streamlit>=1.36.0"`
Expected: `Successfully installed streamlit-1.x.x ...`

- [ ] **Step 3: Verify it imports**

Run: `python -c "import streamlit; print(streamlit.__version__)"`
Expected: A version string like `1.36.0` (or higher).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "add streamlit dependency for demo frontend"
```

---

### Task 2: Pure helpers with TDD

`app_helpers.py` holds three small pure functions used by the UI. We write them test-first because they're the only logic in the project that *can* be unit-tested without spinning Streamlit.

**Files:**
- Create: `tests/test_app_helpers.py`
- Create: `app_helpers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_app_helpers.py`:

```python
"""Unit tests for app_helpers — pure formatting / derivation functions."""

import pytest

from app_helpers import (
    format_score,
    format_bias_drop_caption,
    is_biased,
)


class TestFormatScore:
    def test_rounds_to_two_decimals(self):
        assert format_score(0.8734) == "0.87"

    def test_pads_short_values(self):
        assert format_score(0.2) == "0.20"

    def test_handles_zero(self):
        assert format_score(0.0) == "0.00"

    def test_handles_one(self):
        assert format_score(1.0) == "1.00"


class TestFormatBiasDropCaption:
    def test_typical_drop(self):
        assert format_bias_drop_caption(0.87, 0.21) == "0.87 → 0.21"

    def test_no_change(self):
        assert format_bias_drop_caption(0.30, 0.30) == "0.30 → 0.30"

    def test_increase(self):
        # rare but possible — still format honestly
        assert format_bias_drop_caption(0.40, 0.55) == "0.40 → 0.55"


class TestIsBiased:
    def test_above_threshold(self):
        assert is_biased(0.51) is True

    def test_at_threshold(self):
        # threshold of 0.5 — strict greater-than per pipeline.py
        assert is_biased(0.50) is False

    def test_below_threshold(self):
        assert is_biased(0.49) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_app_helpers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app_helpers'`.

- [ ] **Step 3: Write the minimal implementation**

Create `app_helpers.py`:

```python
"""Pure-logic helpers for the Streamlit demo (app.py).

Kept separate from app.py so the formatting and threshold logic can be
unit-tested without spinning up Streamlit.
"""

from __future__ import annotations

BIAS_THRESHOLD = 0.5  # mirrors src/pipeline.py


def format_score(value: float) -> str:
    """Format a probability as a 2-decimal string."""
    return f"{value:.2f}"


def format_bias_drop_caption(before: float, after: float) -> str:
    """Caption shown under the Bias drop metric tile."""
    return f"{format_score(before)} → {format_score(after)}"


def is_biased(score: float) -> bool:
    """Strict-greater-than threshold to match pipeline.py classifier logic."""
    return score > BIAS_THRESHOLD
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app_helpers.py -v`
Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_app_helpers.py app_helpers.py
git commit -m "add app_helpers with formatting and threshold helpers"
```

---

### Task 3: Skeleton app — header, input, Analyze button

Get the page to render with no result section yet. Verify Streamlit boots and the input widgets work.

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write the skeleton**

Create `app.py`:

```python
"""Streamlit demo frontend for the bias detection and neutralization pipeline.

Run with: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make src/ importable so we can use pipeline.run directly.
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="News Bias Detection & Neutralization",
    layout="centered",
)

st.title("News Bias Detection & Neutralization")

if "sentence" not in st.session_state:
    st.session_state["sentence"] = ""

sentence = st.text_area(
    "News sentence",
    key="sentence",
    height=80,
    placeholder="Paste a news sentence to analyze...",
)

analyze_clicked = st.button(
    "Analyze",
    type="primary",
    disabled=not sentence.strip(),
)
```

- [ ] **Step 2: Run the app and visually verify**

Run: `streamlit run app.py`
Expected:
- Browser opens to `http://localhost:8501`.
- Page shows the title, an empty text area labelled "News sentence", and a disabled "Analyze" button.
- Typing in the text area enables the button.

Stop the server with Ctrl-C when done.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "add streamlit app skeleton with title and input"
```

---

### Task 4: Wire pipeline.run() with spinner and error handling

Calling Analyze should run the pipeline and stash the result in session state. We don't render the result panels yet — just verify the call works and errors are caught.

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the pipeline call**

Edit `app.py`. After the `import streamlit as st` line (and the `sys.path` block), add:

```python
from pipeline import run  # noqa: E402  — must come after sys.path mutation
```

Below the `analyze_clicked = ...` line, add:

```python
if analyze_clicked:
    with st.spinner("Running pipeline..."):
        try:
            st.session_state["result"] = run(sentence.strip())
            st.session_state["error"] = None
        except Exception as exc:  # noqa: BLE001 — surface anything to the UI
            st.session_state["result"] = None
            st.session_state["error"] = exc

if st.session_state.get("error") is not None:
    err = st.session_state["error"]
    msg = f"{type(err).__name__}: {err}"
    if "ollama" in msg.lower() or "connection" in msg.lower():
        st.error(
            f"{msg}\n\n"
            "Make sure `ollama serve` is running and the "
            "`mistral:7b-instruct-v0.3-q4_K_M` model is pulled."
        )
    else:
        st.error(msg)

if st.session_state.get("result") is not None:
    st.json(st.session_state["result"])  # temporary — replaced in Task 5
```

- [ ] **Step 2: Run the app and visually verify (happy path)**

Pre-condition: `ollama serve` is running and the Mistral model is pulled.

Run: `streamlit run app.py`
Type: `The reckless senator pushed through the controversial bill.`
Click Analyze.

Expected:
- Spinner shows "Running pipeline...".
- After the call completes, a JSON dump appears with keys `original`, `rewritten`, `bias_label`, `bias_score_before`, `bias_score_after`, `similarity`, `retries`, `warning`.
- `bias_label` is `"biased"` and `rewritten` differs from `original`.

- [ ] **Step 3: Verify error handling (Ollama down)**

Stop Ollama (`pkill -f "ollama serve"` or stop the app it's running in).
With the Streamlit app still running, click Analyze again.
Expected: A red error card with a message containing "Ollama" and the hint about `ollama serve`.

Restart Ollama before continuing (`ollama serve &`).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "wire pipeline.run with spinner and error handling"
```

---

### Task 5: Side-by-side result panels (biased + non-biased branches)

Replace the temporary JSON dump with the two-panel layout described in the spec. Handle both the biased branch (red original / green rewrite) and the non-biased branch (single green panel + "No rewrite needed" message).

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace the JSON dump with result panels**

In `app.py`, replace the `st.json(st.session_state["result"])  # temporary — replaced in Task 5` line with:

```python
if st.session_state.get("result") is not None:
    from app_helpers import format_score, is_biased

    result = st.session_state["result"]
    biased = is_biased(result["bias_score_before"])

    if biased:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(
                f"""
                <div style="border:1px solid #fca5a5;border-radius:8px;
                            padding:14px;background:#fef2f2">
                  <div style="display:flex;justify-content:space-between;
                              align-items:center;margin-bottom:8px">
                    <span style="font-size:11px;color:#dc2626;
                                 text-transform:uppercase;font-weight:600">
                      Original
                    </span>
                    <span style="font-size:11px;background:#dc2626;color:white;
                                 padding:2px 8px;border-radius:10px">
                      BIASED · {format_score(result["bias_score_before"])}
                    </span>
                  </div>
                  <div style="font-size:14px;color:#111">{result["original"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_right:
            st.markdown(
                f"""
                <div style="border:1px solid #86efac;border-radius:8px;
                            padding:14px;background:#f0fdf4">
                  <div style="display:flex;justify-content:space-between;
                              align-items:center;margin-bottom:8px">
                    <span style="font-size:11px;color:#15803d;
                                 text-transform:uppercase;font-weight:600">
                      Neutral rewrite
                    </span>
                    <span style="font-size:11px;background:#15803d;color:white;
                                 padding:2px 8px;border-radius:10px">
                      {format_score(result["bias_score_after"])}
                    </span>
                  </div>
                  <div style="font-size:14px;color:#111">{result["rewritten"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"""
            <div style="border:1px solid #86efac;border-radius:8px;
                        padding:14px;background:#f0fdf4">
              <div style="display:flex;justify-content:space-between;
                          align-items:center;margin-bottom:8px">
                <span style="font-size:11px;color:#15803d;
                             text-transform:uppercase;font-weight:600">
                  Original
                </span>
                <span style="font-size:11px;background:#15803d;color:white;
                             padding:2px 8px;border-radius:10px">
                  NON-BIASED · {format_score(result["bias_score_before"])}
                </span>
              </div>
              <div style="font-size:14px;color:#111">{result["original"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("No rewrite needed.")
```

- [ ] **Step 2: Visually verify biased branch**

Run: `streamlit run app.py`
Type: `The reckless senator pushed through the controversial bill.`
Click Analyze.
Expected:
- Red "Original" card on the left with `BIASED · 0.xx` badge.
- Green "Neutral rewrite" card on the right with the rewritten sentence and post-rewrite score.

- [ ] **Step 3: Visually verify non-biased branch**

In the same running app, replace the input with a neutral sentence:
`The senate passed the infrastructure bill on Tuesday.`
Click Analyze.
Expected:
- Single green card showing the original sentence with `NON-BIASED · 0.xx` badge.
- An info banner: "No rewrite needed." No second column.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "add side-by-side result panels for biased and non-biased branches"
```

---

### Task 6: Metric strip and warning banner

Three `st.metric` tiles — Semantic similarity, Bias drop, Retries — plus a yellow warning banner when the pipeline reports `warning`.

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the metric strip and warning**

In `app.py`, append the following inside the `if st.session_state.get("result") is not None:` block (after the panels, before its closing scope):

```python
    from app_helpers import format_bias_drop_caption  # noqa: E402

    if result.get("warning"):
        st.warning(result["warning"])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    bias_drop = result["bias_score_before"] - result["bias_score_after"]
    m1.metric(
        "Semantic similarity",
        format_score(result["similarity"]),
        help="≥ 0.80 threshold",
    )
    m2.metric(
        "Bias drop",
        f"{bias_drop:+.2f}",
        help=format_bias_drop_caption(
            result["bias_score_before"], result["bias_score_after"]
        ),
    )
    m3.metric(
        "Retries",
        str(result.get("retries", 0)),
        help="of 3 max",
    )
```

(Keep this block inside the same `if result is not None` scope as Task 5's panels.)

- [ ] **Step 2: Visually verify the metrics show**

Run: `streamlit run app.py`
Type: `The corrupt politician destroyed the country.`
Click Analyze.
Expected:
- Two result panels (red / green) as before.
- Below them, three metric tiles labelled "Semantic similarity", "Bias drop", "Retries".
- Hovering each shows its caption ("≥ 0.80 threshold", "0.xx → 0.xx", "of 3 max").
- For a typical biased sentence, similarity is ≥ 0.80 and bias drop is a negative number like `-0.6X`.

- [ ] **Step 3: (Optional) Verify the warning banner**

Triggering a real warning requires inputs where the pipeline can't find a high-similarity rewrite within 3 retries — hard to do reliably. Instead, do a one-line manual fault-injection: temporarily edit `src/pipeline.py` `add_max_retry_warning` to always set `warning="test warning"`, run the app, confirm the yellow banner shows, then revert the edit.

Skip this step if you trust the conditional logic.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "add metric strip and warning banner to demo"
```

---

### Task 7: Example sentence buttons

Three quick-fill buttons that populate the text area and immediately run the pipeline. Cover one biased-left, one biased-right, one neutral example.

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Apply pending sentence at top of script (before the text_area widget)**

Streamlit forbids assigning to `st.session_state["sentence"]` after the widget keyed on `"sentence"` has been instantiated. So example clicks set a *different* key (`pending_sentence`), and we apply it at the top of the script — before the text area renders.

In `app.py`, replace the `if "sentence" not in st.session_state: ...` block with:

```python
# Apply any pending sentence queued by an example button (must run before
# the text_area widget is instantiated, otherwise Streamlit raises).
if "pending_sentence" in st.session_state:
    st.session_state["sentence"] = st.session_state.pop("pending_sentence")

if "sentence" not in st.session_state:
    st.session_state["sentence"] = ""
```

- [ ] **Step 2: Add the example buttons**

Immediately after the `analyze_clicked = st.button(...)` line, add:

```python
EXAMPLES = {
    "Biased (left-leaning)": (
        "The reckless senator pushed through the controversial bill."
    ),
    "Biased (right-leaning)": (
        "Activist judges legislated from the bench, ignoring the will of the people."
    ),
    "Neutral": (
        "The senate passed the infrastructure bill on Tuesday."
    ),
}

st.caption("Or try an example:")
ex_cols = st.columns(len(EXAMPLES))
example_clicked = None
for col, (label, text) in zip(ex_cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        example_clicked = text

if example_clicked is not None:
    st.session_state["pending_sentence"] = example_clicked
    st.session_state["pending_run"] = example_clicked
    st.rerun()
```

- [ ] **Step 3: Make the analyze block also fire on pending example runs**

Replace the existing Task-4 `if analyze_clicked:` block with:

```python
pending = st.session_state.pop("pending_run", None)
if analyze_clicked or pending is not None:
    target = pending if pending is not None else sentence.strip()
    with st.spinner("Running pipeline..."):
        try:
            st.session_state["result"] = run(target)
            st.session_state["error"] = None
        except Exception as exc:  # noqa: BLE001
            st.session_state["result"] = None
            st.session_state["error"] = exc
```

The rest of the file is unchanged.

- [ ] **Step 4: Visually verify each example**

Run: `streamlit run app.py`

Click "Biased (left-leaning)". Expected: text area fills with the sentence, spinner runs, red+green panels appear with the rewrite.

Click "Biased (right-leaning)". Expected: same flow, red+green panels.

Click "Neutral". Expected: single green panel with "NON-BIASED" badge and "No rewrite needed." info.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "add one-click example sentences for the demo"
```

---

### Task 8: Warmup so the first user click isn't slow

The first BERT + SBERT load takes several seconds. Force it at startup so the demo feels snappy when shown to an audience.

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the warmup**

In `app.py`, after the `from pipeline import run` line, add:

```python
@st.cache_resource(show_spinner="Loading models...")
def _warmup():
    """Force BERT + SBERT to load once when the page first renders."""
    return run("warmup.")


_warmup()
```

- [ ] **Step 2: Visually verify cold start behavior**

Stop any running Streamlit instance. Run: `streamlit run app.py`
Expected on first load:
- Browser shows "Loading models..." for several seconds.
- Then the input/Analyze UI renders.
- Subsequent Analyze clicks are fast (no model reload).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "warm up models at startup so first analyze is fast"
```

---

### Task 9: README run instructions

A few lines at the end of `README.md` so anyone cloning the repo knows how to launch the demo.

**Files:**
- Modify: `README.md` (append)

- [ ] **Step 1: Append the demo section**

Add to the end of `README.md`:

```markdown

## Demo (Streamlit frontend)

A single-page demo that wraps the full pipeline.

```bash
# 1. install demo dep (already in requirements.txt)
pip install -r requirements.txt

# 2. make sure Ollama is running with the rewrite model
ollama serve &
ollama pull mistral:7b-instruct-v0.3-q4_K_M

# 3. launch the app
streamlit run app.py
```

Open http://localhost:8501. Type a sentence or click an example.
```

- [ ] **Step 2: Verify the markdown renders**

Run: `grep -A 1 "Demo (Streamlit frontend)" README.md`
Expected: the heading and the next line print.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "document streamlit demo launch in README"
```

---

## Verification checklist (run after Task 9)

- [ ] `pytest tests/test_app_helpers.py -v` — all helper tests pass.
- [ ] `streamlit run app.py` boots without errors.
- [ ] First page load shows "Loading models..." then the input UI.
- [ ] Each of the three example buttons produces the expected branch (biased-left → red+green, biased-right → red+green, neutral → green-only).
- [ ] Free-text biased input produces a rewrite and a positive bias drop.
- [ ] Stopping Ollama mid-session and clicking Analyze produces the red error card with the Ollama hint.
- [ ] No edits to `src/pipeline.py`, `src/rewrite.py`, or any model checkpoint.
