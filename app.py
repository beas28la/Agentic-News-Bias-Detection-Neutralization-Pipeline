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

from pipeline import run  # noqa: E402 — must come after sys.path mutation

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
