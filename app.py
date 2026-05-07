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
