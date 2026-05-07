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
