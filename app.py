"""Streamlit demo frontend for the bias detection and neutralization pipeline.

Run with: streamlit run app.py
"""

from __future__ import annotations

import html
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

def _render_card(label: str, badge_text: str, body: str, *, variant: str) -> None:
    """Render one of the side-by-side result cards.

    variant must be "red" or "green". Uses unsafe_allow_html=True; body is
    html-escaped to be safe with arbitrary input.
    """
    palette = {
        "red": ("#fca5a5", "#fef2f2", "#dc2626"),
        "green": ("#86efac", "#f0fdf4", "#15803d"),
    }
    border, bg, accent = palette[variant]
    st.markdown(
        f"""
        <div style="border:1px solid {border};border-radius:8px;
                    padding:14px;background:{bg}">
          <div style="display:flex;justify-content:space-between;
                      align-items:center;margin-bottom:8px">
            <span style="font-size:11px;color:{accent};
                         text-transform:uppercase;font-weight:600">
              {html.escape(label)}
            </span>
            <span style="font-size:11px;background:{accent};color:white;
                         padding:2px 8px;border-radius:10px">
              {html.escape(badge_text)}
            </span>
          </div>
          <div style="font-size:14px;color:#111">{html.escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if st.session_state.get("result") is not None:
    from app_helpers import format_bias_drop_caption, format_score, is_biased

    result = st.session_state["result"]
    biased = is_biased(result["bias_score_before"])

    if biased:
        col_left, col_right = st.columns(2)
        with col_left:
            _render_card(
                "Original",
                f"BIASED · {format_score(result['bias_score_before'])}",
                result["original"],
                variant="red",
            )
        with col_right:
            _render_card(
                "Neutral rewrite",
                format_score(result["bias_score_after"]),
                result["rewritten"],
                variant="green",
            )
    else:
        _render_card(
            "Original",
            f"NON-BIASED · {format_score(result['bias_score_before'])}",
            result["original"],
            variant="green",
        )
        st.info("No rewrite needed.")

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
