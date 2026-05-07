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
