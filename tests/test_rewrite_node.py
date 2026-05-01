"""
Unit tests for the rewrite node — Task 2.3.
Preamble-stripping and skip-logic tests are offline (no Ollama needed).
The live rewrite test calls Ollama and is skipped if the server is unreachable.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rewrite import _clean, rewrite_sentence
from pipeline import PipelineState, rewrite_node, BIAS_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(sentence: str, bias_score: float, retry_count: int = 0) -> PipelineState:
    return {
        "input_sentence": sentence,
        "bias_label": "biased" if bias_score > BIAS_THRESHOLD else "non-biased",
        "bias_score": bias_score,
        "rewritten_sentence": None,
        "similarity_score": None,
        "new_bias_score": None,
        "final_output": None,
        "retry_count": retry_count,
        "warning": None,
    }


def _ollama_available() -> bool:
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Preamble stripping (offline — no LLM needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected_start", [
    ("Sure! Here is the neutral version: The senator voted.", "The senator voted."),
    ("Sure, here's a neutral version:\nThe senator voted.", "The senator voted."),
    ("Of course! The senator voted.", "The senator voted."),
    ("Certainly! The senator voted.", "The senator voted."),
    ("Here is a rewrite: The senator voted.", "The senator voted."),
    ("Neutral version: The senator voted.", "The senator voted."),
    ('"The senator voted."', "The senator voted."),
    # No preamble — should pass through unchanged
    ("The senator voted.", "The senator voted."),
])
def test_preamble_stripping(raw, expected_start):
    assert _clean(raw) == expected_start, f"_clean({raw!r}) → {_clean(raw)!r}"


# ---------------------------------------------------------------------------
# Node skip logic (offline)
# ---------------------------------------------------------------------------

def test_rewrite_skipped_when_not_biased(monkeypatch):
    """Node must not call Ollama when bias_score <= threshold."""
    called = []

    def _fake_rewrite(sentence):
        called.append(sentence)
        return "should not be called"

    monkeypatch.setattr("pipeline.rewrite_sentence", _fake_rewrite)

    state = _state("Scientists released a study.", bias_score=0.2)
    result = rewrite_node(state)

    assert called == [], "rewrite_sentence should not be called for non-biased input"
    assert result["rewritten_sentence"] == "Scientists released a study."
    assert result["retry_count"] == 0   # counter must NOT increment on skip


def test_retry_count_increments_on_rewrite(monkeypatch):
    """retry_count must go up by 1 each time the rewrite node fires."""
    monkeypatch.setattr("pipeline.rewrite_sentence", lambda s: "neutral version")

    state = _state("The reckless senator pushed through the bill.", bias_score=0.9, retry_count=1)
    result = rewrite_node(state)

    assert result["retry_count"] == 2
    assert result["rewritten_sentence"] == "neutral version"


def test_state_fields_preserved(monkeypatch):
    """Non-rewrite state fields must be unchanged after the node runs."""
    monkeypatch.setattr("pipeline.rewrite_sentence", lambda s: "neutral")

    state = _state("The reckless senator pushed the bill.", bias_score=0.85)
    state["warning"] = "pre-existing"
    result = rewrite_node(state)

    assert result["input_sentence"] == state["input_sentence"]
    assert result["bias_score"] == 0.85
    assert result["warning"] == "pre-existing"


# ---------------------------------------------------------------------------
# Live Ollama integration (skipped if server unreachable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _ollama_available(), reason="Ollama not reachable")
def test_live_rewrite_produces_non_empty_output():
    sentence = "The reckless senator ruthlessly pushed through the deeply controversial bill."
    result = rewrite_sentence(sentence)
    assert isinstance(result, str)
    assert len(result) > 10, f"Rewrite unexpectedly short: {result!r}"
    # Must not still contain obvious preamble artifacts
    assert not result.lower().startswith("sure"), f"Preamble not stripped: {result!r}"
    assert not result.lower().startswith("of course"), f"Preamble not stripped: {result!r}"
    print(f"\n  original : {sentence}")
    print(f"  rewritten: {result}")
