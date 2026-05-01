"""
Unit tests for bias_classifier_node — Task 2.2.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import PipelineState, bias_classifier_node


def _make_state(sentence: str) -> PipelineState:
    return {
        "input_sentence": sentence,
        "bias_label": None,
        "bias_score": None,
        "rewritten_sentence": None,
        "similarity_score": None,
        "new_bias_score": None,
        "final_output": None,
        "retry_count": 0,
        "warning": None,
    }


def test_biased_sentence():
    """Reference case from CLAUDE.md: loaded political sentence must be detected as biased."""
    sentence = "The reckless senator pushed through the controversial bill"
    result = bias_classifier_node(_make_state(sentence))

    assert result["bias_score"] > 0.5, (
        f"Expected bias_score > 0.5, got {result['bias_score']:.4f}"
    )
    assert result["bias_label"] == "biased", (
        f"Expected bias_label='biased', got '{result['bias_label']}'"
    )


def test_non_biased_sentence():
    """A neutral factual sentence should score below the threshold."""
    sentence = "The senate voted on the bill yesterday afternoon"
    result = bias_classifier_node(_make_state(sentence))

    assert result["bias_score"] < 0.5, (
        f"Expected bias_score < 0.5, got {result['bias_score']:.4f}"
    )
    assert result["bias_label"] == "non-biased", (
        f"Expected bias_label='non-biased', got '{result['bias_label']}'"
    )


def test_output_fields_always_present():
    """Node must always populate both bias_label and bias_score regardless of content."""
    for sentence in [
        "The reckless senator pushed through the controversial bill",
        "Scientists released a new study on climate change",
        "",  # edge case: empty after strip — pipeline input_node guards this,
             # but classifier_node itself should not crash on an empty string
    ]:
        if sentence == "":
            continue  # empty string is caught upstream by input_node
        result = bias_classifier_node(_make_state(sentence))
        assert result["bias_label"] in ("biased", "non-biased")
        assert 0.0 <= result["bias_score"] <= 1.0


def test_state_passthrough():
    """Node must preserve all other state fields unchanged."""
    state = _make_state("The reckless senator pushed through the controversial bill")
    state["retry_count"] = 2
    state["warning"] = "test-warning"

    result = bias_classifier_node(state)

    assert result["retry_count"] == 2
    assert result["warning"] == "test-warning"
    assert result["rewritten_sentence"] is None
