"""
LangGraph pipeline for bias detection and neutralization — Task 1.7 / 2.2 / 2.3.

State schema and all 5 nodes:
  Input → Bias Classifier → Rewrite (if bias_score > 0.5) → Safety Check → Output
                                  ↑                               ↓
                                  └──────────── Retry Loop ───────┘
                    (if similarity_score < 0.80 and retry_count < 3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import torch
from langgraph.graph import END, StateGraph
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rewrite import rewrite_sentence

BASE_DIR = Path(__file__).parent.parent
BERT_MODEL_DIR = BASE_DIR / "models" / "bert_classifier"

BIAS_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.80
MAX_RETRIES = 3
MAX_LENGTH = 128


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    input_sentence: str
    bias_label: Optional[str]       # "biased" | "non-biased"
    bias_score: Optional[float]     # softmax P(biased)
    rewritten_sentence: Optional[str]
    similarity_score: Optional[float]
    new_bias_score: Optional[float]
    final_output: Optional[dict]
    retry_count: int
    warning: Optional[str]


# ---------------------------------------------------------------------------
# Model cache (loaded once per process)
# ---------------------------------------------------------------------------

_tokenizer = None
_model = None


def _load_bert():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(BERT_MODEL_DIR))
        _model.eval()


# ---------------------------------------------------------------------------
# Node 1: Input
# ---------------------------------------------------------------------------

def input_node(state: PipelineState) -> PipelineState:
    """Validate input and initialise counters."""
    sentence = state.get("input_sentence", "").strip()
    if not sentence:
        raise ValueError("input_sentence must be a non-empty string")
    return {
        **state,
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


# ---------------------------------------------------------------------------
# Node 2: Bias Classifier  (Task 2.2 — real implementation)
# ---------------------------------------------------------------------------

def bias_classifier_node(state: PipelineState) -> PipelineState:
    """
    Run BERT inference on input_sentence.
    Sets bias_label ("biased" | "non-biased") and bias_score (P(biased)).
    """
    _load_bert()

    sentence = state["input_sentence"]
    enc = _tokenizer(
        sentence,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = _model(**enc).logits                       # shape [1, 2]

    probs = torch.softmax(logits, dim=-1).numpy()[0]        # [P(non-biased), P(biased)]
    bias_score = float(probs[1])
    bias_label = "biased" if bias_score > BIAS_THRESHOLD else "non-biased"

    return {**state, "bias_label": bias_label, "bias_score": bias_score}


# ---------------------------------------------------------------------------
# Node 3: Rewrite  (Task 2.3 — real implementation)
# ---------------------------------------------------------------------------

def rewrite_node(state: PipelineState) -> PipelineState:
    """
    Call Mistral 7B via Ollama to neutralise the input sentence.
    Skips the LLM call when bias_score <= BIAS_THRESHOLD (routing already
    guards this, but defence-in-depth avoids unnecessary API calls).
    """
    if (state.get("bias_score") or 0.0) <= BIAS_THRESHOLD:
        return {**state, "rewritten_sentence": state["input_sentence"]}

    rewritten = rewrite_sentence(state["input_sentence"])
    return {
        **state,
        "rewritten_sentence": rewritten,
        "retry_count": state["retry_count"] + 1,
    }


# ---------------------------------------------------------------------------
# Node 4: Safety Check  (stub — implemented in Task 2.5)
# ---------------------------------------------------------------------------

def safety_check_node(state: PipelineState) -> PipelineState:
    """Stub: replaced with real SBERT similarity check in Task 2.5."""
    return {**state, "similarity_score": 1.0}


# ---------------------------------------------------------------------------
# Node 5: Output
# ---------------------------------------------------------------------------

def output_node(state: PipelineState) -> PipelineState:
    """Assemble the final JSON-serialisable output dict."""
    final_output = {
        "original": state["input_sentence"],
        "rewritten": state.get("rewritten_sentence"),
        "bias_label": state.get("bias_label"),
        "bias_score_before": state.get("bias_score"),
        "bias_score_after": state.get("new_bias_score"),
        "similarity": state.get("similarity_score"),
        "retries": state.get("retry_count", 0),
        "warning": state.get("warning"),
    }
    return {**state, "final_output": final_output}


# ---------------------------------------------------------------------------
# Routing conditions
# ---------------------------------------------------------------------------

def route_after_classifier(state: PipelineState) -> str:
    if state["bias_score"] <= BIAS_THRESHOLD:
        return "output"
    return "rewrite"


def route_after_safety(state: PipelineState) -> str:
    sim = state.get("similarity_score", 1.0)
    retries = state.get("retry_count", 0)

    if sim >= SIMILARITY_THRESHOLD:
        return "output"
    if retries < MAX_RETRIES:
        return "rewrite"
    # Max retries hit — force through with warning
    return "output"


def add_max_retry_warning(state: PipelineState) -> PipelineState:
    """Called by output_node path after max retries; adds warning to state."""
    sim = state.get("similarity_score", 1.0)
    retries = state.get("retry_count", 0)
    if sim < SIMILARITY_THRESHOLD and retries >= MAX_RETRIES:
        return {**state, "warning": "similarity below threshold after max retries"}
    return state


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("input", input_node)
    graph.add_node("bias_classifier", bias_classifier_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("safety_check", safety_check_node)
    graph.add_node("warn", add_max_retry_warning)
    graph.add_node("output", output_node)

    graph.set_entry_point("input")
    graph.add_edge("input", "bias_classifier")

    graph.add_conditional_edges(
        "bias_classifier",
        route_after_classifier,
        {"output": "output", "rewrite": "rewrite"},
    )

    graph.add_edge("rewrite", "safety_check")

    graph.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {"output": "warn", "rewrite": "rewrite"},
    )

    graph.add_edge("warn", "output")
    graph.add_edge("output", END)

    return graph.compile()


# Compiled graph (import-time; reuse across calls)
pipeline = build_graph()


def run(sentence: str) -> dict:
    """Run the full pipeline on a single sentence and return the output dict."""
    initial_state: PipelineState = {
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
    final_state = pipeline.invoke(initial_state)
    return final_state["final_output"]
