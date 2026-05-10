"""
LangGraph pipeline for bias detection and neutralization.

Pipeline flow:
    Input → Bias Classifier → Rewrite (if biased) → Safety Check → Post-score → Output
                                      ↑                   ↓
                                      └── retry if sim < 0.80 and retries < 3

Usage:
    python src/pipeline.py                     # runs built-in test sentence
    from pipeline import run; run("sentence")  # from repo root with src/ on sys.path
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, TypedDict

import torch
from langgraph.graph import END, StateGraph
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Allow `from rewrite import ...` when running from repo root
sys.path.insert(0, str(Path(__file__).parent))
from rewrite import rewrite_sentence

BASE_DIR = Path(__file__).parent.parent
BERT_MODEL_DIR = BASE_DIR / "models" / "bert_classifier"

BIAS_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.70
MAX_RETRIES = 3
MAX_LENGTH = 128


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    input_sentence: str
    bias_label: Optional[str]
    bias_score: Optional[float]
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
_embedding_model = None


def _load_bert():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(str(BERT_MODEL_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(BERT_MODEL_DIR))
        _model.eval()


def _load_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Node 1: Input
# ---------------------------------------------------------------------------

def input_node(state: PipelineState) -> PipelineState:
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
# Node 2: Bias Classifier
# ---------------------------------------------------------------------------

def bias_classifier_node(state: PipelineState) -> PipelineState:
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
        logits = _model(**enc).logits

    probs = torch.softmax(logits, dim=-1).numpy()[0]
    bias_score = float(probs[1])
    bias_label = "biased" if bias_score > BIAS_THRESHOLD else "non-biased"

    return {**state, "bias_label": bias_label, "bias_score": bias_score}


# ---------------------------------------------------------------------------
# Node 3: Rewrite
# ---------------------------------------------------------------------------

_RETRY_TEMPERATURES = [0.3, 0.2, 0.1]


def rewrite_node(state: PipelineState) -> PipelineState:
    if (state.get("bias_score") or 0.0) <= BIAS_THRESHOLD:
        return {**state, "rewritten_sentence": state["input_sentence"]}

    retry_count = state["retry_count"]
    temperature = _RETRY_TEMPERATURES[min(retry_count, len(_RETRY_TEMPERATURES) - 1)]

    rewritten = rewrite_sentence(
        state["input_sentence"],
        previous_rewrite=state.get("rewritten_sentence"),
        similarity_score=state.get("similarity_score"),
        temperature=temperature,
    )
    return {
        **state,
        "rewritten_sentence": rewritten,
        "retry_count": retry_count + 1,
    }


# ---------------------------------------------------------------------------
# Helper: semantic similarity
# ---------------------------------------------------------------------------

def compute_similarity(original: str, rewritten: str) -> float:
    _load_embedding_model()
    orig_emb = _embedding_model.encode([original])
    rew_emb = _embedding_model.encode([rewritten])
    return float(cosine_similarity(orig_emb, rew_emb)[0][0])


# ---------------------------------------------------------------------------
# Node 4: Safety Check
# ---------------------------------------------------------------------------

def safety_check_node(state: PipelineState) -> PipelineState:
    original = state["input_sentence"]
    rewritten = state.get("rewritten_sentence") or state["input_sentence"]
    similarity = compute_similarity(original, rewritten)
    return {**state, "similarity_score": similarity}


# ---------------------------------------------------------------------------
# Node 5: Post-rewrite bias scoring
# ---------------------------------------------------------------------------

def post_rewrite_bias_node(state: PipelineState) -> PipelineState:
    _load_bert()

    rewritten = state.get("rewritten_sentence") or state["input_sentence"]
    enc = _tokenizer(
        rewritten,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = _model(**enc).logits

    probs = torch.softmax(logits, dim=-1).numpy()[0]
    return {**state, "new_bias_score": float(probs[1])}


# ---------------------------------------------------------------------------
# Node 6: Warning
# ---------------------------------------------------------------------------

def add_max_retry_warning(state: PipelineState) -> PipelineState:
    sim = state.get("similarity_score", 1.0)
    retries = state.get("retry_count", 0)
    if sim < SIMILARITY_THRESHOLD and retries >= MAX_RETRIES:
        return {**state, "warning": "similarity below threshold after max retries"}
    return state


# ---------------------------------------------------------------------------
# Node 7: Output
# ---------------------------------------------------------------------------

def output_node(state: PipelineState) -> PipelineState:
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
    # Non-biased sentences still pass through post_rewrite_bias so that
    # bias_score_after is always populated in the output (no special-casing downstream).
    if state["bias_score"] <= BIAS_THRESHOLD:
        return "post_rewrite_bias"
    return "rewrite"


def route_after_safety(state: PipelineState) -> str:
    sim = state.get("similarity_score", 1.0)
    retries = state.get("retry_count", 0)

    if sim >= SIMILARITY_THRESHOLD:
        return "warn"
    if retries < MAX_RETRIES:
        return "rewrite"
    return "warn"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("input", input_node)
    graph.add_node("bias_classifier", bias_classifier_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("safety_check", safety_check_node)
    graph.add_node("warn", add_max_retry_warning)
    graph.add_node("post_rewrite_bias", post_rewrite_bias_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("input")
    graph.add_edge("input", "bias_classifier")

    graph.add_conditional_edges(
        "bias_classifier",
        route_after_classifier,
        {
            "rewrite": "rewrite",
            "post_rewrite_bias": "post_rewrite_bias",
        },
    )

    graph.add_edge("rewrite", "safety_check")

    graph.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {
            "rewrite": "rewrite",
            "warn": "warn",
        },
    )

    graph.add_edge("warn", "post_rewrite_bias")
    graph.add_edge("post_rewrite_bias", "output")
    graph.add_edge("output", END)

    return graph.compile()


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


if __name__ == "__main__":
    import json
    result = run("The corrupt politician destroyed the country.")
    print(json.dumps(result, indent=2))
