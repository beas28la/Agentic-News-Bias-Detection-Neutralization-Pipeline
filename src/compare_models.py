"""
Task 2.4 — BERT vs TF-IDF comparison on the validation set.
Logs a single MLflow run with side-by-side metrics from both models.
"""

import os
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
TFIDF_DIR = BASE_DIR / "models" / "tfidf_baseline"
BERT_DIR  = BASE_DIR / "models" / "bert_classifier"
MAX_LENGTH = 128
BATCH_SIZE = 32


def load_val() -> tuple[list[str], list[int]]:
    df = pd.read_csv(DATA_DIR / "val.csv")
    texts  = df["sentence"].tolist()
    labels = (df["label"] == "biased").astype(int).tolist()
    return texts, labels


def eval_tfidf(texts: list[str], labels: list[int]) -> dict:
    with open(TFIDF_DIR / "vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    with open(TFIDF_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    X = vec.transform(texts)
    preds = model.predict(X)
    return _metrics(labels, preds, prefix="tfidf")


def eval_bert(texts: list[str], labels: list[int]) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(str(BERT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR))
    model.eval()

    all_preds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        all_preds.extend(torch.argmax(logits, dim=-1).tolist())

    return _metrics(labels, all_preds, prefix="bert")


def _metrics(labels: list[int], preds: list[int], prefix: str) -> dict:
    report = classification_report(
        labels, preds,
        target_names=["non-biased", "biased"],
        output_dict=True,
    )
    return {
        f"{prefix}_f1_macro":            f1_score(labels, preds, average="macro"),
        f"{prefix}_f1_biased":           report["biased"]["f1-score"],
        f"{prefix}_f1_non_biased":       report["non-biased"]["f1-score"],
        f"{prefix}_precision_biased":    report["biased"]["precision"],
        f"{prefix}_recall_biased":       report["biased"]["recall"],
        f"{prefix}_precision_non_biased":report["non-biased"]["precision"],
        f"{prefix}_recall_non_biased":   report["non-biased"]["recall"],
    }


def print_table(tfidf: dict, bert: dict) -> None:
    metrics = [
        ("F1 macro",           "tfidf_f1_macro",            "bert_f1_macro"),
        ("F1 biased",          "tfidf_f1_biased",           "bert_f1_biased"),
        ("F1 non-biased",      "tfidf_f1_non_biased",       "bert_f1_non_biased"),
        ("Precision biased",   "tfidf_precision_biased",    "bert_precision_biased"),
        ("Recall biased",      "tfidf_recall_biased",       "bert_recall_biased"),
        ("Precision non-bias", "tfidf_precision_non_biased","bert_precision_non_biased"),
        ("Recall non-bias",    "tfidf_recall_non_biased",   "bert_recall_non_biased"),
    ]
    combined = {**tfidf, **bert}
    print(f"\n{'Metric':<22} {'TF-IDF':>8}  {'BERT':>8}  {'Delta':>8}")
    print("-" * 52)
    for label, tk, bk in metrics:
        tv, bv = combined[tk], combined[bk]
        print(f"{label:<22} {tv:>8.4f}  {bv:>8.4f}  {bv-tv:>+8.4f}")


def main():
    os.chdir(BASE_DIR)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bias-detection")

    texts, labels = load_val()
    print(f"Val set: {len(texts)} sentences")

    print("Evaluating TF-IDF...")
    tfidf_metrics = eval_tfidf(texts, labels)

    print("Evaluating BERT...")
    bert_metrics = eval_bert(texts, labels)

    print_table(tfidf_metrics, bert_metrics)

    with mlflow.start_run(run_name="bert-vs-tfidf-comparison"):
        mlflow.log_params({
            "val_size": len(texts),
            "tfidf_model": "LogisticRegression + TF-IDF(50k, 1-2gram)",
            "bert_model":  "bert-base-uncased fine-tuned (3 epochs, lr=2e-5)",
            "split": "val",
        })
        mlflow.log_metrics({**tfidf_metrics, **bert_metrics})
        # Delta metrics for quick glance in UI
        mlflow.log_metrics({
            "delta_f1_macro":      bert_metrics["bert_f1_macro"]      - tfidf_metrics["tfidf_f1_macro"],
            "delta_f1_biased":     bert_metrics["bert_f1_biased"]     - tfidf_metrics["tfidf_f1_biased"],
            "delta_f1_non_biased": bert_metrics["bert_f1_non_biased"] - tfidf_metrics["tfidf_f1_non_biased"],
        })

    print("\nMLflow run 'bert-vs-tfidf-comparison' logged.")


if __name__ == "__main__":
    main()
