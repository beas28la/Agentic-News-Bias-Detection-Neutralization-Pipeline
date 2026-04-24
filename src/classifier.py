"""
TF-IDF baseline classifier for binary bias detection (biased vs non-biased).
"""

import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models" / "tfidf_baseline"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "model.pkl"


def load_split(split: str) -> tuple[list[str], list[int]]:
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    texts = df["sentence"].tolist()
    labels = (df["label"] == "biased").astype(int).tolist()
    return texts, labels


def train() -> dict:
    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")

    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, train_labels)

    preds = model.predict(X_val)
    report = classification_report(
        val_labels, preds, target_names=["non-biased", "biased"], output_dict=True
    )

    metrics = {
        "f1_macro": f1_score(val_labels, preds, average="macro"),
        "f1_biased": report["biased"]["f1-score"],
        "f1_non_biased": report["non-biased"]["f1-score"],
        "precision_macro": precision_score(val_labels, preds, average="macro"),
        "recall_macro": recall_score(val_labels, preds, average="macro"),
        "precision_biased": report["biased"]["precision"],
        "recall_biased": report["biased"]["recall"],
        "precision_non_biased": report["non-biased"]["precision"],
        "recall_non_biased": report["non-biased"]["recall"],
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return metrics


def predict(texts: list[str]) -> list[dict]:
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    X = vectorizer.transform(texts)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]  # probability of "biased"
    return [
        {"bias_label": "biased" if p == 1 else "non-biased", "bias_score": float(s)}
        for p, s in zip(preds, probs)
    ]


def main():
    mlflow.set_experiment("bias-detection")
    with mlflow.start_run(run_name="tfidf-baseline"):
        mlflow.log_params(
            {
                "max_features": 50_000,
                "ngram_range": "(1, 2)",
                "model": "LogisticRegression",
                "task": "binary",
            }
        )

        metrics = train()
        mlflow.log_metrics(metrics)

        print("\n=== TF-IDF Baseline (Binary) ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Sanity-check reference case from CLAUDE.md
        test_sentence = "The reckless senator pushed through the controversial bill"
        result = predict([test_sentence])[0]
        print(f"\nReference check: '{test_sentence}'")
        print(f"  bias_label={result['bias_label']}, bias_score={result['bias_score']:.4f}")
        assert result["bias_score"] > 0.5, "Reference case failed: bias_score should be > 0.5"
        print("  [PASS] bias_score > 0.5")

        mlflow.log_artifact(str(VECTORIZER_PATH))
        mlflow.log_artifact(str(MODEL_PATH))


if __name__ == "__main__":
    main()
