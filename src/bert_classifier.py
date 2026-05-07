"""
Fine-tunes bert-base-uncased for binary bias classification (biased vs non-biased).

Usage:
    python src/bert_classifier.py    # trains and saves to models/bert_classifier/
"""

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "bert_classifier"

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
EARLY_STOPPING_PATIENCE = 2
LABEL2ID = {"non-biased": 0, "biased": 1}
ID2LABEL = {0: "non-biased", 1: "biased"}


class BiasDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    """Trainer subclass that applies class weights to cross-entropy loss."""

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = torch.tensor(self.class_weights, dtype=torch.float).to(logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_split(split: str) -> tuple[list[str], list[int]]:
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    texts = df["sentence"].tolist()
    labels = [LABEL2ID[l] for l in df["label"].tolist()]
    return texts, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1_macro = f1_score(labels, preds, average="macro")
    report = classification_report(
        labels, preds, target_names=["non-biased", "biased"], output_dict=True
    )
    return {
        "f1_macro": f1_macro,
        "f1_biased": report["biased"]["f1-score"],
        "f1_non_biased": report["non-biased"]["f1-score"],
        "precision_biased": report["biased"]["precision"],
        "recall_biased": report["biased"]["recall"],
        "precision_non_biased": report["non-biased"]["precision"],
        "recall_non_biased": report["non-biased"]["recall"],
    }


def train() -> dict:
    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

    train_dataset = BiasDataset(train_enc, train_labels)
    val_dataset = BiasDataset(val_enc, val_labels)

    # "balanced" derives weights inversely proportional to class frequency,
    # compensating for any skew between biased/non-biased without manual tuning.
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=np.array(train_labels)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        seed=42,
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    trainer.train()

    eval_results = trainer.evaluate()
    # Trainer prefixes metric keys with "eval_"; strip that prefix for MLflow logging.
    metrics = {k.replace("eval_", ""): v for k, v in eval_results.items() if "f1" in k or "precision" in k or "recall" in k}

    # Save best model and tokenizer
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    return metrics


def predict(texts: list[str]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).numpy()

    return [
        {
            "bias_label": ID2LABEL[int(np.argmax(p))],
            "bias_score": float(p[1]),  # probability of "biased"
        }
        for p in probs
    ]


def main():
    os.chdir(BASE_DIR)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bias-detection")

    with mlflow.start_run(run_name="bert-baseline"):
        mlflow.log_params(
            {
                "base_model": MODEL_NAME,
                "max_length": MAX_LENGTH,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "class_weight": "balanced",
                "task": "binary",
            }
        )

        print(f"\n=== Training BERT Binary Classifier ({MODEL_NAME}) ===")
        print(f"Target to beat: TF-IDF F1 macro = 0.7126\n")

        metrics = train()
        mlflow.log_metrics(metrics)

        print("\n=== BERT Baseline (Binary) ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        tfidf_f1 = 0.7126
        beat = metrics.get("f1_macro", 0) > tfidf_f1
        print(f"\nTF-IDF F1 macro: {tfidf_f1:.4f}")
        print(f"BERT   F1 macro: {metrics.get('f1_macro', 0):.4f}")
        print(f"Beat baseline: {'YES' if beat else 'NO'}")
        mlflow.log_metric("tfidf_f1_macro_baseline", tfidf_f1)
        mlflow.log_metric("beat_baseline", int(beat))

        # Reference unit test from CLAUDE.md
        test_sentence = "The reckless senator pushed through the controversial bill"
        result = predict([test_sentence])[0]
        print(f"\nUnit test: '{test_sentence}'")
        print(f"  bias_label={result['bias_label']}, bias_score={result['bias_score']:.4f}")
        assert result["bias_score"] > 0.5, "Unit test FAILED: bias_score should be > 0.5"
        print("  [PASS] bias_score > 0.5")

        mlflow.log_artifacts(str(MODEL_DIR), artifact_path="bert_classifier")


if __name__ == "__main__":
    main()
