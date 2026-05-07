"""Download and preprocess the BABE dataset for the bias detection pipeline.

BABE (Bias Annotations By Experts, Spinde et al. 2022) is the expert-annotated
successor to MBIC. It is hosted on the HuggingFace Hub at
``mediabiasgroup/BABE`` and contains ~4.1k sentences with sentence-level binary
bias labels, biased word spans, topic, source outlet, and (for a subset) an
outlet-derived political leaning.

Output files:
    data/raw/babe.csv
    data/processed/train.csv   (~85% of HF train split)
    data/processed/val.csv     (~15% of HF train split, stratified)
    data/processed/test.csv    (HF official test split, untouched)

Schema after preprocessing:
    sentence       cleaned text
    sentence_raw   original text (for the rewrite module)
    label          "biased" / "non-biased"   (from BABE `label`: 1/0)
    type           "left" / "center" / "right"  (from BABE `type`, with nulls
                   imputed from the outlet's AllSides leaning)
    outlet         source outlet
    topic          BABE topic tag
    biased_words   JSON-encoded list of biased word spans
    uuid           BABE row id
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Outlet -> political leaning (AllSides ratings)
# ---------------------------------------------------------------------------
# BABE ships a `type` column but it is null for many rows. We fall back to a
# fixed outlet -> leaning map derived from the AllSides Media Bias Chart, which
# is also the source the BABE authors used. Unknown outlets default to "center".
OUTLET_LEANING: dict[str, str] = {
    # Left
    "alternet": "left",
    "huffpost": "left",
    "huffington post": "left",
    "msnbc": "left",
    "vox": "left",
    "the new yorker": "left",
    "new yorker": "left",
    "slate": "left",
    "the daily beast": "left",
    "daily beast": "left",
    "mother jones": "left",
    "buzzfeed news": "left",
    "buzzfeed": "left",
    "the intercept": "left",
    "jacobin": "left",
    # Center
    "reuters": "center",
    "ap": "center",
    "associated press": "center",
    "usa today": "center",
    "bbc": "center",
    "bbc news": "center",
    "npr": "center",
    "abc news": "center",
    "cbs news": "center",
    "the hill": "center",
    "axios": "center",
    "bloomberg": "center",
    "christian science monitor": "center",
    # Right
    "breitbart": "right",
    "federalist": "right",
    "the federalist": "right",
    "daily wire": "right",
    "the daily wire": "right",
    "daily mail": "right",
    "daily caller": "right",
    "the daily caller": "right",
    "fox news": "right",
    "fox": "right",
    "national review": "right",
    "washington examiner": "right",
    "washington times": "right",
    "townhall": "right",
    "new york post": "right",
    "ny post": "right",
    "the american conservative": "right",
}


def infer_leaning(outlet: str | None, current_type: str | None) -> str:
    """Return left/center/right, preferring BABE's `type` when present."""
    if isinstance(current_type, str) and current_type.strip().lower() in {
        "left",
        "center",
        "right",
    }:
        return current_type.strip().lower()
    if not isinstance(outlet, str):
        return "center"
    return OUTLET_LEANING.get(outlet.strip().lower(), "center")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
NON_PRINTABLE_RE = re.compile(r"[^\x20-\x7e]")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs / non-printable chars / excess whitespace.

    We keep punctuation since the rewrite module benefits from sentence
    structure. Stopword removal is intentionally NOT applied here because the
    BERT tokenizer handles stopwords natively; if you need a TF-IDF-friendly
    version, do the stopword stripping inside the TF-IDF vectorizer instead.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = NON_PRINTABLE_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
KEEP_COLS = [
    "uuid",
    "sentence",
    "sentence_raw",
    "label",
    "type",
    "outlet",
    "topic",
    "biased_words",
]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared cleaning/normalization to a raw BABE split DataFrame."""
    df = df.rename(columns={"text": "sentence_raw"})
    df = df.dropna(subset=["sentence_raw", "label"]).reset_index(drop=True)

    df["sentence"] = df["sentence_raw"].apply(clean_text)
    df = df[df["sentence"].str.len() > 0].reset_index(drop=True)

    df["label"] = df["label"].astype(int).map({0: "non-biased", 1: "biased"})
    df["type"] = [
        infer_leaning(o, t) for o, t in zip(df["outlet"], df["type"])
    ]
    return df[KEEP_COLS]


def load_babe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download BABE from HuggingFace and return (train_df, test_df).

    Uses HuggingFace's official train/test split so evaluation is comparable
    to published baselines.
    """
    ds = load_dataset("mediabiasgroup/BABE")
    train_df = _normalize(ds["train"].to_pandas())
    test_df = _normalize(ds["test"].to_pandas())
    return train_df, test_df


def split_val_from_train(
    train_df: pd.DataFrame, val_frac: float = 0.15, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a stratified validation set out of the HF train split.

    Stratifies on (label, outlet); rare strata (<3 samples) fall back to
    label-only stratification to avoid sklearn errors.
    """
    strata = train_df["label"].astype(str) + "::" + train_df["outlet"].astype(str)
    counts = strata.value_counts()
    rare = counts[counts < 3].index
    strata = strata.where(~strata.isin(rare), other=train_df["label"].astype(str))

    train_out, val_out = train_test_split(
        train_df,
        test_size=val_frac,
        random_state=seed,
        stratify=strata,
    )
    return train_out.reset_index(drop=True), val_out.reset_index(drop=True)


def summarize(df: pd.DataFrame, name: str) -> dict:
    """Print and return a small distribution summary."""
    summary = {
        "split": name,
        "n": int(len(df)),
        "label": df["label"].value_counts().to_dict(),
        "type": df["type"].value_counts().to_dict(),
        "top_outlets": df["outlet"].value_counts().head(8).to_dict(),
    }
    print(f"\n[{name}] n={summary['n']}")
    print(f"  label: {summary['label']}")
    print(f"  type : {summary['type']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BABE for bias pipeline.")
    parser.add_argument(
        "--raw_dir", default="data/raw", help="Where to dump the full BABE CSV."
    )
    parser.add_argument(
        "--processed_dir",
        default="data/processed",
        help="Where to write train/val/test CSVs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading BABE from HuggingFace (mediabiasgroup/BABE)...")
    hf_train_df, test_df = load_babe()
    print(f"HF train split: {len(hf_train_df)} rows | HF test split: {len(test_df)} rows")

    full_path = raw_dir / "babe.csv"
    pd.concat([hf_train_df, test_df], ignore_index=True).to_csv(full_path, index=False)
    print(f"Saved full dataset: {full_path}")

    train_df, val_df = split_val_from_train(hf_train_df, val_frac=0.20, seed=args.seed)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    summary = {
        "source": "mediabiasgroup/BABE",
        "seed": args.seed,
        "note": "train/test from HF official split; val carved from HF train",
        "splits": [
            summarize(train_df, "train"),
            summarize(val_df, "val"),
            summarize(test_df, "test"),
        ],
    }
    with open(processed_dir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote split summary to {processed_dir / 'split_summary.json'}")


if __name__ == "__main__":
    main()
