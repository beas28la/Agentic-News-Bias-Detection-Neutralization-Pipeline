import re
import ollama

PROMPTS = {
    "left": (
        "Rewrite the following left-leaning biased sentence in neutral language, "
        "preserving all factual content but removing emotionally charged progressive framing:\n\n"
        "Original: {sentence}\n\nNeutral version:"
    ),
    "right": (
        "Rewrite the following right-leaning biased sentence in neutral language, "
        "preserving all factual content but removing emotionally charged conservative framing:\n\n"
        "Original: {sentence}\n\nNeutral version:"
    ),
    "general": (
        "Rewrite the following biased sentence in neutral, objective language "
        "while preserving all factual content:\n\n"
        "Original: {sentence}\n\nNeutral version:"
    ),
}

_PREAMBLE = re.compile(
    r"^(sure[,!]?\s*here('?s| is).*?:\s*|of course[,!]?\s*|certainly[,!]?\s*)",
    re.IGNORECASE,
)


def rewrite_biased_sentence(sentence: str, bias_type: str = "general") -> str:
    template = PROMPTS.get(bias_type, PROMPTS["general"])
    prompt = template.format(sentence=sentence)

    print(f"[rewrite] bias_type={bias_type!r}")
    print(f"[rewrite] input: {sentence}")

    resp = ollama.generate(
        model="mistral:7b-instruct-v0.3-q4_K_M",
        prompt=prompt,
        options={"temperature": 0.3, "num_predict": 200},
    )
    output = resp["response"].strip()
    output = _PREAMBLE.sub("", output).strip()

    print(f"[rewrite] output: {output}")
    return output


if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    TRAIN_CSV = Path(__file__).parent / "data/processed/train.csv"
    OUTPUT_XLSX = Path(__file__).parent / "results/rewrite_sample.xlsx"
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    biased = df[df["label"] == "biased"].head(20).copy()

    records = []
    for _, row in biased.iterrows():
        rewritten = rewrite_biased_sentence(row["sentence"], "general")
        records.append({
            "uuid": row["uuid"],
            "original_sentence": row["sentence"],
            "outlet": row.get("outlet", ""),
            "rewritten_sentence": rewritten,
        })

    out_df = pd.DataFrame(records)
    out_df.to_excel(OUTPUT_XLSX, index=False)
    print(f"\nExported {len(out_df)} rows → {OUTPUT_XLSX}")
