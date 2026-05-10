"""
LLM rewrite module — Task 2.3.
Uses Mistral 7B via Ollama to neutralise biased sentences.
"""

import logging
import re

import ollama

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [rewrite] %(message)s",
    datefmt="%H:%M:%S",
)

OLLAMA_MODEL = "mistral:7b-instruct-v0.3-q4_K_M"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.3

# Single general prompt template (Task A — binary classification only)
_SYSTEM_PROMPT = (
    "You are an expert editor. Your task is to rewrite a potentially biased news sentence "
    "into neutral, objective language. Preserve all factual content exactly. "
    "Remove emotionally charged words, loaded framing, and one-sided language. "
    "Return ONLY the rewritten sentence with no explanation, no preamble, and no quotation marks."
)

_USER_TEMPLATE = "Rewrite this sentence in neutral language:\n\n{sentence}"

# Used on retries when the previous rewrite drifted too far semantically
_RETRY_TEMPLATE = (
    "Your previous rewrite changed the meaning too much (similarity: {similarity:.0%}).\n\n"
    "Original: {original}\n"
    "Previous rewrite: {previous}\n\n"
    "Rewrite the original sentence again in neutral language, staying much closer to the "
    "original meaning and wording. Only change biased or emotionally charged words."
)

# Strips common LLM preambles: "Sure! Here is...", "Of course, ...", "Certainly! ..."
_PREAMBLE_RE = re.compile(
    r"^(sure[,!]?\s*(here('?s| is)[^:]*:\s*)?|of course[,!]?\s*|certainly[,!]?\s*"
    r"|here('?s| is)[^:]*:\s*|neutral\s+version\s*:\s*)",
    re.IGNORECASE,
)

# Strips leading label artifacts like 'Neutral version: "...' or "Rewritten: ..."
_LABEL_RE = re.compile(r"^[A-Za-z ]+:\s*", re.IGNORECASE)

# Strips surrounding quotes left behind after preamble removal
_QUOTE_RE = re.compile(r'^["\']|["\']$')


def _clean(text: str) -> str:
    """Remove preamble, label prefixes, and surrounding quotes from LLM output."""
    text = text.strip()
    text = _PREAMBLE_RE.sub("", text).strip()
    # Only strip label prefix if it looks like an artifact (short prefix before colon)
    label_match = _LABEL_RE.match(text)
    if label_match and label_match.end() < 30:
        text = text[label_match.end():].strip()
    text = _QUOTE_RE.sub("", text).strip()
    return text


def rewrite_sentence(
    sentence: str,
    previous_rewrite: str | None = None,
    similarity_score: float | None = None,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Call Mistral 7B to rewrite a biased sentence into neutral language.
    On retries, pass previous_rewrite and similarity_score for corrective prompting.
    Returns the cleaned rewritten sentence.
    """
    logger.info("INPUT : %s", sentence)

    if previous_rewrite and similarity_score is not None:
        prompt = _RETRY_TEMPLATE.format(
            original=sentence,
            previous=previous_rewrite,
            similarity=similarity_score,
        )
    else:
        prompt = _USER_TEMPLATE.format(sentence=sentence)

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": temperature,
            "num_predict": MAX_NEW_TOKENS,
        },
    )
    raw = resp["message"]["content"].strip()
    cleaned = _clean(raw)

    logger.info("OUTPUT: %s", cleaned)
    return cleaned
