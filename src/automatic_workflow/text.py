"""Text normalization utilities shared across workflow stages."""

from __future__ import annotations

import re


def clean_text(value: object) -> str:
    """Normalize text loaded from CSV or LLM responses.

    Args:
        value: Raw value to normalize.

    Returns:
        A stripped single-line string. `N/A` is normalized to an empty string.
    """

    if value is None:
        return ""
    text = str(value).replace('"', "'")
    text = text.replace("\ufeff", "").replace("\u00a0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return "" if text.upper() == "N/A" else text


def normalize_key(value: object) -> str:
    """Normalize a CSV header or mapping key for matching."""

    text = clean_text(value).lower()
    text = re.sub(r"\bnuts\s*([23])\b", r"nuts\1", text)
    text = re.sub(r"[(),]", " ", text)
    text = text.replace("/", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_key_aggressive(value: object) -> str:
    """Normalize a key and remove low-signal words for fuzzy matching."""

    stop_words = {"per", "of", "total", "year", "eur", "million"}
    return " ".join(token for token in normalize_key(value).split() if token not in stop_words)


def normalize_title(title: object) -> str:
    """Normalize generated event titles."""

    return clean_text(title).strip("; '")
