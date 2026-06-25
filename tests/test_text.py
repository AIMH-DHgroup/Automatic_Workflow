from __future__ import annotations

from automatic_workflow.text import (
    clean_text,
    normalize_key,
    normalize_key_aggressive,
    normalize_title,
)


def test_clean_text_normalizes_whitespace_and_na() -> None:
    assert clean_text("  A\n\tB  ") == "A B"
    assert clean_text("N/A") == ""
    assert clean_text(None) == ""


def test_normalize_key_handles_punctuation_and_nuts() -> None:
    assert normalize_key("NUTS 2 (A/B-C)") == "nuts2 a b c"


def test_normalize_key_aggressive_removes_low_signal_words() -> None:
    assert normalize_key_aggressive("Total income per year EUR million") == "income"


def test_normalize_title_removes_separators_and_quotes() -> None:
    assert normalize_title(" '; Mountain assets; '") == "Mountain assets"
