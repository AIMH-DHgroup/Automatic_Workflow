"""Helpers for extracting and repairing JSON returned by LLMs."""

from __future__ import annotations

import json
import re
from typing import Any

from json_repair import repair_json


def extract_first_json_object(text: str) -> str | None:
    """Extract and repair the first JSON object from a string.

    Args:
        text: Raw model response.

    Returns:
        A valid JSON object string, or `None` if no valid object can be found.
    """

    if not text or not text.strip():
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    candidate = re.sub(r",\s*(\}|\])", r"\1", match.group(0))
    for attempt in (candidate, repair_json(candidate)):
        try:
            json.loads(attempt)
            return attempt
        except (TypeError, json.JSONDecodeError):
            continue
    return None


def parse_entities_response(text: str) -> list[dict[str, str]]:
    """Parse M2 entity extraction output.

    Args:
        text: Raw model response expected to contain `{ "entities": [...] }`.

    Returns:
        Valid entity dictionaries with `entity_in_the_text` and `wikipedia_title`.
    """

    fixed = extract_first_json_object(text)
    if fixed is None:
        return []
    try:
        data: Any = json.loads(fixed)
    except json.JSONDecodeError:
        return []
    entities = data.get("entities", []) if isinstance(data, dict) else []
    if not isinstance(entities, list):
        return []
    parsed: list[dict[str, str]] = []
    for item in entities:
        if not isinstance(item, dict):
            continue
        title = str(item.get("wikipedia_title", "")).strip()
        if not title:
            continue
        parsed.append(
            {
                "entity_in_the_text": str(item.get("entity_in_the_text", "")).strip(),
                "wikipedia_title": title,
            }
        )
    return parsed
