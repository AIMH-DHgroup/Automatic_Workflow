"""Alternative M2 entity enrichment focused on map-ready entities.

The original `M2-Entities` module is intentionally left unchanged. This module
reuses its stable CSV/Wikidata output contract while changing the entity prompt
and adding Wikipedia page-image enrichment.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ORIGINAL_PATH = PROJECT_ROOT / "M2-Entities" / "main.py"
SPEC = importlib.util.spec_from_file_location("automatic_workflow_m2_original", ORIGINAL_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Unable to load original M2 module from {ORIGINAL_PATH}")
ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL)

INPUT_DIR = Path(__file__).resolve().parent / "input"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
WIKIPEDIA_IMAGE_TIMEOUT = 20

GEO_PRIORITY_SYSTEM_PROMPT = """Recognize all meaningful entities in the text and find the exact English Wikipedia title for each entity.

Prioritize entities that improve map visualization: countries, regions, cities, villages, mountains, rivers, lakes, coastlines, protected areas, landscapes, landmarks, buildings, routes, organizations tied to a place, and named cultural or natural sites.

Do not return only places. Include important people, organizations, works, concepts, and events when they are meaningful to the narrative. Prefer specific named entities over generic words. For geographic entities, prefer the exact Wikipedia page for the real-world place or feature.

Answer only with JSON in this form:
{
  "entities": [
    {
      "entity_in_the_text": "...",
      "wikipedia_title": "..."
    }
  ]
}
"""

_QID_TO_WIKIPEDIA_TITLE: dict[str, str] = {}


def useLLM(text: str) -> str:
    """Run the original Ollama integration with the alternative prompt/model."""

    llm = ORIGINAL.Ollama(
        model=useLLM.model,
        system=GEO_PRIORITY_SYSTEM_PROMPT,
        num_ctx=4096,
        temperature=0.01,
    )
    response = llm.invoke(
        text,
        config={"callbacks": [ORIGINAL.StreamingStdOutCallbackHandler()]},
    )
    print()
    return str(response)


useLLM.model = "gemma3:12b"  # type: ignore[attr-defined]


def fetch_wikipedia_images(titles: list[str]) -> dict[str, list[str]]:
    """Fetch page images from Wikipedia for resolved entity titles."""

    unique_titles = list(dict.fromkeys(title.strip() for title in titles if title.strip()))
    if not unique_titles:
        return {}

    result: dict[str, list[str]] = {title: [] for title in unique_titles}
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": ORIGINAL.USER_AGENT,
            "Accept": "application/json",
        }
    )

    for start in range(0, len(unique_titles), 50):
        chunk = unique_titles[start : start + 50]
        try:
            response = session.get(
                ORIGINAL.WIKIPEDIA_API,
                params={
                    "action": "query",
                    "format": "json",
                    "redirects": 1,
                    "prop": "pageimages|imageinfo",
                    "piprop": "original",
                    "iiprop": "url",
                    "titles": "|".join(chunk),
                },
                timeout=WIKIPEDIA_IMAGE_TIMEOUT,
            )
            response.raise_for_status()
            pages = response.json().get("query", {}).get("pages", {})
        except (requests.RequestException, ValueError) as exc:
            print(f"Wikipedia image fetch failed: {exc}")
            continue

        for page in pages.values():
            title = page.get("title", "")
            image_info = page.get("imageinfo", [])
            if image_info and image_info[0].get("url"):
                result.setdefault(title, []).append(image_info[0]["url"])
            thumbnail = page.get("thumbnail", {}).get("source")
            if thumbnail and thumbnail not in result.setdefault(title, []):
                result[title].append(thumbnail)

    return result


def fetch_wikidata_qids_from_titles(titles: list[str]) -> dict[str, str | None]:
    """Resolve titles through the original Wikipedia-to-Wikidata function."""

    result = ORIGINAL.fetch_wikidata_qids_from_titles(titles)
    _QID_TO_WIKIPEDIA_TITLE.clear()
    for title, qid in result.items():
        if qid:
            _QID_TO_WIKIPEDIA_TITLE[qid] = title
    return result


def fetch_wikidata_details_by_qids(qids: list[str]) -> dict[str, dict[str, Any]]:
    """Combine Wikidata images with images returned by Wikipedia pages."""

    details = ORIGINAL.fetch_wikidata_details_by_qids(qids)
    wikipedia_images = fetch_wikipedia_images(list(_QID_TO_WIKIPEDIA_TITLE.values()))
    for qid in qids:
        title = _QID_TO_WIKIPEDIA_TITLE.get(qid)
        if not title:
            continue
        entity = details.setdefault(qid, {"label_en": None, "images": [], "coord": None})
        existing = entity.setdefault("images", [])
        for image in wikipedia_images.get(title, []):
            if image not in existing:
                existing.append(image)
    return details


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse alternative M2 command-line arguments."""

    parser = argparse.ArgumentParser(description="Map-focused alternative M2 entity enrichment")
    parser.add_argument("input_csv", nargs="?", help="Narrative CSV produced by M1")
    parser.add_argument(
        "--model",
        default=os.getenv("AUTOMATIC_WORKFLOW_ALTERNATIVE_ENTITY_MODEL", "gemma3:12b"),
        help="Ollama model for alternative M2",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run alternative M2 while preserving the original output format."""

    args = parse_args(argv)
    useLLM.model = args.model  # type: ignore[attr-defined]
    original_functions = {
        "useLLM": ORIGINAL.useLLM,
        "fetch_wikidata_qids_from_titles": ORIGINAL.fetch_wikidata_qids_from_titles,
        "fetch_wikidata_details_by_qids": ORIGINAL.fetch_wikidata_details_by_qids,
        "INPUT_DIR": ORIGINAL.INPUT_DIR,
        "OUTPUT_DIR": ORIGINAL.OUTPUT_DIR,
    }
    ORIGINAL.useLLM = useLLM
    ORIGINAL.fetch_wikidata_qids_from_titles = fetch_wikidata_qids_from_titles
    ORIGINAL.fetch_wikidata_details_by_qids = fetch_wikidata_details_by_qids
    ORIGINAL.INPUT_DIR = str(INPUT_DIR)
    ORIGINAL.OUTPUT_DIR = str(OUTPUT_DIR)
    try:
        input_csv = args.input_csv or ORIGINAL.resolve_default_input_csv_path()
        output = ORIGINAL.enrich_csv_with_wikidata_qids(input_csv)
        print(f"Alternative M2 output CSV created -> {output}")
        return 0
    finally:
        ORIGINAL.useLLM = original_functions["useLLM"]
        ORIGINAL.fetch_wikidata_qids_from_titles = original_functions[
            "fetch_wikidata_qids_from_titles"
        ]
        ORIGINAL.fetch_wikidata_details_by_qids = original_functions[
            "fetch_wikidata_details_by_qids"
        ]
        ORIGINAL.INPUT_DIR = original_functions["INPUT_DIR"]
        ORIGINAL.OUTPUT_DIR = original_functions["OUTPUT_DIR"]


if __name__ == "__main__":
    raise SystemExit(main())
