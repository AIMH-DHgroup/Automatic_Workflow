"""Shared implementation for the M1 narrative creation variants."""

from __future__ import annotations

import difflib
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests
from bs4 import BeautifulSoup
from json_repair import repair_json
from langchain_community.llms import Ollama
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from automatic_workflow.config import WorkflowSettings
from automatic_workflow.csv_utils import build_csv_text, read_csv_rows
from automatic_workflow.narrative import serialize_story_text
from automatic_workflow.text import (
    clean_text,
    normalize_key,
    normalize_key_aggressive,
    normalize_title,
)

LOGGER = logging.getLogger(__name__)

NarrativeMode = Literal[
    "paragraphs", "json_events", "paragraphs_with_titles", "chunked", "pointer_titles"
]

CHUNK_WORDS = 500
CHUNK_OVERLAP_WORDS = 50

REGION_PAGES = [
    "https://www.moving-h2020.eu/reference_regions/austrian-alps-austria/",
    "https://www.moving-h2020.eu/reference_regions/stara-planina-bulgaria/",
    "https://www.moving-h2020.eu/reference_regions/sumava-cesky-les-czechia/",
    "https://www.moving-h2020.eu/reference_regions/corsica-france/",
    "https://www.moving-h2020.eu/reference_regions/drome-valley-france/",
    "https://www.moving-h2020.eu/reference_regions/crete-greece/",
    "https://www.moving-h2020.eu/reference_regions/transdanubian-mountains-hungary/",
    "https://www.moving-h2020.eu/reference_regions/central-apennines-italy/",
    "https://www.moving-h2020.eu/reference_regions/eastern-alps-italy/",
    "https://www.moving-h2020.eu/reference_regions/northern-apennines-italy/",
    "https://www.moving-h2020.eu/reference_regions/maleshevski-mountains-north-macedonia/",
    "https://www.moving-h2020.eu/reference_regions/cordilheira-central-portugal/",
    "https://www.moving-h2020.eu/reference_regions/macico-noroeste-portugal/",
    "https://www.moving-h2020.eu/reference_regions/southern-romanian-carpathian-mountains-romania/",
    "https://www.moving-h2020.eu/reference_regions/dinaric-mountains-serbia/",
    "https://www.moving-h2020.eu/reference_regions/slovak-carpathian-mountains-slovakia/",
    "https://www.moving-h2020.eu/reference_regions/betic-systems-spain/",
    "https://www.moving-h2020.eu/reference_regions/sierra-morena-spain/",
    "https://www.moving-h2020.eu/reference_regions/spanish-pyrenees-spain/",
    "https://www.moving-h2020.eu/reference_regions/swiss-alps-switzerland/",
    "https://www.moving-h2020.eu/reference_regions/swiss-jura-switzerland/",
    "https://www.moving-h2020.eu/reference_regions/beydaglari-turkey/",
    "https://www.moving-h2020.eu/reference_regions/highlands-and-islands-uk-scotland/",
]


@dataclass(frozen=True)
class MappingRow:
    """One row from `mappingtable.csv`."""

    db_label_raw: str
    db_key: str
    db_key_aggressive: str
    event_label: str
    prefix: str
    suffix: str
    is_title: bool


def ensure_io_dirs(input_dir: Path, output_dir: Path) -> None:
    """Create standard module input/output directories."""

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_default_input_path(input_dir: Path) -> Path:
    """Resolve the default text/Markdown or CSV input for an M1 module."""

    candidates = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".csv"}
    )
    if not candidates:
        raise FileNotFoundError(f"No .txt or .csv input files found in {input_dir}")
    candidates.sort(key=lambda path: (0 if path.suffix.lower() == ".txt" else 1, path.name.lower()))
    return candidates[0]


def parse_bool(value: object) -> bool:
    """Parse common CSV boolean values."""

    return clean_text(value).lower() in {"true", "1", "yes", "y"}


def build_sentence(prefix: str, content: str, suffix: str) -> str:
    """Build one narrative sentence from mapped CSV fields."""

    body = clean_text(content)
    if not body:
        return ""
    if body.lower() in {"yes", "y"}:
        body = "are"
    elif body.lower() in {"no", "n"}:
        body = "are not"

    sentence = f"{(prefix or '').strip()} {body} {(suffix or '').strip()}".strip()
    return sentence if sentence.endswith(".") else f"{sentence}."


def is_quantitative_event(label: str) -> bool:
    """Return whether an event is numeric/statistical and should avoid decorative images."""

    keywords = ["income", "gross value", "employment", "gva", "population", "density"]
    return any(keyword in (label or "").lower() for keyword in keywords)


def load_mapping(mapping_path: Path) -> tuple[list[MappingRow], dict[str, MappingRow], list[str]]:
    """Load M1 dataset-to-narrative mapping rows."""

    rows = read_csv_rows(mapping_path)
    if not rows:
        raise RuntimeError(f"Mapping file is empty: {mapping_path}")

    mapping_rows: list[MappingRow] = []
    mapping_by_dbkey: dict[str, MappingRow] = {}
    event_order: list[str] = []
    seen_events: set[str] = set()

    for raw in rows[1:]:
        if len(raw) < 5:
            continue
        db_label_raw = raw[0]
        event_label = clean_text(raw[1])
        db_key = normalize_key(db_label_raw)
        if not db_key or not event_label:
            continue
        mapping_row = MappingRow(
            db_label_raw=db_label_raw,
            db_key=db_key,
            db_key_aggressive=normalize_key_aggressive(db_label_raw),
            event_label=event_label,
            prefix=raw[2] if len(raw) > 2 else "",
            suffix=raw[3] if len(raw) > 3 else "",
            is_title=parse_bool(raw[4]),
        )
        mapping_rows.append(mapping_row)
        mapping_by_dbkey[db_key] = mapping_row
        if event_label not in seen_events:
            seen_events.add(event_label)
            event_order.append(event_label)

    return mapping_rows, mapping_by_dbkey, event_order


def build_header_resolution(
    headers_raw: list[str], mapping_rows: list[MappingRow], mapping_by_dbkey: dict[str, MappingRow]
) -> dict[str, MappingRow]:
    """Map dataset headers to mapping rows using exact and fuzzy matching."""

    resolved: dict[str, MappingRow] = {}
    for header in headers_raw:
        header_key = normalize_key(header)
        if header_key in mapping_by_dbkey:
            resolved[header_key] = mapping_by_dbkey[header_key]

    mapping_aggressive_keys = [row.db_key_aggressive for row in mapping_rows]
    for header in headers_raw:
        header_key = normalize_key(header)
        if not header_key or header_key in resolved:
            continue
        header_aggressive = normalize_key_aggressive(header)
        if not header_aggressive:
            continue
        best = difflib.get_close_matches(
            header_aggressive, mapping_aggressive_keys, n=1, cutoff=0.90
        )
        if best:
            resolved[header_key] = mapping_rows[mapping_aggressive_keys.index(best[0])]
    return resolved


def best_region_page(country_name: str, region_pages: list[str] | None = None) -> str:
    """Choose the closest MOVING reference-region page for a country/region name."""

    pages = region_pages or REGION_PAGES
    country_norm = normalize_key(country_name)
    best_url = pages[0]
    best_score = -1.0
    for url in pages:
        slug_norm = normalize_key(url.strip("/").split("/")[-1])
        score = difflib.SequenceMatcher(None, country_norm, slug_norm).ratio()
        if score > best_score:
            best_score = score
            best_url = url
    LOGGER.info("Matched %s to %s with score %.2f", country_name, best_url, best_score)
    return best_url


def extract_images_from_region(url: str, timeout: int = 25) -> list[str]:
    """Scrape non-logo image URLs from a MOVING region page."""

    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("Image extraction failed for %s: %s", url, exc)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    images: list[str] = []
    seen: set[str] = set()
    for image in soup.find_all("img"):
        src = image.get("src")
        if not src or not src.startswith("http"):
            continue
        if any(part in src.lower() for part in ["logo", "icon", "flag"]):
            continue
        if src not in seen:
            seen.add(src)
            images.append(src)
    return images


def get_images_for_country(country: str, cache: dict[str, list[str]]) -> list[str]:
    """Return cached scraped images for a country/region."""

    key = normalize_key(country)
    if key not in cache:
        cache[key] = extract_images_from_region(best_region_page(country))
    return cache[key]


def serialize_story_csv(
    events: dict[str, dict[str, list[str] | str]],
    event_order: list[str],
    country: str,
    image_cache: dict[str, list[str]],
) -> str:
    """Serialize mapped dataset events to narrative CSV text."""

    rows = [["title", "description", "image"]]
    images = list(get_images_for_country(country, image_cache))
    random.shuffle(images)
    image_index = 0

    for event_label in event_order:
        data = events[event_label]
        title = clean_text(data.get("title", "")) or event_label
        desc_values = data.get("desc", [])
        descriptions = desc_values if isinstance(desc_values, list) else []
        description = " ".join(
            clean_text(item) for item in descriptions if clean_text(item)
        ).strip()
        image_url = ""
        if description and not is_quantitative_event(event_label) and image_index < len(images):
            image_url = images[image_index]
            image_index += 1
        rows.append([title, description, image_url])

    return build_csv_text(rows)


def process_csv(dataset_csv_path: Path, mapping_csv_path: Path, output_dir: Path) -> Path:
    """Convert a structured MOVING dataset CSV to one or more narrative CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows, mapping_by_dbkey, event_order = load_mapping(mapping_csv_path)
    all_rows = read_csv_rows(dataset_csv_path)
    if not all_rows:
        raise ValueError(f"Dataset is empty: {dataset_csv_path}")

    headers_raw = all_rows[0]
    header_keys = [normalize_key(header) for header in headers_raw]
    resolved_header_map = build_header_resolution(headers_raw, mapping_rows, mapping_by_dbkey)
    image_cache: dict[str, list[str]] = {}

    for row_index, row in enumerate(all_rows[1:], start=1):
        country = clean_text(row[0]) if row else ""
        events: dict[str, dict[str, list[str] | str]] = {
            event: {"title": "", "desc": []} for event in event_order
        }
        for col in range(min(len(header_keys), len(row))):
            mapping_row = resolved_header_map.get(header_keys[col])
            if not mapping_row:
                continue
            field = clean_text(row[col])
            if not field:
                continue
            sentence = build_sentence(mapping_row.prefix, field, mapping_row.suffix)
            if not sentence:
                continue
            if mapping_row.is_title and not events[mapping_row.event_label]["title"]:
                events[mapping_row.event_label]["title"] = sentence
            elif not mapping_row.is_title:
                desc = events[mapping_row.event_label]["desc"]
                if isinstance(desc, list):
                    desc.append(sentence)

        out_path = output_dir / f"{row_index}.csv"
        out_path.write_text(
            serialize_story_csv(events, event_order, country, image_cache), encoding="utf-8"
        )
        LOGGER.info("Story %s saved to %s", row_index, out_path)

    return output_dir


def invoke_ollama(
    system_prompt: str,
    prompt: str,
    settings: WorkflowSettings,
    temperature: float = 0.01,
    json_mode: bool = False,
) -> str:
    """Invoke the configured local Ollama model."""

    llm_kwargs = {
        "model": settings.ollama_model,
        "system": system_prompt,
        "num_ctx": 4096,
        "temperature": temperature,
    }
    if json_mode:
        llm_kwargs["format"] = "json"
    llm = Ollama(
        **llm_kwargs,
    )
    response = llm.invoke(prompt, config={"callbacks": [StreamingStdOutCallbackHandler()]})
    print()
    return str(response)


def split_text_into_paragraphs(narrative: str, settings: WorkflowSettings) -> list[str]:
    """Use Ollama to split text into paragraphs without rewriting it."""

    llm_output = invoke_ollama(
        "Divide the provided text into paragraphs without deleting, adding or changing words.",
        narrative,
        settings,
    )
    return [part.strip() for part in re.split(r"\n\s*\n", llm_output) if part.strip()]


def split_into_word_chunks(
    text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS
) -> list[str]:
    """Split text into word chunks with overlap for boundary context."""

    if chunk_words <= 0 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_words")
    words = text.split()
    step = chunk_words - overlap_words
    return [" ".join(words[start : start + chunk_words]) for start in range(0, len(words), step)]


def merge_overlapping_texts(
    texts: list[str], overlap_words: int = CHUNK_OVERLAP_WORDS
) -> str:
    """Join chunk outputs once, removing their repeated word overlap."""

    if not texts:
        return ""
    merged = texts[0].strip()
    for text in texts[1:]:
        previous_words = merged.split()
        current_words = text.split()
        overlap = _find_word_overlap(previous_words, current_words, overlap_words)
        merged = " ".join(previous_words + current_words[overlap:])
    return merged


def _find_word_overlap(previous: list[str], current: list[str], limit: int) -> int:
    """Return the largest suffix/prefix overlap between two word lists."""

    overlap = min(limit, len(previous), len(current))
    while overlap and previous[-overlap:] != current[:overlap]:
        overlap -= 1
    return overlap


def merge_overlapping_paragraphs(
    paragraph_groups: list[list[str]], overlap_words: int = CHUNK_OVERLAP_WORDS
) -> list[str]:
    """Recombine chunk paragraphs without flattening their event boundaries."""

    merged: list[str] = []
    for group in paragraph_groups:
        paragraphs = [paragraph.strip() for paragraph in group if paragraph.strip()]
        if not paragraphs:
            continue
        if not merged:
            merged.extend(paragraphs)
            continue

        previous_words = merged[-1].split()
        current_words = paragraphs[0].split()
        overlap = _find_word_overlap(previous_words, current_words, overlap_words)
        if overlap:
            merged[-1] = " ".join(previous_words + current_words[overlap:])
            merged.extend(paragraphs[1:])
        else:
            merged.extend(paragraphs)
    return merged


def split_text_in_chunks(narrative: str, settings: WorkflowSettings) -> list[str]:
    """Split each 500-word chunk with Ollama and remove repeated overlaps."""

    chunks = split_into_word_chunks(narrative)
    split_chunks = [split_text_into_paragraphs(chunk, settings) for chunk in chunks]
    return merge_overlapping_paragraphs(split_chunks)


def split_text_into_sentences(narrative: str) -> list[tuple[int, int, str]]:
    """Return deterministic character spans for sentence-like units."""

    pattern = r"\S.*?(?:[.!?](?:['\"”’)]*)?(?=\s|$)|$)"
    return [
        (match.start(), match.end(), match.group().strip())
        for match in re.finditer(pattern, narrative, re.DOTALL)
    ]


def group_sentences_for_fallback(text: str, sentences_per_event: int = 5) -> list[str]:
    """Create bounded deterministic events when the pointer response is invalid."""

    sentences = [sentence for _, _, sentence in split_text_into_sentences(text)]
    return [
        " ".join(sentences[index : index + sentences_per_event])
        for index in range(0, len(sentences), sentences_per_event)
    ]


def split_text_by_pointers(narrative: str, settings: WorkflowSettings) -> list[str]:
    """Use LLM sentence ranges and slice the original text locally."""

    sentences = split_text_into_sentences(narrative)
    if not sentences:
        raise ValueError("Source text contains no sentence-like units")
    numbered = "\n".join(f"[{index}] {sentence[2]}" for index, sentence in enumerate(sentences, 1))
    response = invoke_ollama(
        "Return only the JSON array requested by the user. Never write an explanation or prose.",
        """Group the numbered sentences below into logical narrative events.
Return only this JSON array shape, in the original order:
[
  {"start": 1, "end": 3},
  {"start": 4, "end": 6}
]

Rules:
- The values are sentence numbers, not character positions.
- Every sentence number must be included exactly once.
- Ranges must be consecutive, ordered, and non-overlapping.
- The first range must start at sentence 1.
- The final range must end at the last sentence.
- Group related sentences when the subject, place, period, activity, or theme is shared.
- Split one source paragraph into multiple events when it contains distinct concepts.
- Return no explanation, Markdown fence, or text outside the JSON array.

Sentences:
""" + numbered,
        settings,
        temperature=0.0,
        json_mode=True,
    )
    pointers = extract_json_array(response)
    if not pointers:
        raise ValueError("LLM returned no sentence pointers")

    paragraphs: list[str] = []
    cursor = 1
    for pointer in pointers:
        start = pointer.get("start")
        end = pointer.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Sentence pointers must contain integer start and end values")
        if start != cursor or end < start or end > len(sentences):
            raise ValueError("Sentence pointers must cover the source in consecutive ranges")
        paragraphs.append(narrative[sentences[start - 1][0] : sentences[end - 1][1]].strip())
        cursor = end + 1

    if cursor != len(sentences) + 1:
        raise ValueError("Sentence pointers do not cover the complete source text")
    return paragraphs


def extract_json_array(text: str) -> list[dict[str, str]] | None:
    """Extract and repair the first JSON array from an LLM response."""

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    candidate = re.sub(r",\s*(\}|\])", r"\1", match.group(0))
    for attempt in (candidate, repair_json(candidate)):
        try:
            parsed = json.loads(attempt)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return None


def split_text_into_json_events(narrative: str, settings: WorkflowSettings) -> list[dict[str, str]]:
    """Use Ollama to split text into JSON event objects."""

    system_prompt = """Divide the provided text into events without deleting, adding or changing words. The answer should be a json like this:

###Json
[
  {
    "title": "...",
    "event": "..."
  }
]

Answer only with the json
"""
    response = invoke_ollama(system_prompt, narrative, settings)
    json_events = extract_json_array(response)
    if json_events is None:
        LOGGER.warning("Invalid JSON from Ollama, falling back to full text")
        return [{"title": "event-1", "event": narrative}]
    return json_events


def serialize_story_from_json(json_data: list[dict[str, str]]) -> str:
    """Serialize JSON event objects to narrative CSV text."""

    rows = [["title", "description"]]
    for item in json_data:
        rows.append(
            [clean_text(item.get("title", "")) or "event", clean_text(item.get("event", ""))]
        )
    return build_csv_text(rows)


def generate_titles_for_paragraphs(paragraphs: list[str], settings: WorkflowSettings) -> list[str]:
    """Use Ollama to generate exactly one title from each paragraph."""

    if not paragraphs:
        return []
    titles: list[str] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        response = invoke_ollama(
            "Write one concise title for the provided event. Answer only with the title.",
            paragraph,
            settings,
        )
        title = normalize_title(response.splitlines()[0] if response.strip() else "")
        if not title:
            LOGGER.warning("No title returned for event %s; fallback used", index)
            title = f"event-{index}"
        titles.append(title)
    return titles


def report_text_preservation(source: str, generated: str) -> bool:
    """Report whether M1 preserved the source text apart from whitespace layout."""

    source_normalized = " ".join(source.split())
    generated_normalized = " ".join(generated.split())
    preserved = source_normalized == generated_normalized
    status = "OK" if preserved else "WARNING"
    print(
        f"[M1] Text preservation check: {status}"
        if preserved
        else (
            "[M1] Text preservation check: WARNING - the model changed or omitted "
            "part of the source text"
        )
    )
    return preserved


def process_txt(
    txt_path: Path, output_dir: Path, mode: NarrativeMode, settings: WorkflowSettings
) -> Path:
    """Convert a text file to a narrative CSV using one M1 variant mode."""

    output_dir.mkdir(parents=True, exist_ok=True)
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    if mode == "paragraphs":
        paragraphs = split_text_into_paragraphs(text, settings)
        report_text_preservation(text, "\n\n".join(paragraphs))
        story_csv = serialize_story_text(paragraphs, [])
    elif mode == "json_events":
        json_events = split_text_into_json_events(text, settings)
        report_text_preservation(text, "\n\n".join(item.get("event", "") for item in json_events))
        story_csv = serialize_story_from_json(json_events)
    else:
        if mode == "pointer_titles":
            try:
                paragraphs = split_text_by_pointers(text, settings)
            except ValueError as exc:
                print(f"[M1] Pointer splitting WARNING: {exc}; using source sentences")
                paragraphs = group_sentences_for_fallback(text)
        elif mode == "chunked":
            paragraphs = split_text_in_chunks(text, settings)
        else:
            paragraphs = split_text_into_paragraphs(text, settings)
        report_text_preservation(text, "\n\n".join(paragraphs))
        story_csv = serialize_story_text(
            paragraphs, generate_titles_for_paragraphs(paragraphs, settings)
        )

    out_path = output_dir / f"{txt_path.stem}.csv"
    out_path.write_text(story_csv, encoding="utf-8", newline="")
    LOGGER.info("CSV created from TXT at %s", out_path)
    return out_path


def run_m1_module(base_dir: Path, mode: NarrativeMode, input_path: str | None = None) -> Path:
    """Run an M1 legacy module directory with package-backed implementation."""

    project_root = base_dir.parent
    settings = WorkflowSettings.from_env(project_root)
    settings.validate()
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    ensure_io_dirs(input_dir, output_dir)
    mapping_path = base_dir / "mappingtable.csv"
    selected = Path(input_path).resolve() if input_path else resolve_default_input_path(input_dir)
    if not selected.is_file():
        raise FileNotFoundError(f"File not found: {selected}")
    if selected.suffix.lower() == ".csv":
        return process_csv(selected, mapping_path, output_dir)
    if selected.suffix.lower() in {".txt", ".md"}:
        return process_txt(selected, output_dir, mode, settings)
    raise ValueError("Input path must end with .csv, .txt or .md")
