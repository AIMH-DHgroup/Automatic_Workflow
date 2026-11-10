from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from automatic_workflow.config import WorkflowSettings  # noqa: E402


# This script is a Python port of `scriptStories5OSM.php`.
#
# Goal:
# - read one narrative CSV chosen explicitly in a variable
# - enrich the entities with Wikidata and QLever OSM
# - create storymap assets under `HTML_Storymap/`
# - create Triplify input JSON under `json/`
# - create OWL files under `owl/` through `Triplify/triplify.jar`
#
# The code intentionally mirrors the current PHP behavior as closely as possible,
# but it is organized into explicit helper functions so each step is easy to read.


BASE_DIR = Path(__file__).resolve().parent
SETTINGS = WorkflowSettings.from_env(PROJECT_ROOT)
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

STORYMAP_DIR = OUTPUT_DIR / "HTML_Storymap"
JSON_DIR = OUTPUT_DIR / "json"
OWL_DIR = OUTPUT_DIR / "owl"
TRIPLIFY_JAR = PROJECT_ROOT / "M3-graph" / "Triplify" / "Triplify.jar"
LOG_PATH = BASE_DIR / "scriptStories5OSM_py.log"
USER_NAME = "moving"
USER_ID = 52
USER_IDENTIFIER = f"{USER_NAME}.{USER_ID}"
M3_OLLAMA_MODEL = os.getenv("AUTOMATIC_WORKFLOW_M3_OLLAMA_MODEL", SETTINGS.ollama_model)
M3_OLLAMA_TIMEOUT = int(os.getenv("AUTOMATIC_WORKFLOW_M3_OLLAMA_TIMEOUT", "30"))
M3_USE_LLM = os.getenv("AUTOMATIC_WORKFLOW_M3_USE_LLM", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
STORYMAP_COVER_IMAGE = os.getenv(
    "AUTOMATIC_WORKFLOW_COVER_IMAGE",
    "https://tool.dlnarratives.eu/images/title-background.jpg",
)
SLIDE_TYPE_GEOGRAPHIC = "geographic"
SLIDE_TYPE_NORMAL = "normal"

# The Python port uses the same candidate Java runtimes used by the PHP script.
JAVA_CANDIDATES = [
    Path(
        r"C:\Users\aimhd\.p2\pool\plugins\org.eclipse.justj.openjdk.hotspot.jre.full.win32.x86_64_21.0.8.v20250724-1412\jre\bin\java.exe"
    ),
    Path(
        r"C:\Users\aimhd\.vscode\extensions\redhat.java-1.54.0-win32-x64\jre\21.0.10-win32-x86_64\bin\java.exe"
    ),
    Path(r"C:\Users\aimhd\Downloads\Protege-5.6.7-win\Protege-5.6.7\jre\bin\java.exe"),
    Path(
        r"C:\Users\aimhd\AppData\Roaming\ModrinthApp\meta\java_versions\zulu21.38.21-ca-jre21.0.5-win_x64\bin\java.exe"
    ),
]

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}


SPECIAL_QID_REPLACEMENTS = {
    "Q12054746": "Q2869587",
    "Q14907183": "Q11714837",
    "Q16968816": "Q16189205",
    "Q3773637": "Q3045473",
    "Q6576947": "Q14888394",
    "Q16886469": "Q1422583",
}


def ensure_io_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def log_message(message: str) -> None:
    """Append one timestamped line to the Python log file."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")


def parse_csv_rows(path: Path) -> list[list[str]]:
    """Read the semicolon-separated CSV produced by M2."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle, delimiter=";", quotechar='"', doublequote=True))


def parse_entity_coords(coord_text: str) -> tuple[str, str]:
    """Return the first valid `(lon, lat)` pair found in a M2 `wikidata_coords` cell."""
    for item in (coord_text or "").split(","):
        value = item.strip()
        if not value or value.lower() == "null":
            continue
        parts = value.split()
        if len(parts) < 2:
            continue
        try:
            lat = float(parts[0])
            lon = float(parts[1])
        except ValueError:
            continue
        return f"{lon}", f"{lat}"
    return "", ""


def parse_wkt_point(point_text: str) -> tuple[str, str]:
    """Return `(lon, lat)` from a Wikidata WKT Point value."""

    match = re.search(r"Point\(\s*([-0-9.]+)\s+([-0-9.]+)\s*\)", point_text or "")
    return (match.group(1), match.group(2)) if match else ("", "")


def build_wikidata_link(qid: str) -> str:
    qid = (qid or "").strip()
    return f"https://www.wikidata.org/wiki/{qid}" if qid else ""


def first_image_from_cell(image_text: str) -> str:
    """Return the first usable image URL from a M2 `wikidata_images` cell."""
    for entity_images in (image_text or "").split(","):
        candidate = entity_images.strip()
        if not candidate or candidate.lower() == "null":
            continue
        first = candidate.split("|")[0].strip()
        if first and first.lower() != "null":
            return first
    return ""


def image_candidates_from_cell(image_text: str) -> list[str]:
    """Return one safe image URL candidate per entity from a M2 image cell."""

    candidates: list[str] = []
    for entity_images in (image_text or "").split(","):
        candidate = entity_images.strip()
        if not candidate or candidate.lower() == "null":
            continue
        first = safe_media_url(candidate.split("|")[0].strip())
        if first and first not in candidates:
            candidates.append(first)
    return candidates


def is_safe_external_url(url: str) -> bool:
    """Return whether a generated media/hyperlink URL uses an allowed scheme."""

    parsed = urlparse((url or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def safe_media_url(url: str) -> str:
    """Drop unsafe media URLs before writing StoryMap JSON."""

    return url if is_safe_external_url(url) else ""


def choose_non_repeated_asset(candidates: list[str], previously_used: set[str]) -> str:
    """Choose the first candidate not used by previous slides.

    If all available candidates were already used, keep the first candidate to
    preserve existing output instead of dropping media or polygons.
    """

    clean_candidates = [candidate for candidate in candidates if candidate]
    for candidate in clean_candidates:
        if candidate not in previously_used:
            return candidate
    return clean_candidates[0] if clean_candidates else ""


def choose_non_repeated_assets(candidates: list[str], previously_used: set[str]) -> list[str]:
    """Choose all candidates not used by previous slides, preserving order.

    If every candidate was already used, return the first candidate so the slide
    still has a geometry instead of becoming empty.
    """

    clean_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in clean_candidates:
            clean_candidates.append(candidate)
    fresh = [candidate for candidate in clean_candidates if candidate not in previously_used]
    return fresh if fresh else clean_candidates[:1]


def fallback_slide_type(has_coordinates: bool) -> str:
    """Choose a slide type without LLM help.

    Geographic slides require coordinates; normal slides are used otherwise.
    """

    return SLIDE_TYPE_GEOGRAPHIC if has_coordinates else SLIDE_TYPE_NORMAL


def normalize_slide_type(value: str, has_coordinates: bool) -> str:
    """Normalize an LLM slide-type response to `geographic` or `normal`."""

    cleaned = (value or "").strip().lower()
    if cleaned in {SLIDE_TYPE_NORMAL, "standard", "plain", "non-geographic", "non geographic"}:
        return SLIDE_TYPE_NORMAL
    if cleaned in {SLIDE_TYPE_GEOGRAPHIC, "geo", "map", "mapped"}:
        return fallback_slide_type(has_coordinates)
    return fallback_slide_type(has_coordinates)


def invoke_m3_ollama(prompt: str) -> str:
    """Classify one slide through Ollama with a bounded HTTP request."""

    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    request = Request(
        f"{host}/api/generate",
        data=json.dumps(
            {
                "model": M3_OLLAMA_MODEL,
                "prompt": prompt,
                "system": "You classify storymap slides. Answer only with geographic or normal.",
                "stream": False,
                "options": {"num_ctx": 2048, "temperature": 0.0, "num_predict": 8},
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json", **HTTP_HEADERS},
        method="POST",
    )
    with urlopen(request, timeout=M3_OLLAMA_TIMEOUT) as response:
        body = json.loads(response.read().decode("utf-8", errors="replace"))
    return str(body.get("response", "")).strip()


def choose_slide_type_with_llm(
    event_title: str,
    description: str,
    entity_labels: list[str],
    has_coordinates: bool,
) -> str:
    """Ask Ollama whether an event should be a geographic or normal slide.

    The model is advisory only. Invalid output, runtime errors, or a geographic
    choice without coordinates fall back to deterministic coordinate-based logic.
    """

    if not M3_USE_LLM:
        return fallback_slide_type(has_coordinates)

    prompt = f"""Choose the slide type for a narrative event.

Return only one word: geographic or normal.

Use geographic only when the event is mainly about a place, movement, territory, landscape, route, region, or local asset that benefits from a map.
Use normal for abstract, conceptual, statistical, organizational, or concluding events.
If no coordinates are available, return normal.

Coordinates available: {"yes" if has_coordinates else "no"}
Title: {event_title}
Description: {description}
Entities: {", ".join(entity_labels)}
"""
    try:
        response = invoke_m3_ollama(prompt)
    except Exception as exc:
        log_message(f"Slide type LLM failed for '{event_title}': {exc}")
        return fallback_slide_type(has_coordinates)

    slide_type = normalize_slide_type(response, has_coordinates)
    if slide_type != response.lower():
        log_message(f"Slide type LLM response '{response}' normalized to '{slide_type}'")
    return slide_type


def has_existing_outputs(csv_key: str) -> bool:
    """
    Skip already processed narratives when all final outputs are already on disk.
    """
    story = list(STORYMAP_DIR.glob(f"N*_{csv_key}"))
    json_files = list(JSON_DIR.glob(f"{USER_IDENTIFIER}_N*_{csv_key}.json"))
    owl = list(OWL_DIR.glob(f"{USER_IDENTIFIER}_N*_{csv_key}.owl"))
    if not story or not json_files or not owl:
        return False
    for json_file in json_files:
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("entities"):
            return True
    return False


def resolve_default_input_csv_path() -> Path:
    ensure_io_dirs()
    candidates = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv" and path.stem.lower().endswith("_qids")
    )
    if not candidates:
        raise FileNotFoundError(f"No input *_qids.csv files found in {INPUT_DIR}")
    return candidates[0]


def http_get_json(url: str, timeout: int = 60) -> dict[str, Any] | None:
    request = Request(url, headers=HTTP_HEADERS)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        log_message(f"HTTP GET failed for {url}: {exc}")
        return None


def http_post_form_json(
    url: str, payload: dict[str, str], timeout: int = 60
) -> dict[str, Any] | None:
    request = Request(
        url,
        data=urlencode(payload).encode("utf-8"),
        headers={**HTTP_HEADERS, "Accept": "application/sparql-results+json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        log_message(f"HTTP POST failed for {url}: {exc}")
        return None


_WIKIDATA_CACHE: dict[str, dict[str, Any]] = {}
_TYPE_CACHE: dict[str, list[str]] = {}
_POLYGON_CACHE: dict[str, str] = {}
_GEOMETRY_CACHE: dict[str, dict[str, str]] = {}


def fetch_wikidata_entity(qid: str) -> dict[str, Any]:
    """
    Fetch labels, descriptions, image, coordinates and coarse type information
    from Wikidata for one entity. Results are cached for the run: the same place
    appears in many events and must not be requested again.
    """
    qid = SPECIAL_QID_REPLACEMENTS.get(qid, qid)
    if not re.fullmatch(r"Q[1-9][0-9]*", qid):
        log_message(f"Invalid Wikidata QID: {qid!r}")
        return {}
    if qid in _WIKIDATA_CACHE:
        return _WIKIDATA_CACHE[qid]
    query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
SELECT DISTINCT ?uri ?type ?itName ?enName ?itDesc ?enDesc ?image ?coordinates
WHERE {{
  VALUES ?uri {{ wd:{qid} }}
  OPTIONAL {{ ?uri wdt:P31 ?type. }}
  OPTIONAL {{ ?uri wdt:P18 ?image. }}
  OPTIONAL {{ ?uri wdt:P625 ?coordinates. }}
  OPTIONAL {{ ?uri rdfs:label ?itName. FILTER(LANG(?itName) = "it") }}
  OPTIONAL {{ ?uri rdfs:label ?enName. FILTER(LANG(?enName) = "en") }}
  OPTIONAL {{ ?uri schema:description ?itDesc. FILTER(LANG(?itDesc) = "it") }}
  OPTIONAL {{ ?uri schema:description ?enDesc. FILTER(LANG(?enDesc) = "en") }}
}}
LIMIT 500
"""
    data = http_post_form_json(
        "https://query.wikidata.org/sparql",
        {"format": "json", "query": query},
        timeout=15,
    ) or {}
    if data:  # a failed request is not cached, so a later event can retry it
        _WIKIDATA_CACHE[qid] = data
    return data


KNOWN_TYPE_QIDS = " ".join(
    "wd:" + qid
    for qid in (
        "Q15222213 Q17334923 Q43229 Q8436 Q488383 Q7184903 Q386724 Q234460 Q5 "
        "Q186081 Q1190554 Q35120 Q15474042 Q4167836 Q41176 Q8205328 Q5127848 Q27096213 "
        "Q34379 Q110295396 Q187947 Q33506 Q16970 Q3947 Q16560 Q23413 Q44613 Q1640824 "
        "Q42195 Q839954 Q120964921 Q1294787 Q639669"
    ).split()
)


def fetch_entity_type_qids(qid: str) -> list[str]:
    """Fetch only known superclass QIDs needed by the local classifier."""

    if qid in _TYPE_CACHE:
        return _TYPE_CACHE[qid]
    query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT DISTINCT ?type WHERE {{
  VALUES ?item {{ wd:{qid} }}
  ?item wdt:P31/wdt:P279* ?type.
  VALUES ?type {{ {KNOWN_TYPE_QIDS} }}
}}
LIMIT 100
"""
    data = http_post_form_json(
        "https://query.wikidata.org/sparql",
        {"format": "json", "query": query},
        timeout=10,
    ) or {}
    types = list(
        dict.fromkeys(
            row.get("type", {}).get("value", "").rsplit("/", 1)[-1]
            for row in data.get("results", {}).get("bindings", [])
            if row.get("type", {}).get("value")
        )
    )
    _TYPE_CACHE[qid] = types
    return types


def wikidata_values(bindings: list[dict[str, Any]], field: str) -> list[str]:
    """Return unique, nonempty SPARQL values in response order."""

    return list(
        dict.fromkeys(
            row.get(field, {}).get("value", "")
            for row in bindings
            if row.get(field, {}).get("value", "")
        )
    )


def qids_from_bindings(bindings: list[dict[str, Any]], field: str) -> list[str]:
    """Extract unique QIDs from URI-valued SPARQL bindings."""

    return list(dict.fromkeys(value.rsplit("/", 1)[-1] for value in wikidata_values(bindings, field)))


def fetch_extra_sparql(qid: str, query_body: str) -> list[dict[str, Any]]:
    """Run a targeted SPARQL query for optional entity relationships."""

    if not re.fullmatch(r"Q[1-9][0-9]*", qid):
        return []
    query = (
        "PREFIX wd: <http://www.wikidata.org/entity/>\n"
        "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        + query_body.replace("__QID__", "wd:" + qid)
    )
    url = "https://query.wikidata.org/sparql?format=json&query=" + quote(query)
    try:
        return (http_get_json(url, timeout=15) or {}).get("results", {}).get("bindings", [])
    except Exception as exc:  # Optional enrichment must not block M3.
        log_message(f"Optional Wikidata enrichment failed for {qid}: {exc}")
        return []


def linked_entities_from_rows(
    rows: list[dict[str, Any]], uri_field: str, it_label_field: str, en_label_field: str
) -> list[dict[str, str]]:
    """Collect linked Wikidata entities with a preferred Italian label."""

    found: dict[str, dict[str, str]] = {}
    for row in rows:
        uri = row.get(uri_field, {}).get("value", "")
        qid = uri.rsplit("/", 1)[-1]
        if not re.fullmatch(r"Q[1-9][0-9]*", qid):
            continue
        label = row.get(it_label_field, {}).get("value") or row.get(en_label_field, {}).get("value") or qid
        if qid not in found or found[qid]["label"] == qid:
            found[qid] = {"qid": qid, "label": label, "url": build_wikidata_link(qid)}
    return list(found.values())


@lru_cache(maxsize=2048)
def fetch_epigraph_details(qid: str) -> dict[str, Any]:
    rows = fetch_extra_sparql(
        qid,
        """
SELECT DISTINCT ?inscriptionText ?inscriptionDedicatedTo
                ?inscriptionWork ?workItLabel ?workEnLabel
                ?inscriptionClient ?clientItLabel ?clientEnLabel
WHERE {
  OPTIONAL { __QID__ wdt:P1684 ?inscriptionText. }
  OPTIONAL { __QID__ wdt:P825 ?inscriptionDedicatedTo. }
  OPTIONAL {
    __QID__ wdt:P1574 ?inscriptionWork.
    OPTIONAL { ?inscriptionWork rdfs:label ?workItLabel. FILTER(LANG(?workItLabel) = "it") }
    OPTIONAL { ?inscriptionWork rdfs:label ?workEnLabel. FILTER(LANG(?workEnLabel) = "en") }
  }
  OPTIONAL {
    __QID__ wdt:P88 ?inscriptionClient.
    OPTIONAL { ?inscriptionClient rdfs:label ?clientItLabel. FILTER(LANG(?clientItLabel) = "it") }
    OPTIONAL { ?inscriptionClient rdfs:label ?clientEnLabel. FILTER(LANG(?clientEnLabel) = "en") }
  }
}
LIMIT 5000
""",
    )
    result: dict[str, Any] = {}
    texts = wikidata_values(rows, "inscriptionText")
    dedicatees = qids_from_bindings(rows, "inscriptionDedicatedTo")
    works = linked_entities_from_rows(rows, "inscriptionWork", "workItLabel", "workEnLabel")
    clients = linked_entities_from_rows(rows, "inscriptionClient", "clientItLabel", "clientEnLabel")
    if texts:
        result["inscriptionText"] = texts[0] if len(texts) == 1 else texts
    if dedicatees:
        result["inscriptionDedicatedTo"] = dedicatees[0] if len(dedicatees) == 1 else dedicatees
    if works:
        result["inscriptionWorkReference"] = works
    if clients:
        result["inscriptionClient"] = clients
    return result


@lru_cache(maxsize=2048)
def fetch_archaeological_parts(qid: str) -> list[dict[str, str]]:
    rows = fetch_extra_sparql(
        qid,
        """
SELECT DISTINCT ?part ?partItLabel ?partEnLabel WHERE {
  { __QID__ wdt:P9047 ?part. }
  UNION { __QID__ wdt:P527 ?part. }
  UNION { ?part wdt:P361 __QID__. }
  FILTER(STRSTARTS(STR(?part), "http://www.wikidata.org/entity/Q"))
  OPTIONAL { ?part rdfs:label ?partItLabel. FILTER(LANG(?partItLabel) = "it") }
  OPTIONAL { ?part rdfs:label ?partEnLabel. FILTER(LANG(?partEnLabel) = "en") }
}
LIMIT 5000
""",
    )
    return linked_entities_from_rows(rows, "part", "partItLabel", "partEnLabel")


def add_special_entity_fields(entity: dict[str, Any], bindings: list[dict[str, Any]]) -> None:
    """Add optional archaeological, epigraphic, and location relationships."""

    types = set(entity.get("type", []))
    if types.intersection(BUILDING_TYPES):
        located = qids_from_bindings(bindings, "locatedIn")
        if located:
            entity["locatedIn"] = located[0]
    if types.intersection(EPIGRAPH_TYPES):
        entity.update(fetch_epigraph_details(entity["_id"]))
    if types.intersection(ARCHAEOLOGICAL_SITE_TYPES):
        parts = fetch_archaeological_parts(entity["_id"])
        if parts:
            entity["partsOfSite"] = parts


def extract_temporal_fields(entity: dict[str, Any], bindings: list[dict[str, Any]]) -> None:
    """Preserve explicit historical periods and construction time spans."""

    periods: dict[str, dict[str, str | None]] = {}
    for row in bindings:
        uri = row.get("historicalPeriod", {}).get("value", "")
        if not uri:
            continue
        qid = uri.rsplit("/", 1)[-1]
        periods[qid] = {"qid": qid, "label": row.get("historicalPeriodLabel", {}).get("value")}
    entity["historicalPeriods"] = list(periods.values())
    entity["historicalPeriod"] = next(iter(periods.values()), None)
    entity["timeSpan"] = None
    if not PHYSICAL_SPECIFIC_TYPES.intersection(entity.get("type", [])):
        return
    starts = wikidata_values(bindings, "constructionStart") or wikidata_values(bindings, "foundation")
    ends = wikidata_values(bindings, "constructionEnd")
    entity["timeSpan"] = {
        "start": starts[0] if starts else None,
        "end": ends[0] if ends else None,
        "startSource": "P793/P580" if wikidata_values(bindings, "constructionStart") else ("P571" if starts else None),
        "endSource": "P793/P582" if ends else None,
    }


def fetch_qlever_polygon(qid: str) -> str:
    """
    Fetch a polygon/multipolygon WKT from the OSM QLever endpoint by using the
    Wikidata id as bridge key. Cached for the run, like the Wikidata lookup.
    """
    geometry = _GEOMETRY_CACHE.setdefault(qid, {})
    if "polygon" in geometry:
        return geometry["polygon"]
    if qid in _POLYGON_CACHE:
        geometry["polygon"] = _POLYGON_CACHE[qid]
        return geometry["polygon"]
    query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX osm: <https://www.openstreetmap.org/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX osm2rdfkey: <https://osm2rdf.cs.uni-freiburg.de/rdf/key#>

SELECT ?osm_id ?wkt WHERE {{
  ?osm_id osm2rdfkey:wikidata wd:{qid} .
  ?osm_id rdf:type osm:relation .
  ?osm_id geo:hasGeometry/geo:asWKT ?wkt .
  FILTER(
    STRSTARTS(STR(?wkt), "POLYGON")
    || STRSTARTS(STR(?wkt), "MULTIPOLYGON")
  )
}}
LIMIT 1
"""
    data = http_post_form_json("https://qlever.dev/api/osm-planet", {"query": query}, timeout=30) or {}
    bindings = data.get("results", {}).get("bindings", [])
    wkt = bindings[0].get("wkt", {}).get("value", "") if bindings else ""
    if data:  # "no polygon" is a valid answer and is cached; a failed request ({}) is not
        _POLYGON_CACHE[qid] = wkt
        geometry["polygon"] = wkt
    return wkt


SPECIFIC_CLASS_PRIORITY = (
    ("artisan", "Q1294787"),
    ("music sheet", "Q187947"),
    ("epigraph", "Q1640824"),
    ("rock engraving", "Q42195"),
    ("archaeological site", "Q839954"),
    ("archaeological site", "Q120964921"),
    ("castle", "Q23413"),
    ("monastery", "Q44613"),
    ("church", "Q16970"),
)

PHYSICAL_SPECIFIC_TYPES = {
    "Q34379", "Q187947", "Q16970", "Q16560", "Q33506", "Q3947",
    "Q23413", "Q44613", "Q1640824", "Q42195",
}
ARCHAEOLOGICAL_SITE_TYPES = {"Q839954", "Q120964921"}
EPIGRAPH_TYPES = {"Q1640824"}
BUILDING_TYPES = {"Q41176", "Q33506", "Q16970", "Q3947", "Q16560", "Q23413", "Q44613"}


def classify_entity_general(qid_types: list[str]) -> str:
    """Map the original broad Wikidata taxonomy to frontend classes."""
    if "Q27096213" in qid_types or "Q17334923" in qid_types:
        return "place"
    if "Q5" in qid_types:
        return "person"
    if "Q43229" in qid_types:
        return "organization"
    if "Q234460" in qid_types or "Q386724" in qid_types:
        return "work"
    if "Q7184903" in qid_types or "Q4026292" in qid_types or "Q5127848" in qid_types:
        return "concept"
    if any(x in qid_types for x in ["Q41176", "Q8205328", "Q488383", "Q15222213"]):
        return "object"
    return "other"


def classify_entity(entity: dict[str, Any], qid_types: list[str]) -> str:
    """Prefer the new specific Wikidata classes before the old taxonomy."""

    for class_name, type_qid in SPECIFIC_CLASS_PRIORITY:
        if type_qid in qid_types:
            return class_name
    return classify_entity_general(qid_types)


def choose_default_event_style(event_title: str) -> tuple[str, str]:
    """Return the default map color and event type for the current event."""
    map_color = "#ff9900"
    event_type = "valorisation event"
    return map_color, event_type


def derive_narration_title(path: Path) -> str:
    title = path.stem.strip()
    title = re.sub(r"(?:[_\s-]?qids)$", "", title, flags=re.IGNORECASE).strip()
    return title


def build_storymap_html(narration_title: str, id_narra: str) -> str:
    """Create the same basic HTML shell used by the PHP storymap output."""
    return f"""<html>
\t<head>
\t\t<meta charset="UTF-8">
\t\t<title>Narrative - {escape(narration_title)}</title>
\t\t<link rel="stylesheet" type="text/css" href="../../../lib/bootstrap.min.css" />
\t\t<link rel="stylesheet" type="text/css" href="../../../lib/narra.css" />
\t\t<script src="../../../lib/jquery-3.2.1.min.js" type="text/javascript" charset="utf-8"></script>
\t\t<script src="../../../lib/jquery-ui.min.js" type="text/javascript" charset="utf-8"></script>
\t\t<script src="../../../lib/bootstrap.min.js" type="text/javascript" charset="utf-8"></script>
\t\t<script src="../../../lib/typeahead.bundle.min.js" type="text/javascript" charset="utf-8"></script>
\t\t<script src="../../lib/visualization.js" type="text/javascript" charset="utf-8"></script>
\t\t<link rel="stylesheet" href="https://cdn.knightlab.com/libs/storymapjs/0.8.6/css/storymap.css">
\t\t<script type="text/javascript" src="../../../lib/storymap.js"></script>
\t\t<link rel="stylesheet" type="text/css" href="../../../lib/timeline.css" />
\t\t<script src="../../../lib/timeline-min.js" type="text/javascript" charset="utf-8"></script>
\t\t<link rel="stylesheet" type="text/css" href="../../lib/narrativeVisualization.css" />
\t</head>
\t<body>
\t\t<div id="menu">
\t\t\t<div id="titleTable">
\t\t\t<h1>{escape(narration_title)}</h1>
\t\t\t</div>
\t\t\t<div class="otherVisual">
\t\t\t  <button class="dropbtn">Other Visualizations</button>
\t\t\t  <div class="otherVisual-content">
\t\t\t\t<a href="?visualization=map">Storymap</a>
\t\t\t\t<a href="?visualization=timeline">Timeline</a>
\t\t\t  </div>
\t\t\t</div>
\t\t\t<a href="../../../Search/?user={USER_IDENTIFIER}&id=N{id_narra}&dataset={USER_NAME}">
\t\t\t<div class="otherNarratives">
\t\t\t  <button class="dropbtn">Search</button>
\t\t\t</div>
\t\t\t</a>
\t\t</div>
\t\t<div id="mapdiv"></div>
\t</body>
</html>"""


def locate_java() -> str:
    """Find a usable Java executable."""
    if SETTINGS.java_bin:
        configured = Path(SETTINGS.java_bin)
        if configured.exists():
            return str(configured)
        log_message(f"Configured Java executable does not exist: {configured}")

    for candidate in JAVA_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    try:
        completed = subprocess.run(
            ["where.exe", "java"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout.splitlines()[0].strip()
    except Exception as exc:  # pragma: no cover - fallback path
        log_message(f"Unable to detect Java via 'where java': {exc}")

    return ""


def process_csv(
    path: Path,
    id_narra: str,
    csv_key: str,
    narration_title_override: str | None = None,
) -> None:
    """
    Process one narrative CSV and write the storymap/json/owl assets.

    The function mirrors the PHP flow:
    - read rows
    - build slides
    - enrich entities
    - write storymap/json/owl assets
    """
    STORYMAP_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    OWL_DIR.mkdir(parents=True, exist_ok=True)

    rows = parse_csv_rows(path)
    if not rows:
        log_message(f"Empty CSV skipped: {path.name}")
        return

    final_storymap: dict[str, Any] = {"slides": []}
    narration_title = narration_title_override or derive_narration_title(path)
    used_slide_images: set[str] = set()
    used_slide_polygons: set[str] = set()

    for row_index, line in enumerate(rows[1:], start=1):
        if len(line) < 2 or not line[1]:
            continue

        event_title = line[0] if len(line) > 0 else ""
        safe_event_title = escape(event_title)
        if row_index == 1 and not narration_title:
            narration_title = event_title

        entity_labels = (
            [item.strip() for item in line[2].split(",") if item.strip()] if len(line) > 2 else []
        )
        qids = (
            [
                SPECIAL_QID_REPLACEMENTS.get(item.strip(), item.strip())
                for item in line[3].split(",")
                if item.strip()
            ]
            if len(line) > 3
            else []
        )
        entity_links = [build_wikidata_link(qid) for qid in qids]
        event_lon, event_lat = parse_entity_coords(line[4] if len(line) > 4 else "")
        image_candidates = image_candidates_from_cell(line[5] if len(line) > 5 else "")

        if row_index == 1:
            final_storymap["A1"] = {
                "id": id_narra,
                "_id": "A1",
                "name": narration_title,
                "author": USER_IDENTIFIER,
                "idNarra": id_narra,
                "coverImg": STORYMAP_COVER_IMAGE,
            }

        map_color, event_type = choose_default_event_style(event_title)
        arr_props: dict[str, Any] = {}
        entity_giving_coordinates = ""
        entity_giving_Geographiclabel = ""
        polygon_candidates: list[str] = []
        geometry_candidates: list[dict[str, str]] = []
        slide_text = escape(line[1]) + " <h5>Entities</h5> "
        log_message(
            f"Event {row_index} ({event_title}) in {path.name} has {len(entity_links)} entity links"
        )

        for j, qid in enumerate(qids):
            entity_link = entity_links[j] if j < len(entity_links) else build_wikidata_link(qid)
            geometry_cache = _GEOMETRY_CACHE.setdefault(qid, {})

            arr_props[qid] = {
                "class": "other",
                "notes": "",
                "primary": [{"text": "", "title": "", "author": "", "reference": ""}],
                "secondary": [{"text": "", "title": "", "author": "", "reference": ""}],
            }

            wikidata_data = fetch_wikidata_entity(qid)
            bindings = wikidata_data.get("results", {}).get("bindings", [])
            inherited_types = fetch_entity_type_qids(qid)
            if inherited_types:
                bindings = bindings + [
                    {"type": {"value": f"http://www.wikidata.org/entity/{type_qid}"}}
                    for type_qid in inherited_types
                ]

            single_entity = {
                "_id": qid,
                "_rev": "",
                "itName": "",
                "enName": "",
                "itDesc": "",
                "enDesc": "",
                "image": "",
                "coordinatesPoint": geometry_cache.get("point", ""),
                "coordinatesPolygon": geometry_cache.get("polygon", ""),
                "type": [],
                "role": [],
                "timeSpan": None,
                "historicalPeriod": None,
                "historicalPeriods": [],
            }

            if bindings:
                first = bindings[0]
                single_entity["itName"] = first.get("itName", {}).get("value", "")
                single_entity["enName"] = first.get("enName", {}).get("value", "")
                single_entity["itDesc"] = first.get("itDesc", {}).get("value", "")
                single_entity["enDesc"] = first.get("enDesc", {}).get("value", "")
                point = first.get("coordinates", {}).get("value", "")
                if point:
                    geometry_cache["point"] = point
                    single_entity["coordinatesPoint"] = point
                single_entity["image"] = first.get("image", {}).get("value", "")
                entity_image = safe_media_url(single_entity["image"])
                if entity_image and entity_image not in image_candidates:
                    image_candidates.append(entity_image)
                entity_label_for_geometry = (
                    entity_labels[j] if j < len(entity_labels) else (single_entity["enName"] or qid)
                )
                if single_entity["coordinatesPoint"]:
                    geometry_candidates.append(
                        {
                            "qid": qid,
                            "label": entity_label_for_geometry,
                            "type": "point",
                            "geometry": single_entity["coordinatesPoint"],
                        }
                    )

                if single_entity["coordinatesPoint"]:
                    point_inner = single_entity["coordinatesPoint"]
                    if "(" in point_inner and ")" in point_inner:
                        point_inner = point_inner[
                            point_inner.find("(") + 1 : point_inner.rfind(")")
                        ]
                        parts = point_inner.split()
                        if len(parts) >= 2 and event_lon and event_lat:
                            if parts[0] == event_lon and parts[1] == event_lat:
                                print((single_entity["enName"] or qid) + "</br>")
                                entity_giving_coordinates = qid
                                entity_giving_Geographiclabel = (
                                    "<a href='https://www.wikidata.org/wiki/"
                                    + escape(qid)
                                    + "' target='_blank'>"
                                    + escape(single_entity["enName"])
                                    + "</a>"
                                )

                    single_entity["coordinatesPolygon"] = fetch_qlever_polygon(qid)
                    if single_entity["coordinatesPolygon"]:
                        polygon_candidates.append(single_entity["coordinatesPolygon"])
                        geometry_candidates.append(
                            {
                                "qid": qid,
                                "label": entity_label_for_geometry,
                                "type": "polygon",
                                "geometry": single_entity["coordinatesPolygon"],
                            }
                        )

                if event_title not in {
                    "Interest on this VC",
                    "Mountain landscape and reference chains",
                    "Key local assets",
                    "Challenges",
                    "Innovation",
                    "Geography and population",
                    "Income and gross value added",
                    "Tourism",
                    "Employment",
                    "Concluding remarks",
                }:
                    event_type = "natural event"
                    map_color = "#2eb82e"

                type_qids = list(
                    dict.fromkeys(
                        qids_from_bindings(bindings, "type")
                        + qids_from_bindings(bindings, "specificType")
                        + qids_from_bindings(bindings, "occupationType")
                    )
                )
                single_entity["type"] = type_qids
                extract_temporal_fields(single_entity, bindings)
                add_special_entity_fields(single_entity, bindings)
                arr_props[qid]["class"] = classify_entity(single_entity, type_qids)

            final_storymap.setdefault("items", {})[qid] = single_entity
            entity_label = (
                entity_labels[j] if j < len(entity_labels) else (single_entity["enName"] or qid)
            )
            slide_text += (
                f"<span class='tl-entities'><a onmouseover='$(this).tooltip(); $(this).tooltip(\"show\")' "
                f"data-toggle='tooltip' title='{escape(single_entity['enDesc'])}' target='_blank' href='{escape(entity_link)}'>"
                f"{escape(entity_label)}</a></span>"
            )
            if j < len(entity_links) - 1:
                slide_text += " • "

        if not event_lon or not event_lat:
            for qid in qids:
                point_lon, point_lat = parse_wkt_point(
                    final_storymap.get("items", {}).get(qid, {}).get("coordinatesPoint", "")
                )
                if point_lon and point_lat:
                    event_lon, event_lat = point_lon, point_lat
                    entity_giving_coordinates = qid
                    break

        polygon = ""
        if entity_giving_coordinates:
            entity_info = final_storymap.get("items", {}).get(entity_giving_coordinates, {})
            if entity_info.get("coordinatesPolygon"):
                polygon = entity_info["coordinatesPolygon"]
                zoom = 5
                polygon_candidates.insert(0, polygon)
            elif entity_info.get("coordinatesPoint"):
                zoom = 13
            else:
                zoom = 13
        else:
            zoom = 13

        selected_image = choose_non_repeated_asset(image_candidates, used_slide_images)
        selected_polygons = choose_non_repeated_assets(polygon_candidates, used_slide_polygons)
        selected_polygon_set = set(selected_polygons)
        selected_geometries = [
            geometry
            for geometry in geometry_candidates
            if geometry["type"] != "polygon" or geometry["geometry"] in selected_polygon_set
        ]
        polygon = (selected_polygons[0] if selected_polygons else "") or polygon
        if polygon:
            zoom = 5

        has_coordinates = bool(event_lon and event_lat)
        slide_type = choose_slide_type_with_llm(
            event_title, line[1], entity_labels, has_coordinates
        )
        if slide_type == SLIDE_TYPE_NORMAL:
            event_lon = ""
            event_lat = ""
            polygon = ""
            selected_polygons = []
            selected_geometries = []
            zoom = 0

        event_id = f"ev{row_index}"
        json_object = {
            "_id": event_id,
            "text": {"text": slide_text, "headline": safe_event_title},
            "location": {
                "name": "",
                "lat": float(event_lat) if event_lat else 0.0,
                "lon": float(event_lon) if event_lon else 0.0,
                "zoom": zoom,
                "line": slide_type == SLIDE_TYPE_GEOGRAPHIC,
            },
            "media": {"url": selected_image},
            "date": "",
            "title": safe_event_title,
            "latitud": event_lat,
            "start": "",
            "end": "",
            "objurl": "",
            "end_date": {"year": "null", "month": "", "day": ""},
            "start_date": {"year": "null", "month": "", "day": ""},
            "type": event_type,
            "slideType": slide_type,
            "longitud": event_lon,
            "unique_id": f"slide-{event_id}",
            "eventMedia": selected_image,
            "eventMediaCaption": "",
            "notes": "",
            "description": escape(line[1]),
            "position": row_index,
            "props": arr_props,
            "mapMarkerColor": map_color,
            "polygon": polygon,
            "polygons": selected_polygons,
            "eventGeometries": selected_geometries,
            "backgroundImg": "https://tool.dlnarratives.eu/images/slideBlack.png",
            "eventPlaceLabel": entity_giving_Geographiclabel,
        }

        final_storymap.setdefault("events", {})[event_id] = json_object
        final_storymap["slides"].append(json_object)
        if selected_image:
            used_slide_images.add(selected_image)
        if polygon:
            used_slide_polygons.add(polygon)

    if not narration_title:
        log_message(f"No valid events found in {path.name}")
        return

    # Write storymap files.
    narrative_dir = STORYMAP_DIR / f"N{id_narra}"
    narrative_dir.mkdir(parents=True, exist_ok=True)
    (narrative_dir / "slide.json").write_text(
        json.dumps(final_storymap, ensure_ascii=False), encoding="utf-8"
    )
    (narrative_dir / f"{path.name}.txt").write_text("", encoding="utf-8")
    (narrative_dir / "index.html").write_text(
        build_storymap_html(narration_title, id_narra), encoding="utf-8"
    )
    log_message(f"Wrote storymap assets for N{id_narra}")

    # Write Triplify JSON.
    triplify_payload = {
        "entities": final_storymap.get("items", {}),
        "narra": final_storymap.get("A1", {}),
        "events": final_storymap.get("events", {}),
    }
    json_output = JSON_DIR / f"{USER_IDENTIFIER}_N{id_narra}.json"
    json_output.write_text(json.dumps(triplify_payload, ensure_ascii=False), encoding="utf-8")
    log_message(f"Wrote Triplify JSON {json_output.name}")

    # Run Triplify if Java is available.
    java_bin = locate_java()
    if java_bin:
        owl_output = OWL_DIR / f"{USER_IDENTIFIER}_N{id_narra}.owl"
        cmd = [java_bin, "-jar", str(TRIPLIFY_JAR), str(json_output), str(owl_output)]
        log_message(f"Running Triplify for N{id_narra}")
        completed = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        log_message(f"Triplify finished for N{id_narra} with exit code {completed.returncode}")
        if completed.returncode != 0 and SETTINGS.require_owl:
            raise RuntimeError(
                f"Triplify failed for N{id_narra} with exit code {completed.returncode}"
            )
    else:
        print(f"Java not found in PATH. Skipping Triplify for N{id_narra}</br>")
        log_message(f"Java not found, Triplify skipped for N{id_narra}")
        if SETTINGS.require_owl:
            raise RuntimeError("Java not found and OWL output is required")


def main(csv_path: str | None = None, title: str | None = None) -> int:
    """Entry point of the Python port for one explicitly selected CSV."""
    SETTINGS.validate()
    ensure_io_dirs()
    STORYMAP_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    OWL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    selected_csv = Path(csv_path).resolve() if csv_path else resolve_default_input_csv_path()
    if selected_csv.suffix.lower() != ".csv" or not selected_csv.stem.lower().endswith("_qids"):
        print(f"Input must be a *_qids.csv file: {selected_csv}")
        log_message(f"Invalid input CSV name: {selected_csv}")
        return 1
    log_message(f"Starting Python script on CSV: {selected_csv}")

    csv_key = selected_csv.stem.strip()

    if not selected_csv.exists():
        print(f"CSV file not found: {selected_csv}")
        log_message(f"CSV file not found: {selected_csv}")
        return 1

    if has_existing_outputs(csv_key):
        print(f"Outputs already exist for {csv_key}, file skipped")
        log_message(f"Outputs already exist for {csv_key}, file skipped")
        return 0

    id_narra = f"1_{csv_key}"
    log_message(f"Processing CSV {selected_csv.name} as narrative N{id_narra}")

    process_csv(selected_csv, id_narra, csv_key, narration_title_override=title)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate StoryMap, JSON, and OWL artifacts")
    parser.add_argument("input_csv", nargs="?", help="Input *_qids.csv produced by M2")
    parser.add_argument("--title", help="Optional narrative title override")
    args = parser.parse_args()
    raise SystemExit(main(args.input_csv, title=args.title))
