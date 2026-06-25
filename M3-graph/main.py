from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from langchain_community.llms import Ollama


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

# Choose here the only CSV that the script must process.
CSV_PATH = INPUT_DIR / "VC_01_AT_qids.csv"

STORYMAP_DIR = OUTPUT_DIR / "HTML_Storymap"
JSON_DIR = OUTPUT_DIR / "json"
OWL_DIR = OUTPUT_DIR / "owl"
TRIPLIFY_JAR = BASE_DIR / "Triplify" / "Triplify.jar"
LOG_PATH = BASE_DIR / "scriptStories5OSM_py.log"
USER_NAME = "moving"
USER_ID = 52
USER_IDENTIFIER = f"{USER_NAME}.{USER_ID}"
M3_OLLAMA_MODEL = os.getenv("AUTOMATIC_WORKFLOW_M3_OLLAMA_MODEL", SETTINGS.ollama_model)
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
        llm = Ollama(
            model=M3_OLLAMA_MODEL,
            system="You classify storymap slides. Answer only with geographic or normal.",
            num_ctx=2048,
            temperature=0.0,
        )
        response = str(llm.invoke(prompt)).strip()
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
    js = list(JSON_DIR.glob(f"{USER_IDENTIFIER}_N*_{csv_key}.json"))
    owl = list(OWL_DIR.glob(f"{USER_IDENTIFIER}_N*_{csv_key}.owl"))
    return bool(story and js and owl)


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


def fetch_wikidata_entity(qid: str) -> dict[str, Any]:
    """
    Fetch labels, descriptions, image, coordinates and coarse type information
    from Wikidata for one entity.
    """
    qid = SPECIAL_QID_REPLACEMENTS.get(qid, qid)
    query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
SELECT DISTINCT ?uri ?coordinates ?type ?itName ?enName ?itDesc ?enDesc ?image ?imgMappa ?birth ?death ?foundation ?foundation2 ?completion ?occupation ?position
WHERE {{
VALUES ?uri {{wd:{qid}}}
OPTIONAL {{?uri wdt:P31 ?class.}}
OPTIONAL {{
  ?class wdt:P279* ?type.
  VALUES ?type {{
    wd:Q15222213 wd:Q17334923 wd:Q43229 wd:Q8436 wd:Q488383 wd:Q7184903 wd:Q386724 wd:Q234460 wd:Q5 wd:Q186081 wd:Q1190554 wd:Q35120 wd:Q15474042 wd:Q4167836 wd:Q41176 wd:Q8205328 wd:Q5127848 wd:Q27096213
  }}
}}
OPTIONAL {{ ?uri wdt:P18 ?image. }}
OPTIONAL {{ ?uri wdt:P569 ?birth. }}
OPTIONAL {{ ?uri wdt:P570 ?death. }}
OPTIONAL {{ ?uri wdt:P571 ?foundation. }}
OPTIONAL {{ ?uri wdt:P580 ?foundation2. }}
OPTIONAL {{ ?uri wdt:P1619 ?completion. }}
OPTIONAL {{ ?uri wdt:P106 ?occupation. }}
OPTIONAL {{ ?uri wdt:P39 ?position. }}
OPTIONAL {{ ?uri rdfs:label ?itName filter (lang(?itName) = 'it'). }}
OPTIONAL {{ ?uri rdfs:label ?enName filter (lang(?enName) = 'en'). }}
OPTIONAL {{ ?uri schema:description ?itDesc filter (lang(?itDesc) = 'it'). }}
OPTIONAL {{ ?uri schema:description ?enDesc filter (lang(?enDesc) = 'en'). }}
OPTIONAL {{ ?uri wdt:P242 ?imgMappa. }}
OPTIONAL {{ ?uri wdt:P625 ?coordinates. }}
}} limit 50000
"""
    url = "https://query.wikidata.org/sparql?format=json&query=" + quote(query)
    return http_get_json(url) or {}


def fetch_qlever_polygon(qid: str) -> str:
    """
    Fetch a polygon/multipolygon WKT from the OSM QLever endpoint by using the
    Wikidata id as bridge key.
    """
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
    data = http_post_form_json("https://qlever.dev/api/osm-planet", {"query": query}) or {}
    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return ""
    return bindings[0].get("wkt", {}).get("value", "")


def classify_entity(entity: dict[str, Any], qid_types: list[str]) -> str:
    """Map Wikidata type ids to the frontend classes used by the storymap."""
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


def process_csv(path: Path, id_narra: str, csv_key: str) -> None:
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
    narration_title = derive_narration_title(path)
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
                "coverImg": "https://tool.dlnarratives.eu/images/cover.png",
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

            arr_props[qid] = {
                "class": "other",
                "notes": "",
                "primary": [{"text": "", "title": "", "author": "", "reference": ""}],
                "secondary": [{"text": "", "title": "", "author": "", "reference": ""}],
            }

            wikidata_data = fetch_wikidata_entity(qid)
            bindings = wikidata_data.get("results", {}).get("bindings", [])

            single_entity = {
                "_id": qid,
                "_rev": "",
                "itName": "",
                "enName": "",
                "itDesc": "",
                "enDesc": "",
                "image": "",
                "coordinatesPoint": "",
                "coordinatesPolygon": "",
                "type": [],
                "role": [],
            }

            if bindings:
                first = bindings[0]
                single_entity["itName"] = first.get("itName", {}).get("value", "")
                single_entity["enName"] = first.get("enName", {}).get("value", "")
                single_entity["itDesc"] = first.get("itDesc", {}).get("value", "")
                single_entity["enDesc"] = first.get("enDesc", {}).get("value", "")
                single_entity["coordinatesPoint"] = first.get("coordinates", {}).get("value", "")
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

                type_qids = [
                    binding.get("type", {}).get("value", "").rsplit("/", 1)[-1]
                    for binding in bindings
                    if binding.get("type", {}).get("value")
                ]
                single_entity["type"] = type_qids
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


def main(csv_path: str | None = None) -> int:
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

    process_csv(selected_csv, id_narra, csv_key)
    return 0


if __name__ == "__main__":
    input_csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(main(input_csv_path))
