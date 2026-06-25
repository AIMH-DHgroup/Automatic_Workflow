import argparse
import os
import csv
import re
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import requests
from json_repair import repair_json

from langchain_community.llms import Ollama
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from automatic_workflow.config import WorkflowSettings  # noqa: E402


# -------------------------
# CONFIG
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS = WorkflowSettings.from_env(PROJECT_ROOT)
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

AVAILABLE_M2_MODELS = (
    "gemma2:9b-instruct-q8_0",
    "llama3.1:8b",
    "llama3.1:8b-instruct-q8_0",
    "llama3:8b",
    "mistral:7b",
    "qwen2.5:7b",
    "phi3:medium",
)
SELECTED_OLLAMA_MODEL = os.getenv("AUTOMATIC_WORKFLOW_M2_OLLAMA_MODEL", SETTINGS.ollama_model)


SYSTEM_PROMPT = """recognize the entities in the text and, for each of them, find the exact title corresponding to the Wikipedia page. The final result should be a json like this:

### Json
{
  "entities": [
    {
      "entity_in_the_text": "...",
      "wikipedia_title": "..."
    }
  ]
}

Answer only with the json
"""

# Wikipedia (enwiki)
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
HTTP_TIMEOUT = 20

# Wikidata SPARQL
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

# IMPORTANT: contatto reale per evitare 403/429
USER_AGENT = SETTINGS.wikidata_user_agent
ACCEPT_LANGUAGE = "en"

SLEEP_BETWEEN_ROWS_SEC = 0.0
LLM_MAX_RETRIES = 1

# Output CSV options (Excel-friendly)
OUTPUT_DELIMITER = ";"
OUTPUT_ENCODING = "utf-8-sig"

# Separators inside the output cells
ENTITY_SEP = ","   # separates entities (aligns with QIDs order)
MULTI_SEP = "|"    # separates multiple images for the same entity


def ensure_io_dirs() -> None:
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_default_input_csv_path() -> str:
    ensure_io_dirs()
    candidates: List[str] = []
    for name in sorted(os.listdir(INPUT_DIR)):
        path = os.path.join(INPUT_DIR, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith(".csv") or name.lower().endswith("_qids.csv"):
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No input CSV files found in {INPUT_DIR}")
    return candidates[0]


def available_m2_models() -> tuple[str, ...]:
    """Return Ollama models supported by the M2 command-line selector."""

    return AVAILABLE_M2_MODELS


def configure_m2_model(model: str | None) -> str:
    """Configure the Ollama model used by M2 entity extraction.

    Args:
        model: Optional model selected by the user. When omitted, the existing
            environment/default model is kept.

    Returns:
        The model name that will be used.

    Raises:
        ValueError: If the selected model is not in `AVAILABLE_M2_MODELS`.
    """

    global SELECTED_OLLAMA_MODEL

    selected = (model or SELECTED_OLLAMA_MODEL).strip()
    if selected not in AVAILABLE_M2_MODELS:
        supported = ", ".join(AVAILABLE_M2_MODELS)
        raise ValueError(f"Unsupported M2 model '{selected}'. Supported models: {supported}")
    SELECTED_OLLAMA_MODEL = selected
    return SELECTED_OLLAMA_MODEL


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse M2 command-line arguments while preserving positional input usage."""

    parser = argparse.ArgumentParser(description="Enrich narrative CSV rows with Wikidata entities")
    parser.add_argument("input_csv", nargs="?", help="Narrative CSV produced by M1")
    parser.add_argument(
        "--model",
        choices=AVAILABLE_M2_MODELS,
        help="Ollama model to use for entity extraction",
    )
    parser.add_argument("--list-models", action="store_true", help="Print supported M2 models and exit")
    return parser.parse_args(argv)


# -------------------------
# LLM
# -------------------------
def useLLM(text: str) -> str:
    llm = Ollama(
        model=SELECTED_OLLAMA_MODEL,
        system=SYSTEM_PROMPT,
        num_ctx=4096,
        temperature=0.01,
    )
    response = llm.invoke(text, config={"callbacks": [StreamingStdOutCallbackHandler()]})
    print()
    return response


# -------------------------
# JSON EXTRACT + REPAIR
# -------------------------
def estrai_json_da_stringa_ripara(stringa: str) -> Optional[str]:
    """
    Estrae il primo blocco { ... } dalla risposta dell'LLM,
    ripulisce errori comuni e prova a riparare con json_repair.
    Ritorna una stringa JSON valida o None.
    """
    if not stringa or not stringa.strip():
        return None

    s = stringa.strip()

    # remove markdown fences if any
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    match = re.search(r"\{.*\}", s, re.DOTALL)
    if not match:
        return None

    json_string = match.group(0)

    # remove trailing commas before } or ]
    json_string_pulito = re.sub(r",\s*(\}|\])", r"\1", json_string)

    # direct parse attempt
    try:
        json.loads(json_string_pulito)
        return json_string_pulito
    except Exception:
        pass

    # repair attempt
    try:
        good_json_string = repair_json(json_string_pulito)
        json.loads(good_json_string)
        return good_json_string
    except Exception:
        return None


def parse_keywords_from_llm_output(llm_output: str) -> List[Dict[str, str]]:
    fixed = estrai_json_da_stringa_ripara(llm_output)
    if not fixed:
        return []

    try:
        data = json.loads(fixed)
    except Exception:
        return []

    kws = data.get("entities", [])
    if not isinstance(kws, list):
        return []

    out: List[Dict[str, str]] = []
    for item in kws:
        if not isinstance(item, dict):
            continue
        kw = str(item.get("entity_in_the_text", "")).strip()
        title = str(item.get("wikipedia_title", "")).strip()
        if title:
            out.append({"entity_in_the_text": kw, "wikipedia_title": title})
    return out


def llm_keywords_safe(text: str) -> Tuple[List[Dict[str, str]], str]:
    last_output = ""
    for attempt in range(LLM_MAX_RETRIES + 1):
        last_output = useLLM(text)
        items = parse_keywords_from_llm_output(last_output)
        if items:
            return items, last_output
        time.sleep(0.2 * (attempt + 1))
    return [], last_output


# -------------------------
# HTTP (SESSION + RETRIES)
# -------------------------
def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": ACCEPT_LANGUAGE,
        }
    )
    return s


def _get_with_retries(
    session: requests.Session,
    url: str,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    max_retries: int = 6,
    base_sleep: float = 1.0,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)

            if r.status_code in (403, 429, 500, 502, 503, 504):
                sleep = base_sleep * (2 ** attempt)
                ra = r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    sleep = max(sleep, float(ra))
                time.sleep(sleep)
                continue

            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            time.sleep(base_sleep * (2 ** attempt))

    if last_exc:
        raise last_exc
    raise RuntimeError("HTTP request failed (unexpected).")


# -------------------------
# WIKIPEDIA TITLE -> WIKIDATA QID
# -------------------------
def fetch_wikidata_qids_from_titles(titles: List[str]) -> Dict[str, Optional[str]]:
    seen = set()
    unique_titles: List[str] = []
    for t in titles:
        t2 = (t or "").strip()
        if t2 and t2 not in seen:
            seen.add(t2)
            unique_titles.append(t2)

    if not unique_titles:
        return {}

    result: Dict[str, Optional[str]] = {t: None for t in unique_titles}
    session = _make_session()

    chunk_size = 40
    for i in range(0, len(unique_titles), chunk_size):
        chunk = unique_titles[i : i + chunk_size]

        params = {
            "action": "query",
            "format": "json",
            "redirects": 1,
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "titles": "|".join(chunk),
        }

        r = _get_with_retries(session, WIKIPEDIA_API, params=params)
        data = r.json()

        pages = (data.get("query", {}) or {}).get("pages", {}) or {}
        for _, page in pages.items():
            title_returned = page.get("title")
            pageprops = page.get("pageprops") or {}
            qid = pageprops.get("wikibase_item")

            if not title_returned:
                continue

            if title_returned in result:
                result[title_returned] = qid

            normalized = title_returned.replace("_", " ").strip().lower()
            for t in chunk:
                if t.replace("_", " ").strip().lower() == normalized:
                    result[t] = qid

        redirects = (data.get("query", {}) or {}).get("redirects", []) or []
        redir_map = {
            d.get("from"): d.get("to")
            for d in redirects
            if d.get("from") and d.get("to")
        }
        for src, dst in redir_map.items():
            if src in result and result[src] is None:
                if dst in result and result[dst]:
                    result[src] = result[dst]
                else:
                    dst_norm = dst.replace("_", " ").strip().lower()
                    for t in chunk:
                        if t.replace("_", " ").strip().lower() == dst_norm and result.get(t):
                            result[src] = result[t]

    return result


# -------------------------
# WIKIDATA QID -> (label_en, images[], coords)
# -------------------------
_COORD_RE = re.compile(r"Point\(([-0-9.]+)\s+([-0-9.]+)\)")


def _qid_ok(q: str) -> bool:
    return bool(re.fullmatch(r"Q[1-9]\d*", (q or "").strip()))


def fetch_wikidata_details_by_qids(qids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Returns mapping:
    {
      "Q42": {"label_en": "...", "images": ["url1","url2",...], "coord": "lat lon" or None},
      ...
    }
    """
    # dedup & validate, keep order
    seen = set()
    uq: List[str] = []
    for q in qids:
        q = (q or "").strip()
        if _qid_ok(q) and q not in seen:
            seen.add(q)
            uq.append(q)

    if not uq:
        return {}

    session = _make_session()
    out: Dict[str, Dict[str, Any]] = {q: {"label_en": None, "images": [], "coord": None} for q in uq}

    chunk_size = 50
    for i in range(0, len(uq), chunk_size):
        chunk = uq[i : i + chunk_size]
        values = " ".join(f"wd:{q}" for q in chunk)

        query = f"""
SELECT ?item ?itemLabel ?image ?coord WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item wdt:P18 ?image. }}
  OPTIONAL {{ ?item wdt:P625 ?coord. }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
"""

        headers = {"Accept": "application/sparql-results+json"}
        r = _get_with_retries(session, WIKIDATA_SPARQL, params={"query": query}, headers=headers)
        data = r.json()

        bindings = (data.get("results", {}) or {}).get("bindings", []) or []
        for b in bindings:
            item_uri = (b.get("item", {}) or {}).get("value", "")
            m = re.search(r"/entity/(Q\d+)$", item_uri)
            if not m:
                continue
            qid = m.group(1)
            if qid not in out:
                continue

            label = (b.get("itemLabel", {}) or {}).get("value")
            image = (b.get("image", {}) or {}).get("value")
            coord_val = (b.get("coord", {}) or {}).get("value")

            if label and not out[qid]["label_en"]:
                out[qid]["label_en"] = label

            if image:
                out[qid]["images"].append(image)

            if coord_val and not out[qid]["coord"]:
                mm = _COORD_RE.search(coord_val)
                if mm:
                    lon = mm.group(1)
                    lat = mm.group(2)
                    # IMPORTANT: no commas here, so we can safely join entities with commas
                    out[qid]["coord"] = f"{lat} {lon}"
                else:
                    out[qid]["coord"] = coord_val

    # dedup images per QID, preserving order
    for q in out:
        imgs = out[q].get("images") or []
        seen_i = set()
        dedup = []
        for u in imgs:
            if u not in seen_i:
                seen_i.add(u)
                dedup.append(u)
        out[q]["images"] = dedup

    return out


# -------------------------
# CSV INPUT
# -------------------------
def read_input_csv_rows(path: str) -> List[List[str]]:
    # M2 consumes the narrative CSV produced by M1, which is always comma-separated
    # and uses standard doubled quotes for embedded ". Avoid Sniffer because it may
    # infer `doublequote=False` on long narrative rows and split descriptions badly.
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"', doublequote=True)
        return list(reader)


# -------------------------
# MAIN CSV PROCESSOR
# -------------------------
def enrich_csv_with_wikidata_qids(input_csv_path: str) -> str:
    ensure_io_dirs()
    if not os.path.isfile(input_csv_path):
        raise FileNotFoundError(f"File not found: {input_csv_path}")

    input_name = os.path.basename(input_csv_path)
    base, ext = os.path.splitext(input_name)
    output_path = os.path.join(OUTPUT_DIR, f"{base}_qids{ext}")

    rows = read_input_csv_rows(input_csv_path)

    if not rows:
        raise ValueError("CSV is empty.")

    header = rows[0]
    out_rows: List[List[str]] = []
    out_rows.append(header + ["wikidata_names_en", "wikidata_qids", "wikidata_coords", "wikidata_images"])

    for row_idx in range(1, len(rows)):
        row = rows[row_idx]

        text = ""
        if len(row) > 1 and row[1] is not None:
            text = str(row[1]).strip()

        if not text:
            out_rows.append(row + ["", "", "", ""])
            print(f"[row {row_idx}] 2nd column empty -> no QIDs")
            continue

        # 1) LLM -> titles
        keyword_items, _raw_llm = llm_keywords_safe(text)
        if not keyword_items:
            snippet = text[:120].replace("\n", " ")
            print(f"[row {row_idx}] No JSON keywords from LLM -> empty fields | text_snippet='{snippet}'")
            out_rows.append(row + ["", "", "", ""])
            continue

        titles = [it["wikipedia_title"].strip() for it in keyword_items if it.get("wikipedia_title")]

        # 2) Wikipedia -> QIDs
        try:
            title_to_qid = fetch_wikidata_qids_from_titles(titles)
        except Exception as e:
            print(f"[row {row_idx}] Wikipedia fetch failed: {e} | titles={titles}")
            out_rows.append(row + ["", "", "", ""])
            continue

        # keep QIDs ordered & dedup
        qids: List[str] = []
        seen_q = set()
        for t in titles:
            q = title_to_qid.get(t)
            if q and q not in seen_q:
                seen_q.add(q)
                qids.append(q)

        if not qids:
            out_rows.append(row + ["", "", "", ""])
            continue

        # 3) Wikidata SPARQL -> details (label, images[], coords)
        try:
            details = fetch_wikidata_details_by_qids(qids)
        except Exception as e:
            print(f"[row {row_idx}] Wikidata SPARQL failed: {e} | qids={qids}")
            out_rows.append(row + [ENTITY_SEP.join(qids), "", "", ""])
            continue

        names_en: List[str] = []
        images_cell: List[str] = []
        coords: List[str] = []

        for q in qids:
            d = details.get(q, {})
            name = d.get("label_en") or "null"

            imgs = d.get("images") or []
            if imgs:
                # multiple images per entity separated by MULTI_SEP
                img_cell = MULTI_SEP.join(imgs)
            else:
                img_cell = "null"

            coord = d.get("coord") or "null"

            names_en.append(name)
            images_cell.append(img_cell)
            coords.append(coord)

        out_rows.append(
            row
            + [
                ENTITY_SEP.join(names_en),
                ENTITY_SEP.join(qids),
                ENTITY_SEP.join(coords),
                ENTITY_SEP.join(images_cell)
            ]
        )

        if SLEEP_BETWEEN_ROWS_SEC > 0:
            time.sleep(SLEEP_BETWEEN_ROWS_SEC)

    # Write output (Excel-friendly)
    with open(output_path, "w", newline="", encoding=OUTPUT_ENCODING) as out:
        writer = csv.writer(
            out,
            delimiter=OUTPUT_DELIMITER,
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writerows(out_rows)

    return output_path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for M2 entity enrichment."""

    SETTINGS.validate()
    args = parse_args(argv)
    if args.list_models:
        for model in available_m2_models():
            print(model)
        return 0

    configure_m2_model(args.model)
    input_csv_path = args.input_csv if args.input_csv else resolve_default_input_csv_path()
    out = enrich_csv_with_wikidata_qids(input_csv_path)
    print(f"Output CSV created -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
