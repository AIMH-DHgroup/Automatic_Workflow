from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


CURRENT_PATH = Path(__file__).resolve().parents[1] / "M3-narratia" / "main.py"
SPEC = importlib.util.spec_from_file_location("m3_standard_current", CURRENT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"Unable to load standard M3 module from {CURRENT_PATH}")
CURRENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURRENT)

BASE_DIR = Path(__file__).resolve().parent
CURRENT.BASE_DIR = BASE_DIR
CURRENT.INPUT_DIR = BASE_DIR / "input"
CURRENT.OUTPUT_DIR = BASE_DIR / "output"
CURRENT.STORYMAP_DIR = CURRENT.OUTPUT_DIR / "HTML_Storymap"
CURRENT.JSON_DIR = CURRENT.OUTPUT_DIR / "json"
CURRENT.OWL_DIR = CURRENT.OUTPUT_DIR / "owl"
CURRENT.TRIPLIFY_JAR = BASE_DIR / "Triplify" / "Triplify.jar"
CURRENT.LOG_PATH = BASE_DIR / "scriptStories5OSM_py.log"


def main(csv_path: str | None = None, title: str | None = None) -> int:
    original_title = CURRENT.derive_narration_title
    if title:
        CURRENT.derive_narration_title = lambda path: title
    try:
        return CURRENT.main(csv_path, title=title)
    finally:
        CURRENT.derive_narration_title = original_title


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate standard StoryMap, JSON, and OWL assets")
    parser.add_argument("input_csv", nargs="?", help="Input *_qids.csv produced by M2")
    parser.add_argument("--title", help="Optional narrative title override")
    args = parser.parse_args()
    raise SystemExit(main(args.input_csv, args.title))
