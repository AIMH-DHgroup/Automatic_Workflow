from __future__ import annotations

import sys
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from automatic_workflow.m1 import run_m1_module  # noqa: E402


def run(input_path: str | None = None, chunked: bool = False, pointer_titles: bool = False) -> Path:
    """Run M1 with generated titles and optional chunked or pointer splitting."""

    mode = "pointer_titles" if pointer_titles else "chunked" if chunked else "paragraphs_with_titles"
    return run_m1_module(Path(__file__).resolve().parent, mode, input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create narrative CSV from text or Markdown")
    parser.add_argument("input_path", nargs="?")
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="Split the text into 500-word overlapping chunks before paragraph splitting",
    )
    parser.add_argument(
        "--pointer-titles",
        action="store_true",
        help="Let the LLM mark paragraph boundaries, then slice the original text deterministically",
    )
    args = parser.parse_args()
    result = run(args.input_path, args.chunked, args.pointer_titles)
    if result:
        print(result)
