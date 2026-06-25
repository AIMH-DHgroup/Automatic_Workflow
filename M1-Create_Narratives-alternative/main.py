from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from automatic_workflow.m1 import run_m1_module  # noqa: E402


def run(input_path: str | None = None) -> Path:
    """Run the alternative M1 JSON-events variant."""

    return run_m1_module(Path(__file__).resolve().parent, "json_events", input_path)


if __name__ == "__main__":
    result = run(sys.argv[1] if len(sys.argv) > 1 else None)
    if result:
        print(result)
