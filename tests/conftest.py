from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def project_layout(tmp_path: Path) -> Path:
    """Create the minimal legacy folder layout needed by pipeline tests."""

    for relative in [
        "M1-Create_Narratives-alternative-2/input",
        "M1-Create_Narratives-alternative-2/output",
        "M2-Entities/input",
        "M2-Entities/output",
        "M3-graph/input",
        "M3-graph/output",
    ]:
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    return tmp_path
