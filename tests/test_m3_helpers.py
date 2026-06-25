from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_m3_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "M3-graph" / "main.py"
    spec = importlib.util.spec_from_file_location("m3_main", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_entity_coords_returns_lon_lat_order() -> None:
    m3 = load_m3_module()
    assert m3.parse_entity_coords("45.1 9.2,null") == ("9.2", "45.1")


def test_first_image_from_cell_uses_first_non_null_image() -> None:
    m3 = load_m3_module()
    assert m3.first_image_from_cell("null,https://example.org/a.jpg|https://example.org/b.jpg") == (
        "https://example.org/a.jpg"
    )
