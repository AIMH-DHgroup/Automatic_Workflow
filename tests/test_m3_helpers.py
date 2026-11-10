from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_m3_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "M3-narratia" / "main.py"
    spec = importlib.util.spec_from_file_location("m3_main", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_entity_coords_returns_lon_lat_order() -> None:
    m3 = load_m3_module()
    assert m3.parse_entity_coords("45.1 9.2,null") == ("9.2", "45.1")
    assert m3.parse_wkt_point("Point(-2.81053 39.88591)") == ("-2.81053", "39.88591")


def test_first_image_from_cell_uses_first_non_null_image() -> None:
    m3 = load_m3_module()
    assert m3.first_image_from_cell("null,https://example.org/a.jpg|https://example.org/b.jpg") == (
        "https://example.org/a.jpg"
    )


def test_qlever_polygon_cache_avoids_duplicate_requests(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    m3 = load_m3_module()
    m3._GEOMETRY_CACHE.clear()
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"results": {"bindings": [{"wkt": {"value": "POLYGON ((1 1, 2 2, 1 1))"}}]}}

    monkeypatch.setattr(m3, "http_post_form_json", fake_post)
    assert m3.fetch_qlever_polygon("Q1") == "POLYGON ((1 1, 2 2, 1 1))"
    assert m3.fetch_qlever_polygon("Q1") == "POLYGON ((1 1, 2 2, 1 1))"
    assert calls == 1


def test_empty_entity_json_is_not_considered_complete(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    m3 = load_m3_module()
    monkeypatch.setattr(m3, "STORYMAP_DIR", tmp_path / "story")
    monkeypatch.setattr(m3, "JSON_DIR", tmp_path / "json")
    monkeypatch.setattr(m3, "OWL_DIR", tmp_path / "owl")
    m3.STORYMAP_DIR.mkdir()
    m3.JSON_DIR.mkdir()
    m3.OWL_DIR.mkdir()
    (m3.STORYMAP_DIR / "N1_story").mkdir()
    (m3.JSON_DIR / "moving.52_N1_story.json").write_text('{"entities": {}, "events": {}}')
    (m3.OWL_DIR / "moving.52_N1_story.owl").write_text("")

    assert m3.has_existing_outputs("story") is False


def test_extended_wikidata_fields_extract_period_and_specific_class() -> None:
    m3 = load_m3_module()
    entity = {"_id": "Q1", "type": ["Q839954", "Q1640824"]}
    bindings = [
        {
            "historicalPeriod": {"value": "http://www.wikidata.org/entity/Q1985727"},
            "historicalPeriodLabel": {"value": "Roman period"},
            "constructionStart": {"value": "-0027-01-01T00:00:00Z"},
            "specificType": {"value": "http://www.wikidata.org/entity/Q1640824"},
        }
    ]

    m3.extract_temporal_fields(entity, bindings)

    assert entity["historicalPeriod"]["label"] == "Roman period"
    assert entity["timeSpan"]["start"] == "-0027-01-01T00:00:00Z"
    assert m3.classify_entity(entity, entity["type"]) == "epigraph"
    assert m3.classify_entity({}, ["Q27096213"]) == "place"
    assert m3.classify_entity({}, ["Q839954"]) == "archaeological site"


def test_entity_type_query_is_cached(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    m3 = load_m3_module()
    m3._TYPE_CACHE.clear()
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"results": {"bindings": [{"type": {"value": "http://www.wikidata.org/entity/Q27096213"}}]}}

    monkeypatch.setattr(m3, "http_post_form_json", fake_post)
    assert m3.fetch_entity_type_qids("Q1759993") == ["Q27096213"]
    assert m3.fetch_entity_type_qids("Q1759993") == ["Q27096213"]
    assert calls == 1
