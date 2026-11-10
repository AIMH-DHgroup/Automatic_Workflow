from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_m2_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "M2-Entities" / "main.py"
    spec = importlib.util.spec_from_file_location("m2_main", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_m2_wikipedia_failure_outputs_empty_enrichment(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    m2 = load_m2_module()
    input_file = tmp_path / "story.csv"
    input_file.write_text("title,description\nEvent,Rome is important.\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(m2, "OUTPUT_DIR", str(output_dir))
    monkeypatch.setattr(
        m2,
        "llm_keywords_safe",
        lambda text: ([{"entity_in_the_text": "Rome", "wikipedia_title": "Rome"}], "{}"),
    )

    def fail_titles(titles: list[str]) -> dict[str, str | None]:
        raise RuntimeError("wikipedia unavailable")

    monkeypatch.setattr(m2, "fetch_wikidata_qids_from_titles", fail_titles)

    output_path = Path(m2.enrich_csv_with_wikidata_qids(str(input_file)))

    assert output_path.exists()
    assert '"Event";"Rome is important.";"";"";"";""' in output_path.read_text(encoding="utf-8-sig")


def test_m2_model_selection_validates_supported_models() -> None:
    m2 = load_m2_module()

    assert "gemma3:12b" in m2.available_m2_models()
    assert m2.configure_m2_model("llama3.1:8b") == "llama3.1:8b"

    assert m2.configure_m2_model("gemma4:12b") == "gemma4:12b"

    try:
        m2.configure_m2_model(" ")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected empty model to raise ValueError")


def test_m2_resolution_strategy_selects_wikidata_search(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    m2 = load_m2_module()
    m2.configure_resolution_strategy("wikidata")
    monkeypatch.setattr(m2, "_search_wikidata_qids", lambda *args: {"Rome": "Q220"})

    assert m2.resolve_entity_qids(["Rome"]) == {"Rome": "Q220"}


def test_m2_combined_resolution_uses_second_endpoint_only_for_missing_qids(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    m2 = load_m2_module()
    m2.configure_resolution_strategy("wikipedia_then_wikidata")
    monkeypatch.setattr(
        m2,
        "fetch_wikidata_qids_from_titles",
        lambda titles: {"Rome": "Q220", "Unknown": None},
    )
    requested: list[str] = []

    def fallback(titles: list[str], entity_terms=None) -> dict[str, str | None]:
        requested.extend(titles)
        return {"Unknown": "Q999"}

    monkeypatch.setattr(m2, "_search_wikidata_qids", fallback)

    assert m2.resolve_entity_qids(["Rome", "Unknown"]) == {"Rome": "Q220", "Unknown": "Q999"}
    assert requested == ["Unknown"]


def test_wikidata_search_tries_entity_text_before_wikipedia_title(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    m2 = load_m2_module()
    queries: list[str] = []

    class Response:
        def json(self) -> dict[str, list[dict[str, str]]]:
            return {"search": [{"id": "Q220"}]} if queries[-1] == "Rome" else {"search": []}

    def fake_get(session, url, params=None, headers=None, max_retries=6, base_sleep=1.0):
        queries.append(params["search"])
        return Response()

    monkeypatch.setattr(m2, "_get_with_retries", fake_get)
    assert m2._search_wikidata_qids(["Roma"], {"Roma": "Rome"}) == {"Roma": "Q220"}
    assert queries == ["Rome"]


def test_entity_qid_cache_avoids_duplicate_api_calls(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    m2 = load_m2_module()
    m2.ENTITY_QID_CACHE.clear()
    m2.configure_resolution_strategy("wikidata")
    calls = 0

    def search(titles, entity_terms=None):
        nonlocal calls
        calls += 1
        return {title: "Q220" for title in titles}

    monkeypatch.setattr(m2, "_search_wikidata_qids", search)
    terms = {"Rome": "Rome"}
    assert m2.resolve_entity_qids(["Rome"], terms) == {"Rome": "Q220"}
    assert m2.resolve_entity_qids(["Rome"], terms) == {"Rome": "Q220"}
    assert calls == 1


def test_alternative_m2_wrappers_call_original_functions_once() -> None:
    module_path = Path(__file__).resolve().parents[1] / "M2-Entities-alternative" / "main.py"
    spec = importlib.util.spec_from_file_location("m2_alternative_main", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ORIGINAL_FETCH_QIDS = lambda titles: {title: "Q1" for title in titles}

    assert module.fetch_wikidata_qids_from_titles(["Rome"]) == {"Rome": "Q1"}
