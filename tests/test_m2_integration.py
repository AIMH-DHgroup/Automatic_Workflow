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

    assert "gemma2:9b-instruct-q8_0" in m2.available_m2_models()
    assert m2.configure_m2_model("llama3.1:8b") == "llama3.1:8b"

    try:
        m2.configure_m2_model("not-a-model")
    except ValueError as exc:
        assert "Unsupported M2 model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected unsupported model to raise ValueError")
