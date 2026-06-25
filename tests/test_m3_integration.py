from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def load_m3_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "M3-graph" / "main.py"
    spec = importlib.util.spec_from_file_location("m3_integration_main", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qlever_failure_returns_empty_polygon(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    m3 = load_m3_module()
    monkeypatch.setattr(m3, "http_post_form_json", lambda url, payload: None)

    assert m3.fetch_qlever_polygon("Q123") == ""


def test_process_csv_runs_triplify_success(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    m3 = load_m3_module()
    input_file = tmp_path / "story_qids.csv"
    input_file.write_text(
        '"title";"description";"wikidata_names_en";"wikidata_qids";"wikidata_coords";"wikidata_images"\n'
        '"Event";"<b>Unsafe</b>";"Rome";"Q1";"41.9 12.5";"javascript:bad"\n',
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(m3, "STORYMAP_DIR", tmp_path / "storymap")
    monkeypatch.setattr(m3, "JSON_DIR", tmp_path / "json")
    monkeypatch.setattr(m3, "OWL_DIR", tmp_path / "owl")
    monkeypatch.setattr(m3, "LOG_PATH", tmp_path / "m3.log")
    monkeypatch.setattr(m3, "locate_java", lambda: "java")
    monkeypatch.setattr(m3, "fetch_qlever_polygon", lambda qid: "")
    monkeypatch.setattr(m3, "choose_slide_type_with_llm", lambda *args: "geographic")
    monkeypatch.setattr(
        m3,
        "fetch_wikidata_entity",
        lambda qid: {
            "results": {
                "bindings": [
                    {
                        "enName": {"value": "Rome"},
                        "enDesc": {"value": "City"},
                        "coordinates": {"value": "Point(12.5 41.9)"},
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(
        m3.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        ),
    )

    m3.process_csv(input_file, "1_story_qids", "story_qids")

    slide = (tmp_path / "storymap" / "N1_story_qids" / "slide.json").read_text(encoding="utf-8")
    assert "&lt;b&gt;Unsafe&lt;/b&gt;" in slide
    assert "javascript:bad" not in slide
    assert '"slideType": "geographic"' in slide


def test_process_csv_raises_when_required_triplify_fails(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    m3 = load_m3_module()
    input_file = tmp_path / "story_qids.csv"
    input_file.write_text(
        '"title";"description";"wikidata_names_en";"wikidata_qids";"wikidata_coords";"wikidata_images"\n'
        '"Event";"Description";"Rome";"Q1";"41.9 12.5";"https://example.org/a.jpg"\n',
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(m3, "STORYMAP_DIR", tmp_path / "storymap")
    monkeypatch.setattr(m3, "JSON_DIR", tmp_path / "json")
    monkeypatch.setattr(m3, "OWL_DIR", tmp_path / "owl")
    monkeypatch.setattr(m3, "LOG_PATH", tmp_path / "m3.log")
    monkeypatch.setattr(m3, "SETTINGS", SimpleNamespace(require_owl=True, java_bin=None))
    monkeypatch.setattr(m3, "locate_java", lambda: "java")
    monkeypatch.setattr(m3, "fetch_qlever_polygon", lambda qid: "")
    monkeypatch.setattr(m3, "choose_slide_type_with_llm", lambda *args: "normal")
    monkeypatch.setattr(m3, "fetch_wikidata_entity", lambda qid: {"results": {"bindings": []}})
    monkeypatch.setattr(
        m3.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args, returncode=1, stdout="", stderr="boom"
        ),
    )

    with pytest.raises(RuntimeError, match="Triplify failed"):
        m3.process_csv(input_file, "1_story_qids", "story_qids")


def test_slide_type_classifier_uses_llm_response(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    m3 = load_m3_module()

    class FakeOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, prompt: str) -> str:
            return "normal"

    monkeypatch.setattr(m3, "Ollama", FakeOllama)

    assert m3.choose_slide_type_with_llm("Conclusion", "Summary", [], True) == "normal"


def test_slide_type_classifier_falls_back_without_coordinates(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    m3 = load_m3_module()

    class FakeOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, prompt: str) -> str:
            return "geographic"

    monkeypatch.setattr(m3, "Ollama", FakeOllama)

    assert (
        m3.choose_slide_type_with_llm("Landscape", "A mountain area", ["Alps"], False) == "normal"
    )


def test_slide_type_classifier_falls_back_on_exception(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    m3 = load_m3_module()
    monkeypatch.setattr(m3, "LOG_PATH", Path("NUL"))

    class FailingOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def invoke(self, prompt: str) -> str:
            raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(m3, "Ollama", FailingOllama)

    assert m3.choose_slide_type_with_llm("Place", "A mapped place", ["Rome"], True) == "geographic"
