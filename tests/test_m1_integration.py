from __future__ import annotations

from pathlib import Path

import pytest

from automatic_workflow.config import WorkflowSettings
from automatic_workflow.m1 import (
    best_region_page,
    build_header_resolution,
    build_sentence,
    extract_images_from_region,
    extract_json_array,
    load_mapping,
    parse_bool,
    process_csv,
    process_txt,
    run_m1_module,
    serialize_story_from_json,
)


def test_mapping_and_sentence_helpers(tmp_path: Path) -> None:
    mapping = tmp_path / "mappingtable.csv"
    mapping.write_text(
        "db,event,prefix,suffix,is_title\n"
        "Country,Intro,Region,is active,true\n"
        "Population,Stats,Population,is people,false\n",
        encoding="utf-8",
    )

    rows, by_key, order = load_mapping(mapping)
    resolved = build_header_resolution(["Country", "Total population year"], rows, by_key)

    assert order == ["Intro", "Stats"]
    assert resolved["country"].event_label == "Intro"
    assert parse_bool("yes") is True
    assert build_sentence("There", "yes", "assets") == "There are assets."
    assert build_sentence("There", "no", "assets") == "There are not assets."


def test_process_csv_with_mocked_images(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    mapping = tmp_path / "mappingtable.csv"
    mapping.write_text(
        "db,event,prefix,suffix,is_title\n"
        "Country,Intro,Region,is active,true\n"
        "Asset,Intro,Asset,is available,false\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("Country,Asset\nItaly,Cheese\n", encoding="utf-8")
    monkeypatch.setattr("automatic_workflow.m1.get_images_for_country", lambda country, cache: [])

    output_dir = process_csv(dataset, mapping, tmp_path / "out")

    assert (output_dir / "1.csv").read_text(encoding="utf-8") == (
        "title,description,image\nRegion Italy is active.,Asset Cheese is available.,\n"
    )


def test_region_image_helpers_handle_success_and_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    assert "italy" in best_region_page("Italy")

    class Response:
        text = '<img src="https://example.org/a.jpg"><img src="https://example.org/logo.png">'

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr("automatic_workflow.m1.requests.get", lambda *args, **kwargs: Response())
    assert extract_images_from_region("https://example.org") == ["https://example.org/a.jpg"]

    def fail_get(*args, **kwargs):  # type: ignore[no-untyped-def]
        import requests

        raise requests.Timeout("timeout")

    monkeypatch.setattr("automatic_workflow.m1.requests.get", fail_get)
    assert extract_images_from_region("https://example.org") == []


def test_json_event_parsing_and_serialization() -> None:
    parsed = extract_json_array('[{"title":"One","event":"Text",}]')

    assert parsed == [{"title": "One", "event": "Text"}]
    assert serialize_story_from_json(parsed or []) == "title,description\nOne,Text\n"


def test_process_txt_json_events_falls_back_on_bad_json(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("automatic_workflow.m1.invoke_ollama", lambda *args, **kwargs: "not json")
    input_file = tmp_path / "story.txt"
    input_file.write_text("Original text", encoding="utf-8")

    output = process_txt(
        input_file,
        tmp_path / "out",
        "json_events",
        WorkflowSettings(root_dir=tmp_path),
    )

    assert output.read_text(encoding="utf-8") == "title,description\nevent-1,Original text\n"


def test_process_txt_with_mocked_ollama_titles(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    responses = iter(["First paragraph.\n\nSecond paragraph.", "First;Second"])

    def fake_invoke(system_prompt: str, prompt: str, settings: WorkflowSettings) -> str:
        return next(responses)

    monkeypatch.setattr("automatic_workflow.m1.invoke_ollama", fake_invoke)

    input_file = tmp_path / "story.txt"
    input_file.write_text("First paragraph. Second paragraph.", encoding="utf-8")

    output = process_txt(
        input_file,
        tmp_path / "output",
        "paragraphs_with_titles",
        WorkflowSettings(root_dir=tmp_path),
    )

    assert output.read_text(encoding="utf-8") == (
        "title,description\nFirst,First paragraph.\nSecond,Second paragraph.\n"
    )


def test_run_m1_module_rejects_unsupported_input(tmp_path: Path) -> None:
    base = tmp_path / "M1"
    (base / "input").mkdir(parents=True)
    (base / "output").mkdir()
    bad = base / "input" / "story.md"
    bad.write_text("bad", encoding="utf-8")

    with pytest.raises(ValueError, match=".csv or .txt"):
        run_m1_module(base, "paragraphs", str(bad))
