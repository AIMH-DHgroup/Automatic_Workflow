from __future__ import annotations

from pathlib import Path

import pytest

from automatic_workflow import pipeline


def test_copy_to_module_input_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pipeline.copy_to_module_input(tmp_path / "missing.txt", tmp_path / "input")


def test_run_pipeline_orchestrates_legacy_steps(
    monkeypatch: pytest.MonkeyPatch, project_layout: Path
) -> None:
    source = project_layout / "story.txt"
    source.write_text("A story", encoding="utf-8")
    calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def fake_run_step(script_dir: Path, script_name: str, *args: str) -> None:
        calls.append((script_dir, script_name, args))
        if script_dir.name == "M1-Create_Narratives-alternative-2":
            (script_dir / "output" / "story.csv").write_text(
                "title,description\nOne,Text\n", encoding="utf-8"
            )
        elif script_dir.name == "M2-Entities":
            (script_dir / "output" / "story_qids.csv").write_text(
                '"title";"description";"wikidata_names_en";"wikidata_qids";"wikidata_coords";"wikidata_images"\n',
                encoding="utf-8-sig",
            )
        elif script_dir.name == "M3-graph":
            (script_dir / "output" / "HTML_Storymap").mkdir(parents=True, exist_ok=True)
            (script_dir / "output" / "json").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipeline, "run_step", fake_run_step)

    outputs = pipeline.run_pipeline(source, root_dir=project_layout)

    assert len(calls) == 3
    assert outputs["narrative_csv"].name == "story.csv"
    assert outputs["qids_csv"].name == "story_qids.csv"
    assert outputs["storymap_dir"].exists()


def test_run_pipeline_first_line_title_prepares_body_and_passes_title(
    monkeypatch: pytest.MonkeyPatch, project_layout: Path
) -> None:
    source = project_layout / "story.txt"
    source.write_text("Narrative Title\nBody paragraph one.\nBody paragraph two.", encoding="utf-8")
    calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def fake_run_step(script_dir: Path, script_name: str, *args: str) -> None:
        calls.append((script_dir, script_name, args))
        if script_dir.name == "M1-Create_Narratives-alternative-2":
            prepared_input = Path(args[0])
            assert prepared_input.read_text(encoding="utf-8") == (
                "Body paragraph one.\nBody paragraph two.\n"
            )
            (script_dir / "output" / f"{prepared_input.stem}.csv").write_text(
                "title,description\nOne,Text\n", encoding="utf-8"
            )
        elif script_dir.name == "M2-Entities":
            step2_input = Path(args[0])
            (script_dir / "output" / f"{step2_input.stem}_qids.csv").write_text(
                '"title";"description";"wikidata_names_en";"wikidata_qids";"wikidata_coords";"wikidata_images"\n',
                encoding="utf-8-sig",
            )
        elif script_dir.name == "M3-graph":
            (script_dir / "output" / "HTML_Storymap").mkdir(parents=True, exist_ok=True)
            (script_dir / "output" / "json").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipeline, "run_step", fake_run_step)

    outputs = pipeline.run_pipeline(source, root_dir=project_layout, first_line_title=True)

    assert outputs["narrative_csv"].name == "story_body.csv"
    assert calls[2][2][-2:] == ("--title", "Narrative Title")


def test_run_pipeline_extracts_markdown_heading_title(
    monkeypatch: pytest.MonkeyPatch, project_layout: Path
) -> None:
    source = project_layout / "story.md"
    source.write_text("# Markdown Title\n\nBody paragraph.", encoding="utf-8")
    calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def fake_run_step(script_dir: Path, script_name: str, *args: str) -> None:
        calls.append((script_dir, script_name, args))
        if script_dir.name == "M1-Create_Narratives-alternative-2":
            prepared = Path(args[0])
            assert prepared.name == "story_body.md"
            assert prepared.read_text(encoding="utf-8") == "Body paragraph.\n"
            (script_dir / "output" / "story_body.csv").write_text(
                "title,description\nOne,Text\n", encoding="utf-8"
            )
        elif script_dir.name == "M2-Entities":
            (script_dir / "output" / "story_body_qids.csv").write_text(
                '"title";"description";"wikidata_names_en";"wikidata_qids";"wikidata_coords";"wikidata_images"\n',
                encoding="utf-8-sig",
            )
        elif script_dir.name == "M3-graph":
            (script_dir / "output" / "HTML_Storymap").mkdir(parents=True, exist_ok=True)
            (script_dir / "output" / "json").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipeline, "run_step", fake_run_step)
    outputs = pipeline.run_pipeline(source, root_dir=project_layout)

    assert outputs["narrative_csv"].name == "story_body.csv"
    assert calls[2][2][-2:] == ("--title", "Markdown Title")


def test_run_pipeline_explicit_title_does_not_strip_first_line(
    monkeypatch: pytest.MonkeyPatch, project_layout: Path
) -> None:
    source = project_layout / "story.txt"
    source.write_text("Looks Like A Title\nBut remains body text.", encoding="utf-8")
    calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def fake_run_step(script_dir: Path, script_name: str, *args: str) -> None:
        calls.append((script_dir, script_name, args))
        if script_dir.name == "M1-Create_Narratives-alternative-2":
            step1_input = Path(args[0])
            assert step1_input.read_text(encoding="utf-8") == (
                "Looks Like A Title\nBut remains body text."
            )
            (script_dir / "output" / "story.csv").write_text(
                "title,description\nOne,Text\n", encoding="utf-8"
            )
        elif script_dir.name == "M2-Entities":
            (script_dir / "output" / "story_qids.csv").write_text(
                '"title";"description";"wikidata_names_en";"wikidata_qids";"wikidata_coords";"wikidata_images"\n',
                encoding="utf-8-sig",
            )
        elif script_dir.name == "M3-graph":
            (script_dir / "output" / "HTML_Storymap").mkdir(parents=True, exist_ok=True)
            (script_dir / "output" / "json").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipeline, "run_step", fake_run_step)

    pipeline.run_pipeline(source, root_dir=project_layout, title="Explicit Title")

    assert calls[2][2][-2:] == ("--title", "Explicit Title")


def test_run_pipeline_explicit_title_overrides_extracted_title(
    monkeypatch: pytest.MonkeyPatch, project_layout: Path
) -> None:
    source = project_layout / "story.txt"
    source.write_text("Extracted Title\nBody text.", encoding="utf-8")
    calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def fake_run_step(script_dir: Path, script_name: str, *args: str) -> None:
        calls.append((script_dir, script_name, args))
        if script_dir.name == "M1-Create_Narratives-alternative-2":
            prepared_input = Path(args[0])
            assert prepared_input.read_text(encoding="utf-8") == "Body text.\n"
            (script_dir / "output" / f"{prepared_input.stem}.csv").write_text(
                "title,description\nOne,Text\n", encoding="utf-8"
            )
        elif script_dir.name == "M2-Entities":
            step2_input = Path(args[0])
            (script_dir / "output" / f"{step2_input.stem}_qids.csv").write_text(
                '"title";"description";"wikidata_names_en";"wikidata_qids";"wikidata_coords";"wikidata_images"\n',
                encoding="utf-8-sig",
            )
        elif script_dir.name == "M3-graph":
            (script_dir / "output" / "HTML_Storymap").mkdir(parents=True, exist_ok=True)
            (script_dir / "output" / "json").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipeline, "run_step", fake_run_step)

    pipeline.run_pipeline(
        source,
        root_dir=project_layout,
        first_line_title=True,
        title="Explicit Title",
    )

    assert calls[2][2][-2:] == ("--title", "Explicit Title")


def test_run_pipeline_can_select_alternative_entity_module_and_model(
    monkeypatch: pytest.MonkeyPatch, project_layout: Path
) -> None:
    alternative_m2 = project_layout / "M2-Entities-alternative"
    (alternative_m2 / "input").mkdir(parents=True)
    (alternative_m2 / "output").mkdir(parents=True)
    source = project_layout / "story.txt"
    source.write_text("Body text", encoding="utf-8")
    calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def fake_run_step(script_dir: Path, script_name: str, *args: str) -> None:
        calls.append((script_dir, script_name, args))
        if script_dir.name == "M1-Create_Narratives-alternative-2":
            (script_dir / "output" / "story.csv").write_text(
                "title,description\nOne,Text\n", encoding="utf-8"
            )
        elif script_dir.name == "M2-Entities-alternative":
            assert args[:2] == ("--model", "llama3.1:8b")
            (script_dir / "output" / "story_qids.csv").write_text(
                '"title";"description";"wikidata_names_en";"wikidata_qids";"wikidata_coords";"wikidata_images"\n',
                encoding="utf-8-sig",
            )
        elif script_dir.name == "M3-graph":
            (script_dir / "output" / "HTML_Storymap").mkdir(parents=True, exist_ok=True)
            (script_dir / "output" / "json").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipeline, "run_step", fake_run_step)
    outputs = pipeline.run_pipeline(
        source,
        root_dir=project_layout,
        entity_module="alternative",
        entity_model="llama3.1:8b",
        m1_mode="chunked",
    )

    assert calls[1][0] == alternative_m2
    assert calls[0][2][0] == "--chunked"
    assert outputs["qids_csv"].parent == alternative_m2 / "output"


def test_run_pipeline_can_select_relik_backend(
    monkeypatch: pytest.MonkeyPatch, project_layout: Path
) -> None:
    alternative_m2 = project_layout / "M2-Entities-alternative"
    (alternative_m2 / "input").mkdir(parents=True)
    (alternative_m2 / "output").mkdir(parents=True)
    source = project_layout / "story.txt"
    source.write_text("Body text", encoding="utf-8")
    calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def fake_run_step(script_dir: Path, script_name: str, *args: str) -> None:
        calls.append((script_dir, script_name, args))
        if script_dir.name == "M1-Create_Narratives-alternative-2":
            (script_dir / "output" / "story.csv").write_text(
                "title,description\nOne,Text\n", encoding="utf-8"
            )
        elif script_dir.name == "M2-Entities-alternative":
            assert args[:6] == (
                "--device",
                "cpu",
                "--relik-model",
                "local-model",
                "--backend",
                "relik",
            )
            (script_dir / "output" / "story_qids.csv").write_text(
                '"title";"description";"wikidata_names_en";"wikidata_qids";"wikidata_coords";"wikidata_images"\n',
                encoding="utf-8-sig",
            )
        elif script_dir.name == "M3-graph":
            (script_dir / "output" / "HTML_Storymap").mkdir(parents=True, exist_ok=True)
            (script_dir / "output" / "json").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipeline, "run_step", fake_run_step)
    pipeline.run_pipeline(
        source,
        root_dir=project_layout,
        entity_module="alternative",
        entity_backend="relik",
        relik_model="local-model",
        device="cpu",
    )


def test_run_pipeline_rejects_non_txt_input(project_layout: Path) -> None:
    source = project_layout / "story.csv"
    source.write_text("title,description\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a .txt or .md"):
        pipeline.run_pipeline(source, root_dir=project_layout)


def test_run_pipeline_can_select_narratia_m3(
    monkeypatch: pytest.MonkeyPatch, project_layout: Path
) -> None:
    source = project_layout / "story.txt"
    source.write_text("Body text", encoding="utf-8")
    calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def fake_run_step(script_dir: Path, script_name: str, *args: str) -> None:
        calls.append((script_dir, script_name, args))
        if script_dir.name == "M1-Create_Narratives-alternative-2":
            (script_dir / "output" / "story.csv").write_text(
                "title,description\nOne,Text\n", encoding="utf-8"
            )
        elif script_dir.name == "M2-Entities":
            (script_dir / "output" / "story_qids.csv").write_text(
                '"title";"description";"wikidata_names_en";"wikidata_qids";"wikidata_coords";"wikidata_images"\n',
                encoding="utf-8-sig",
            )
        elif script_dir.name == "M3-narratia":
            (script_dir / "output" / "HTML_Storymap").mkdir(parents=True, exist_ok=True)
            (script_dir / "output" / "json").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipeline, "run_step", fake_run_step)
    pipeline.run_pipeline(source, root_dir=project_layout, m3_module="narratia")

    assert calls[2][1] == "main.py"
