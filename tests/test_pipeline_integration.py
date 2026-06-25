from __future__ import annotations

from pathlib import Path

from automatic_workflow.pipeline import run_pipeline


def write_script(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_full_pipeline_with_fixture_scripts(tmp_path: Path) -> None:
    m1 = tmp_path / "M1-Create_Narratives-alternative-2"
    m2 = tmp_path / "M2-Entities"
    m3 = tmp_path / "M3-graph"
    for module in [m1, m2, m3]:
        (module / "input").mkdir(parents=True)
        (module / "output").mkdir()

    write_script(
        m1 / "main.py",
        "from pathlib import Path\nimport sys\nout=Path('output')/(Path(sys.argv[1]).stem+'.csv')\nout.write_text('title,description\\nEvent,Text\\n', encoding='utf-8')\n",
    )
    write_script(
        m2 / "main.py",
        'from pathlib import Path\nimport sys\nout=Path(\'output\')/(Path(sys.argv[1]).stem+\'_qids.csv\')\nout.write_text(\'"title";"description";"wikidata_names_en";"wikidata_qids";"wikidata_coords";"wikidata_images"\\n\', encoding=\'utf-8-sig\')\n',
    )
    write_script(
        m3 / "main.py",
        "from pathlib import Path\nPath('output/HTML_Storymap').mkdir(parents=True, exist_ok=True)\nPath('output/json').mkdir(parents=True, exist_ok=True)\nPath('output/owl').mkdir(parents=True, exist_ok=True)\n",
    )
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("Fixture narrative", encoding="utf-8")

    outputs = run_pipeline(fixture, root_dir=tmp_path)

    assert outputs["narrative_csv"].exists()
    assert outputs["qids_csv"].exists()
    assert outputs["storymap_dir"].exists()
    assert outputs["triplify_json_dir"].exists()
