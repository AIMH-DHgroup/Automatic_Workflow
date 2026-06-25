from __future__ import annotations

from pathlib import Path

from automatic_workflow.triplify_audit import audit_triplify_dependencies, parse_jar_name


def test_parse_jar_name_extracts_artifact_and_version() -> None:
    assert parse_jar_name(Path("commons-httpclient-3.1.jar")) == ("commons-httpclient", "3.1")


def test_audit_triplify_dependencies_flags_known_risks(tmp_path: Path) -> None:
    lib = tmp_path / "Triplify_lib"
    lib.mkdir()
    (lib / "commons-httpclient-3.1.jar").write_text("", encoding="utf-8")
    (lib / "safe-lib-1.0.jar").write_text("", encoding="utf-8")

    findings = audit_triplify_dependencies(tmp_path)

    assert len(findings) == 1
    assert findings[0].artifact == "commons-httpclient"
