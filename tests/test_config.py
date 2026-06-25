from __future__ import annotations

from pathlib import Path

import pytest

from automatic_workflow.config import WorkflowSettings


def test_settings_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AUTOMATIC_WORKFLOW_OLLAMA_MODEL", "test-model")
    monkeypatch.setenv("AUTOMATIC_WORKFLOW_REQUIRE_OWL", "true")

    settings = WorkflowSettings.from_env(tmp_path)

    assert settings.root_dir == tmp_path.resolve()
    assert settings.ollama_model == "test-model"
    assert settings.require_owl is True


def test_settings_validate_rejects_placeholder_contact(tmp_path: Path) -> None:
    settings = WorkflowSettings(
        root_dir=tmp_path,
        wikidata_user_agent="automatic-workflow (contact: youremail@example.com)",
    )

    with pytest.raises(ValueError, match="real contact"):
        settings.validate()
