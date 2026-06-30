"""Configuration helpers for Automatic Workflow.

The legacy scripts historically embedded model names, API identity strings, and
Java paths directly in source code. This module centralizes those settings and
loads production overrides from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_OLLAMA_MODEL = "gemma3:12b"
DEFAULT_USER_AGENT = "automatic-workflow/1.0 (contact: workflow@example.org)"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class WorkflowSettings:
    """Runtime settings shared by workflow modules.

    Attributes:
        root_dir: Repository or project root directory.
        ollama_model: Local Ollama model name used for LLM calls.
        wikidata_user_agent: Contact-compliant User-Agent for Wikimedia APIs.
        java_bin: Optional explicit Java executable path.
        require_owl: Whether Triplify/OWL failure should fail the stage.
        log_level: Standard Python logging level name.
    """

    root_dir: Path
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    wikidata_user_agent: str = DEFAULT_USER_AGENT
    java_bin: str | None = None
    require_owl: bool = False
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, root_dir: Path | None = None) -> WorkflowSettings:
        """Create settings from environment variables.

        Args:
            root_dir: Optional explicit project root. Defaults to the current
                working directory.

        Returns:
            Populated workflow settings.
        """

        return cls(
            root_dir=(root_dir or Path.cwd()).resolve(),
            ollama_model=os.getenv("AUTOMATIC_WORKFLOW_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            wikidata_user_agent=os.getenv(
                "AUTOMATIC_WORKFLOW_WIKIDATA_USER_AGENT", DEFAULT_USER_AGENT
            ),
            java_bin=os.getenv("AUTOMATIC_WORKFLOW_JAVA_BIN") or None,
            require_owl=_env_bool("AUTOMATIC_WORKFLOW_REQUIRE_OWL", False),
            log_level=os.getenv("AUTOMATIC_WORKFLOW_LOG_LEVEL", "INFO"),
        )

    def validate(self) -> None:
        """Validate settings that are unsafe or likely to break external APIs.

        Raises:
            ValueError: If a setting is present but invalid.
        """

        if not self.ollama_model.strip():
            raise ValueError("AUTOMATIC_WORKFLOW_OLLAMA_MODEL must not be empty")
        if "youremail@example.com" in self.wikidata_user_agent:
            raise ValueError(
                "AUTOMATIC_WORKFLOW_WIKIDATA_USER_AGENT must contain a real contact address"
            )
