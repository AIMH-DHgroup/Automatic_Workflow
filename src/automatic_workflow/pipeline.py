"""End-to-end orchestration for the legacy three-module workflow."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from automatic_workflow.config import WorkflowSettings
from automatic_workflow.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when a workflow stage fails or expected output is missing."""


def ensure_module_dirs(paths: list[Path]) -> None:
    """Create module input/output directories."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def copy_to_module_input(source: Path, destination_dir: Path) -> Path:
    """Copy a file into a module input directory and return the destination."""

    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / source.name
    if source.resolve() != destination_path.resolve():
        shutil.copy2(source, destination_path)
    return destination_path


def resolve_default_m1_input(m1_input_dir: Path, legacy_default: Path) -> Path:
    """Resolve the default M1 text input from `input/` or legacy sample file."""

    candidates = sorted(
        path for path in m1_input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    )
    if candidates:
        return candidates[0]
    if legacy_default.exists():
        return copy_to_module_input(legacy_default, m1_input_dir)
    raise FileNotFoundError(f"No .txt input files found in {m1_input_dir}")


def prepare_first_line_title_input(source: Path, destination_dir: Path) -> tuple[Path, str]:
    """Create an M1 input file without the first line and return its title.

    Args:
        source: Original text file whose first line is the narrative title.
        destination_dir: M1 input directory where the prepared body file is written.

    Returns:
        Tuple of prepared text path and extracted title.

    Raises:
        ValueError: If the file has no non-empty first line or no remaining body text.
    """

    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or not lines[0].strip():
        raise ValueError("First-line title mode requires a non-empty first line")
    title = lines[0].strip()
    body = "\n".join(lines[1:]).strip()
    if not body:
        raise ValueError("First-line title mode requires text after the first line")

    destination_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = destination_dir / f"{source.stem}_body{source.suffix}"
    prepared_path.write_text(body + "\n", encoding="utf-8")
    return prepared_path, title


def run_step(script_dir: Path, script_name: str, *args: str) -> None:
    """Run a legacy module script and raise on failure."""

    cmd = [sys.executable, script_name, *args]
    LOGGER.info("Running step", extra={"cmd": cmd, "cwd": str(script_dir)})
    env = os.environ.copy()
    # The legacy scripts are process boundaries. Avoid inheriting pytest-cov
    # subprocess settings, which can create incompatible coverage data files on
    # Windows when temporary fixture scripts are executed in integration tests.
    env.pop("COVERAGE_PROCESS_START", None)
    env.pop("COVERAGE_FILE", None)
    for key in list(env):
        if key.startswith("COV_CORE_"):
            env.pop(key, None)
    completed = subprocess.run(cmd, cwd=script_dir, check=False, env=env)
    if completed.returncode != 0:
        raise PipelineError(f"Step failed with exit code {completed.returncode}: {' '.join(cmd)}")


def run_pipeline(
    input_txt: Path | None = None,
    root_dir: Path | None = None,
    first_line_title: bool = False,
) -> dict[str, Path]:
    """Run the full M1 -> M2 -> M3 workflow.

    Args:
        input_txt: Optional explicit `.txt` input file.
        root_dir: Optional project root. Defaults to the package settings root.
        first_line_title: Whether to use the first input line as narrative title
            and send only the remaining text through M1/M2/M3.

    Returns:
        Paths to the main generated artifacts/directories.
    """

    settings = WorkflowSettings.from_env(root_dir=root_dir)
    settings.validate()
    root = settings.root_dir
    m1_dir = root / "M1-Create_Narratives-alternative-2"
    m2_dir = root / "M2-Entities"
    m3_dir = root / "M3-graph"

    m1_input_dir = m1_dir / "input"
    m1_output_dir = m1_dir / "output"
    m2_input_dir = m2_dir / "input"
    m2_output_dir = m2_dir / "output"
    m3_input_dir = m3_dir / "input"
    m3_output_dir = m3_dir / "output"
    ensure_module_dirs(
        [m1_input_dir, m1_output_dir, m2_input_dir, m2_output_dir, m3_input_dir, m3_output_dir]
    )

    source_input = (
        input_txt.resolve()
        if input_txt
        else resolve_default_m1_input(m1_input_dir, m1_dir / "dante.txt")
    )
    if source_input.suffix.lower() != ".txt":
        raise ValueError(f"Pipeline input must be a .txt file: {source_input}")

    narrative_title: str | None = None
    if first_line_title:
        step1_input, narrative_title = prepare_first_line_title_input(source_input, m1_input_dir)
    else:
        step1_input = copy_to_module_input(source_input, m1_input_dir)
    step1_output = m1_output_dir / f"{step1_input.stem}.csv"
    step2_input = m2_input_dir / step1_output.name
    step2_output = m2_output_dir / f"{step1_output.stem}_qids{step1_output.suffix}"
    step3_input = m3_input_dir / step2_output.name

    run_step(m1_dir, "main.py", str(step1_input))
    if not step1_output.exists():
        raise PipelineError(f"Step 1 output not found: {step1_output}")
    copy_to_module_input(step1_output, m2_input_dir)

    run_step(m2_dir, "main.py", str(step2_input))
    if not step2_output.exists():
        raise PipelineError(f"Step 2 output not found: {step2_output}")
    copy_to_module_input(step2_output, m3_input_dir)

    if narrative_title:
        run_step(m3_dir, "main.py", str(step3_input), "--title", narrative_title)
    else:
        run_step(m3_dir, "main.py", str(step3_input))
    storymap_dir = m3_output_dir / "HTML_Storymap"
    json_dir = m3_output_dir / "json"
    if not storymap_dir.exists() or not json_dir.exists():
        raise PipelineError("Step 3 did not create expected storymap/json output directories")

    return {
        "narrative_csv": step1_output,
        "qids_csv": step2_output,
        "storymap_dir": storymap_dir,
        "triplify_json_dir": json_dir,
        "owl_dir": m3_output_dir / "owl",
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the full workflow."""

    parser = argparse.ArgumentParser(description="Run the Automatic Workflow pipeline")
    parser.add_argument("input", nargs="?", type=Path, help="Optional input .txt file")
    parser.add_argument(
        "--first-line-title",
        action="store_true",
        help="Use the first line of the input TXT as narrative title and process only the remaining text",
    )
    args = parser.parse_args(argv)

    settings = WorkflowSettings.from_env(Path(__file__).resolve().parents[2])
    configure_logging(settings.log_level)
    try:
        outputs = run_pipeline(
            args.input,
            root_dir=settings.root_dir,
            first_line_title=args.first_line_title,
        )
    except Exception as exc:
        LOGGER.error("Pipeline failed: %s", exc)
        return 1

    LOGGER.info("Pipeline completed successfully")
    for label, path in outputs.items():
        LOGGER.info("%s: %s", label, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
