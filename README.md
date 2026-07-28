# Automatic Workflow

Automatic Workflow transforms narrative text or structured CSV data into:

- narrative CSV files
- Wikidata-enriched CSV files with entity labels, QIDs, coordinates, and images
- StoryMap HTML assets
- Triplify JSON
- OWL files generated through Java Triplify

The repository still supports the original module scripts, while shared code and new tooling live under `src/automatic_workflow`.

## Repository Layout

```text
src/automatic_workflow/        Reusable package code
tests/                         Pytest suite
docs/                          Project documentation
M1-Create_Narratives*/         Legacy narrative creation modules
M2-Entities/                   Legacy entity enrichment module
M3-graph/                      Legacy storymap/Triplify module
run_pipeline.py                Backward-compatible pipeline wrapper
```

## Requirements

- Python 3.11+
- Ollama installed and running for `.txt` processing and entity extraction
- Ollama model configured by `AUTOMATIC_WORKFLOW_OLLAMA_MODEL` defaulting to `gemma3:12b`
- Java installed for OWL generation through `M3-graph/Triplify/Triplify.jar`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

Copy `.env.example` to `.env` for local notes, or export the variables in your shell.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `AUTOMATIC_WORKFLOW_OLLAMA_MODEL` | Ollama model for M1/M2 LLM calls | `gemma3:12b` |
| `AUTOMATIC_WORKFLOW_M2_OLLAMA_MODEL` | Optional M2-specific Ollama model from the supported list | unset |
| `AUTOMATIC_WORKFLOW_M3_OLLAMA_MODEL` | Optional M3-specific Ollama model for geographic vs normal slide classification | unset |
| `AUTOMATIC_WORKFLOW_COVER_IMAGE` | StoryMap narrative cover image URL | `https://tool.dlnarratives.eu/images/title-background.jpg` |
| `AUTOMATIC_WORKFLOW_WIKIDATA_USER_AGENT` | Wikimedia API User-Agent with contact info | `automatic-workflow/1.0 (contact: workflow@example.org)` |
| `AUTOMATIC_WORKFLOW_JAVA_BIN` | Optional explicit Java executable | auto-detect |
| `AUTOMATIC_WORKFLOW_REQUIRE_OWL` | Fail M3 if Java/Triplify cannot produce OWL | `0` |
| `AUTOMATIC_WORKFLOW_LOG_LEVEL` | Logging level for package CLI | `INFO` |

Before production use, set `AUTOMATIC_WORKFLOW_WIKIDATA_USER_AGENT` to a real project/contact address.

## Usage

Run the full workflow with the backward-compatible script:

```bash
python run_pipeline.py "path\to\input.txt"
```

If the first line of the TXT is the narrative title, enable first-line title mode:

```bash
python run_pipeline.py --first-line-title "path\to\input.txt"
```

In this mode, the first line is used as the generated narrative title and only the remaining text is sent through M1, M2, and M3.

You can also provide the title explicitly:

```bash
python run_pipeline.py --title "Your Title" "path\to\input.txt"
```

If both `--title` and `--first-line-title` are provided, the first line is still removed from the text sent through the workflow, but `--title` takes precedence as the generated narrative title.

To use the alternative map-focused entity extractor, which prioritizes
geographic entities and adds images from Wikipedia pages, run:

```bash
python run_pipeline.py --entity-module alternative --entity-model "llama3.1:8b" "path\to\input.txt"
```

The original M2 extractor remains the default. The alternative extractor
returns the same `*_qids.csv` format, so M3 can consume it without changes.

Or use the installed console command:

```bash
automatic-workflow "path\to\input.txt"
```

Legacy module commands remain supported:

```bash
cd M1-Create_Narratives-alternative-2
python main.py "path\to\input.txt"

cd ..\M2-Entities
python main.py "path\to\narrative.csv"

# Show supported M2 models
python main.py --list-models

# Choose an M2 model explicitly
python main.py --model "llama3.1:8b" "path\to\narrative.csv"

cd ..\M3-graph
python main.py "path\to\narrative_qids.csv"
```

M3 asks Ollama whether each generated slide should be `geographic` or `normal`. If Ollama is unavailable or returns an invalid answer, M3 falls back to coordinate-based behavior.

## Workflow

1. M1 reads a `.txt` input and creates `M1-Create_Narratives-alternative-2/output/<name>.csv`.
2. The pipeline copies that CSV into `M2-Entities/input/`.
3. M2 creates `M2-Entities/output/<name>_qids.csv`.
4. The pipeline copies that CSV into `M3-graph/input/`.
5. M3 creates StoryMap, JSON, and optional OWL artifacts under `M3-graph/output/`.

## Development Checks

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest --cov
```

Audit bundled Triplify jars with offline risk hints:

```bash
python scripts/audit_triplify_deps.py
```

## Security Notes

- Do not commit `.env`, generated outputs, logs, archives, or virtual environments.
- `M3-graph/Triplify/config.properties` no longer ships default database credentials. Use a private local config if Virtuoso storage is required.
- Treat generated HTML as untrusted output if input narratives come from untrusted users.
- The bundled Java Triplify dependencies should be audited before production deployment.

## Documentation

Additional documentation is in `docs/` and can be served with:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```
