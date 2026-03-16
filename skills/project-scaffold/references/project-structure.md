# Project Structure Reference

## Contents

- Directory Layout
- Package Management
- Configuration Pattern
- CLAUDE.md Template
- Documentation Convention
- Pre-Commit Configuration
- Docker Support
- Environment Variable Management

Minimal directory layout for biomedical ML research projects.

## Directory Layout

```
project-name/
├── pyproject.toml          # uv/hatch project config + dependencies
├── .python-version         # Pin Python version
├── .env.example            # Template for secrets
├── .gitignore
├── CLAUDE.md               # Coding agent instructions
├── README.md
├── Dockerfile              # Optional, if container support is requested
├── .dockerignore           # Optional, if container support is requested
├── src/<package>/
│   ├── __init__.py
│   └── config.py           # Paths and project settings
├── scripts/                # CLI entry points (project-specific)
├── tests/
│   └── test_config.py
└── docs/<YYYYMMDD>/        # Dated reports and figures
```

Add modules as your project requires them - don't pre-create files you don't need yet.

## Package Management

Use `uv` (not pip or conda) for dependency management:

```bash
uv sync                     # Install from lockfile
uv add <package>            # Add dependency
uv run pytest tests/        # Run in project venv
```

## Configuration Pattern

Use dataclasses for typed, composable configs:

```python
from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    # Add project-specific fields as needed

    @property
    def experiment_id(self) -> str:
        return f"{self.name}__s{self.seed}"
```

## CLAUDE.md Template

Every project should have a CLAUDE.md with:
1. **Project Overview** - 1-2 sentence description
2. **Commands** - Install, test, and project-specific commands
3. **Architecture** - Module descriptions with file paths
4. **Key Design Decisions** - Non-obvious choices
5. **Documentation Workflow** - Where docs go
6. **Commit Guidelines** - Conventional Commits, PR format

## Documentation Convention

Date-stamped folders prevent overwriting previous results:

```
docs/
├── 20260215/
│   ├── analysis_report.md
│   └── fig_results.png
└── 20260220/
    ├── follow_up.md
    └── fig_comparison.png
```

## Pre-Commit Configuration

Set up code quality tooling with ruff (linting + formatting):

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

Add ruff configuration to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
ignore = ["E501"]  # line length handled by formatter
```

Install:

```bash
uv add --group dev pre-commit ruff
uv run pre-commit install
```

The scaffold script writes this ruff configuration directly, so the project is lint-ready after `uv sync`.

## Docker Support

For reproducible environments (common in clinical ML):

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY scripts/ scripts/
RUN uv sync --no-dev --frozen
ENTRYPOINT ["uv", "run"]
```

If the user asks for containerization up front, scaffold `Dockerfile` and `.dockerignore` immediately; otherwise keep the project lean and add them later only when needed.

## Environment Variable Management

Use `python-dotenv` for local development:

```python
from dotenv import load_dotenv
load_dotenv()  # loads from .env file
```

The `.env.example` file documents all expected variables. Never commit `.env` itself.
