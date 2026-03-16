---
name: project-scaffold
description: >
  Bootstrap new biomedical ML research repositories with production-grade
  structure. Use when: (1) Creating a new research project from scratch,
  (2) Setting up pyproject.toml with uv, (3) Establishing directory layout,
  CLAUDE.md, or config patterns for a biomedical/healthcare ML project.
---

# Project Scaffold

## Quick Start

```bash
python scripts/init_project.py my-study --path /path/to/workspace --python-version 3.11
python scripts/init_project.py my-study --path /path/to/workspace --python-version 3.11 --docker --dependency monai
cd my-study && uv sync && uv run pytest tests/
```

Creates a uv-managed Python project with: `src/<pkg>/config.py` for
paths/settings, `tests/`, a dated `docs/<YYYYMMDD>/` directory, `CLAUDE.md`,
pre-commit hooks, ruff config, and standard biomedical ML dependencies.

## What Gets Created

```
my-study/
├── pyproject.toml          # uv/hatch config + dependencies + ruff config
├── .python-version         # Pinned Python version
├── .pre-commit-config.yaml # ruff linting + formatting
├── .env.example            # Template for secrets
├── .gitignore
├── CLAUDE.md               # Agent guidance
├── README.md
├── Dockerfile              # Optional, if --docker is passed
├── .dockerignore           # Optional, if --docker is passed
├── src/<pkg>/
│   ├── __init__.py
│   └── config.py           # Paths and project settings (env var overrides)
├── scripts/                # CLI entry points
├── tests/
│   └── test_config.py
└── docs/<YYYYMMDD>/        # Dated reports and figures
```

## Decision Points

**ASK the user** before scaffolding:
- What Python version (3.10, 3.11, 3.12)?
- Do they need Docker support for reproducible environments?
- Any additional dependencies beyond the defaults (numpy, pandas, scikit-learn, scipy, tqdm, matplotlib, wandb)?

## References

| File | Read When |
|------|-----------|
| [references/project-structure.md](references/project-structure.md) | Directory layout conventions, uv setup, config patterns, CLAUDE.md template, pre-commit/ruff setup, Docker support, env var management |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/init_project.py` | Scaffold a minimal new project (supports `--python-version`, `--docker`, and repeatable `--dependency` flags) |
