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
python scripts/init_project.py my-study --path /path/to/workspace
cd my-study && uv sync && uv run pytest tests/
```

Creates a uv-managed Python project with: `src/<pkg>/config.py` for
paths/settings, `tests/`, `docs/`, `CLAUDE.md`, and wandb in dependencies.

## References

| File | Apply When |
|------|------------|
| [references/project-structure.md](references/project-structure.md) | Directory layout conventions, uv setup, config patterns, CLAUDE.md template |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/init_project.py` | Scaffold a minimal new project |
