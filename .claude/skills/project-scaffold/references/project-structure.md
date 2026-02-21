# Project Structure Reference

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
