#!/usr/bin/env python3
"""Bootstrap a new biomedical ML research project.

Usage:
    python init_project.py <project-name> [--path <parent-dir>]

Creates a minimal uv-managed Python project skeleton. Add domain-specific
modules (training, evaluation, GPU utils, etc.) as your project requires them.
"""

import argparse
import os
import sys
import textwrap
from datetime import date


DEFAULT_DEPENDENCIES = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "scipy>=1.10",
    "tqdm>=4.60",
    "matplotlib>=3.7",
    "python-dotenv",
    "wandb>=0.15",
]

DEFAULT_DEV_DEPENDENCIES = [
    "pytest>=7.0",
    "pytest-cov",
    "pre-commit",
    "ruff>=0.8",
]

RUFF_TARGETS = {
    "3.10": "py310",
    "3.11": "py311",
    "3.12": "py312",
}


def write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(content).lstrip("\n"))


def format_toml_list(items: list[str], indent: int = 12) -> str:
    padding = " " * indent
    return "\n".join(f'{padding}"{item}",' for item in items)


def init_project(
    name: str,
    parent: str,
    python_version: str = "3.10",
    docker: bool = False,
    extra_dependencies: list[str] | None = None,
) -> str:
    root = os.path.join(parent, name)
    if os.path.exists(root):
        print(f"Error: {root} already exists", file=sys.stderr)
        sys.exit(1)

    pkg = name.replace("-", "_")
    docs_dir = date.today().strftime("%Y%m%d")
    dependencies = DEFAULT_DEPENDENCIES + list(extra_dependencies or [])
    dev_dependencies = DEFAULT_DEV_DEPENDENCIES
    docker_files = ""

    if docker:
        docker_files = textwrap.dedent(f"""\

            ## Docker

            ```bash
            docker build -t {name} .
            docker run --rm {name} pytest tests/
            ```
        """)

    # ── pyproject.toml ──────────────────────────────────────────────
    write(f"{root}/pyproject.toml", f"""\
        [project]
        name = "{name}"
        version = "0.1.0"
        requires-python = ">={python_version}"
        dependencies = [
{format_toml_list(dependencies)}
        ]

        [dependency-groups]
        dev = [
{format_toml_list(dev_dependencies)}
        ]

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["src/{pkg}"]

        [tool.pytest.ini_options]
        testpaths = ["tests"]

        [tool.ruff]
        line-length = 100
        target-version = "{RUFF_TARGETS[python_version]}"

        [tool.ruff.lint]
        select = ["E", "F", "I", "UP", "B"]
        ignore = ["E501"]
    """)

    # ── .python-version ─────────────────────────────────────────────
    write(f"{root}/.python-version", f"{python_version}\n")

    # ── .gitignore ──────────────────────────────────────────────────
    write(f"{root}/.gitignore", """\
        __pycache__/
        *.pyc
        .env
        .venv/
        *.egg-info/
        dist/
        build/
        wandb/
        .DS_Store
    """)

    # ── .env.example ────────────────────────────────────────────────
    write(f"{root}/.env.example", """\
        # Copy to .env and fill in values
        # WANDB_API_KEY=xxx
    """)

    # ── .pre-commit-config.yaml ──────────────────────────────────
    write(f"{root}/.pre-commit-config.yaml", """\
        repos:
          - repo: https://github.com/astral-sh/ruff-pre-commit
            rev: v0.8.0
            hooks:
              - id: ruff
                args: [--fix]
              - id: ruff-format
    """)

    # ── src/<pkg>/__init__.py ───────────────────────────────────────
    write(f"{root}/src/{pkg}/__init__.py", "")

    # ── src/<pkg>/config.py ─────────────────────────────────────────
    write(f"{root}/src/{pkg}/config.py", f"""\
        \"\"\"Project-wide paths and settings.

        Edit these to match your data layout. All other modules import from here.
        \"\"\"

        from __future__ import annotations

        import os


        # ── Paths (override via environment variables) ──────────────
        DATA_ROOT = os.environ.get("{pkg.upper()}_DATA_ROOT", "/path/to/data")
        OUTPUT_DIR = os.environ.get("{pkg.upper()}_OUTPUT_DIR", "/path/to/output")
    """)

    # ── scripts/ (empty, project-specific) ──────────────────────────
    os.makedirs(f"{root}/scripts", exist_ok=True)
    write(f"{root}/scripts/.gitkeep", "")

    # ── tests/ ──────────────────────────────────────────────────────
    write(f"{root}/tests/__init__.py", "")

    write(f"{root}/tests/test_config.py", f"""\
        \"\"\"Tests for configuration module.\"\"\"

        from {pkg}.config import DATA_ROOT, OUTPUT_DIR


        def test_config_paths_exist():
            assert isinstance(DATA_ROOT, str)
            assert isinstance(OUTPUT_DIR, str)
    """)

    # ── docs/ ───────────────────────────────────────────────────────
    write(f"{root}/docs/{docs_dir}/.gitkeep", "")

    # ── CLAUDE.md ───────────────────────────────────────────────────
    write(f"{root}/CLAUDE.md", f"""\
        # CLAUDE.md

        This file provides guidance to the coding agent when working with this repository.

        ## Project Overview

        <!-- TODO: Describe your project in 1-2 sentences -->

        ## Commands

        ```bash
        # Install dependencies
        uv sync

        # Run tests
        uv run pytest tests/

        # Run a single test
        uv run pytest tests/test_config.py
        ```

        ## Architecture

        **`src/{pkg}/config.py`** -- Paths and project settings.

        ## Key Design Decisions

        <!-- TODO: Document non-obvious choices -->

        ## Documentation Workflow

        - New docs live in `docs/<YYYYMMDD>/`; create a fresh date folder before adding files.

        ## Commit & PR Guidelines

        - Imperative commit subjects; `type: summary` (Conventional Commit style).
        - PRs: concise summary, test evidence, path/data assumptions.
    """)

    # ── README.md ───────────────────────────────────────────────────
    write(f"{root}/README.md", f"""\
        # {name}

        <!-- TODO: Project description -->

        ## Setup

        ```bash
        # Install uv if needed
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Install dependencies
        uv sync

        # Copy and fill in environment variables
        cp .env.example .env
        ```

        ## Usage

        ```bash
        # Run tests
        uv run pytest tests/
        ```

        ## Documentation

        Save reports and figures under `docs/{docs_dir}/`.
{docker_files}
    """)

    if docker:
        write(f"{root}/Dockerfile", f"""\
            FROM python:{python_version}-slim

            COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

            WORKDIR /app

            COPY pyproject.toml README.md ./
            COPY src/ src/
            COPY scripts/ scripts/
            RUN uv sync --no-dev

            COPY tests/ tests/
            COPY CLAUDE.md ./

            ENTRYPOINT ["uv", "run"]
            CMD ["pytest", "tests/"]
        """)

        write(f"{root}/.dockerignore", """\
            .git
            .venv
            __pycache__/
            *.pyc
            .pytest_cache/
            docs/
            dist/
            build/
            .env
        """)

    return root


def main():
    parser = argparse.ArgumentParser(description="Bootstrap a biomedical ML research project")
    parser.add_argument("name", help="Project name (e.g., my-study)")
    parser.add_argument("--path", default=".", help="Parent directory (default: current dir)")
    parser.add_argument("--python-version", default="3.10",
                        choices=["3.10", "3.11", "3.12"],
                        help="Python version (default: 3.10)")
    parser.add_argument("--docker", action="store_true",
                        help="Add Dockerfile and .dockerignore for a reproducible container setup")
    parser.add_argument("--dependency", action="append", default=[],
                        help="Add an extra runtime dependency (repeatable)")
    args = parser.parse_args()

    root = init_project(
        args.name,
        args.path,
        args.python_version,
        docker=args.docker,
        extra_dependencies=args.dependency,
    )
    print(f"Project created at: {root}")
    print(f"\nNext steps:")
    print(f"  cd {root}")
    print(f"  uv sync")
    print(f"  uv run pytest tests/")
    if args.docker:
        print(f"  docker build -t {args.name} .")


if __name__ == "__main__":
    main()
