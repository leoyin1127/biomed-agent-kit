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


def write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(textwrap.dedent(content).lstrip("\n"))


def init_project(name: str, parent: str) -> str:
    root = os.path.join(parent, name)
    if os.path.exists(root):
        print(f"Error: {root} already exists", file=sys.stderr)
        sys.exit(1)

    pkg = name.replace("-", "_")

    # ── pyproject.toml ──────────────────────────────────────────────
    write(f"{root}/pyproject.toml", f"""\
        [project]
        name = "{name}"
        version = "0.1.0"
        requires-python = ">=3.10"
        dependencies = [
            "numpy>=1.24",
            "pandas>=2.0",
            "scikit-learn>=1.3",
            "scipy>=1.10",
            "tqdm>=4.60",
            "matplotlib>=3.7",
            "python-dotenv",
            "wandb>=0.15",
        ]

        [dependency-groups]
        dev = ["pytest>=7.0", "pytest-cov"]

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["src/{pkg}"]

        [tool.pytest.ini_options]
        testpaths = ["tests"]
    """)

    # ── .python-version ─────────────────────────────────────────────
    write(f"{root}/.python-version", "3.10\n")

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
    os.makedirs(f"{root}/docs", exist_ok=True)
    write(f"{root}/docs/.gitkeep", "")

    # ── CLAUDE.md ───────────────────────────────────────────────────
    write(f"{root}/CLAUDE.md", f"""\
        # CLAUDE.md

        This file provides guidance to Claude Code when working with this repository.

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
    """)

    return root


def main():
    parser = argparse.ArgumentParser(description="Bootstrap a biomedical ML research project")
    parser.add_argument("name", help="Project name (e.g., my-study)")
    parser.add_argument("--path", default=".", help="Parent directory (default: current dir)")
    args = parser.parse_args()

    root = init_project(args.name, args.path)
    print(f"Project created at: {root}")
    print(f"\nNext steps:")
    print(f"  cd {root}")
    print(f"  uv sync")
    print(f"  uv run pytest tests/")


if __name__ == "__main__":
    main()
