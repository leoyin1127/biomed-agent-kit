# biomed-agent-kit

A collection of Claude Code skills for biomedical and healthcare ML research. Each skill provides reusable patterns, code snippets, and workflows that Claude can apply when working on your projects.

## Skills

| Skill | Purpose |
|-------|---------|
| [project-scaffold](.claude/skills/project-scaffold/) | Bootstrap new research repos with uv, CLAUDE.md, and standard layout |
| [experiment](.claude/skills/experiment/) | Experiment grids, GPU management, data splitting |
| [evaluation](.claude/skills/evaluation/) | Classification, segmentation, and survival metrics with confidence intervals |
| [reporting](.claude/skills/reporting/) | Publication-quality figures and markdown reports |
| [repo-integration](.claude/skills/repo-integration/) | Integrate published GitHub repos and pretrained models into your project |

## Setup

Install skills into your project by copying the `.claude/skills/` directory, or install individual skills as needed.

### Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for package management

## Usage

Skills trigger automatically based on what you ask Claude. Examples:

```
# Triggers project-scaffold
"Create a new project for my tumor classification study"

# Triggers experiment
"Set up a 5-fold cross-validation grid over 3 encoders"

# Triggers evaluation
"Compute Dice scores with 95% confidence intervals"

# Triggers reporting
"Generate a forest plot comparing all models"

# Triggers repo-integration
"Integrate the pretrained model from this paper's GitHub repo"
```

## License

MIT
