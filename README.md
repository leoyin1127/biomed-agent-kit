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

Skills trigger automatically based on what you ask Claude.

### project-scaffold

```
"Create a new project for my tumor classification study"
"Set up a new repo for lung CT analysis with uv and wandb"
"Bootstrap a research project called cell-segmentation"
"Initialize a project with proper CLAUDE.md and test structure"
```

### experiment

```
"Set up a 5-fold cross-validation grid over 3 encoders"
"Run experiments across 4 GPUs with OOM protection"
"Design patient-level splits so no patient appears in both train and test"
"Create a leave-one-site-out evaluation for our multi-center dataset"
"Set up an experiment grid over feature sets and classifiers with wandb logging"
"Add resumable experiment tracking so I can restart interrupted runs"
```

### evaluation

```
"Compute AUC-ROC with 95% confidence intervals across folds"
"Compute Dice scores and HD95 for each organ class"
"Run a Wilcoxon test comparing model A vs model B across folds"
"Calculate C-index with bootstrap confidence intervals for our survival model"
"Evaluate the segmentation model with surface Dice at 2mm tolerance"
"Compare all models with Bonferroni-corrected p-values"
```

### reporting

```
"Generate a forest plot comparing all models"
"Create a heatmap of AUC scores: feature sets vs encoders"
"Write a markdown report summarizing the experiment results"
"Make publication-quality figures at 300 DPI for the paper"
"Generate a radar chart comparing top 3 configurations across all metrics"
```

### repo-integration

```
"Integrate the pretrained model from this paper's GitHub repo"
"Add this ViT encoder from GitHub as a feature extractor in our pipeline"
"Vendor the preprocessing code from REPO and adapt it to our data format"
"Wrap this external segmentation model so it accepts our DICOM inputs"
"This repo requires torch 1.x but we use torch 2 — help me integrate it"
"Set up the pretrained weights from this paper's Google Drive link"
```

## License

MIT
