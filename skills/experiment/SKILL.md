---
name: experiment
description: >
  Run systematic ML experiments with production-grade patterns. Use when:
  (1) Setting up experiment grids with cross-validation, (2) Managing GPU
  memory, multi-GPU worker pools, or OOM protection, (3) Designing
  patient-level or site-aware data splits, (4) Tracking experiment
  completion with resumability, (5) Distributing work across GPUs,
  (6) Hyperparameter tuning with Optuna.
---

# Experiment Orchestration

## Workflow

Running a systematic experiment involves these steps:

1. **Design data splits** -- patient-level, site-aware, or temporal
2. **Define experiment grid** -- combinations of tasks, features, models
3. **Set up tracking** -- wandb or MLflow for logging, JSON for completion
4. **Configure GPU distribution** -- multi-GPU pools, VRAM-based scaling
5. **Run with resumability** -- file-locked completion tracking
6. **Aggregate results** -- fold-level metrics with confidence intervals

## Decision Tree

**How to split the data?**
- Medical imaging with multiple samples per patient → Patient-level splits (GroupKFold). See [data-splitting.md](references/data-splitting.md)
- Multi-site/multi-center study → Leave-one-site-out or site-aware stratified. See [data-splitting.md](references/data-splitting.md)
- Time-ordered data (EHR, longitudinal) → Temporal splits. See [data-splitting.md](references/data-splitting.md)
- Standard classification → Stratified K-Fold. See [experiment-patterns.md](references/experiment-patterns.md)

**How to track experiments?**
- Cloud-hosted, rich visualization → wandb
- On-premises, regulated environment → MLflow
- **ASK the user** which they prefer

**How to distribute across GPUs?**
- Multiple independent experiments → ProcessPoolExecutor with GPU pinning. See [gpu-patterns.md](references/gpu-patterns.md)
- Single large model → DDP (DistributedDataParallel). See [gpu-patterns.md](references/gpu-patterns.md)
- Unsure about GPU memory → Use VRAM-based worker scaling. See [gpu-patterns.md](references/gpu-patterns.md)

**Need hyperparameter tuning?**
- Yes → Optuna with pruning + SQLite persistence. See [hyperparameter-tuning.md](references/hyperparameter-tuning.md)
- **ASK the user** what hyperparameters to tune and compute budget

## References

| File | Read When |
|------|-----------|
| [references/experiment-patterns.md](references/experiment-patterns.md) | Experiment grid generation, resumable completion tracking (filelock), CV splits, wandb/MLflow logging, results aggregation |
| [references/gpu-patterns.md](references/gpu-patterns.md) | GPU memory cleanup, multi-GPU pools (ProcessPoolExecutor), VRAM scaling, OOM protection, DDP, reproducibility seeds |
| [references/data-splitting.md](references/data-splitting.md) | Patient-level splits, stratified group K-fold, site-aware splits, temporal splits, class imbalance (weighted loss, focal loss, oversampling) |
| [references/hyperparameter-tuning.md](references/hyperparameter-tuning.md) | Optuna study setup, pruning, multi-objective optimization, search space guidelines, CV integration |
