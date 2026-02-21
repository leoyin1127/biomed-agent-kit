---
name: experiment
description: >
  Run systematic ML experiments with production-grade patterns. Use when:
  (1) Setting up experiment grids with cross-validation, (2) Managing GPU
  memory, multi-GPU worker pools, or OOM protection, (3) Designing
  patient-level or site-aware data splits, (4) Tracking experiment
  completion with resumability, (5) Distributing work across GPUs.
---

# Experiment Orchestration

## References

| File | Apply When |
|------|------------|
| [references/experiment-patterns.md](references/experiment-patterns.md) | Experiment grids, completion tracking, CV splits, wandb, multi-GPU distribution |
| [references/gpu-patterns.md](references/gpu-patterns.md) | GPU memory cleanup, multi-GPU pools, VRAM scaling, OOM protection, reproducibility |
| [references/data-splitting.md](references/data-splitting.md) | Patient-level splits, site-aware splits, temporal splits, class imbalance strategies |
