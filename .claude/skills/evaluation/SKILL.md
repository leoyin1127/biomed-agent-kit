---
name: evaluation
description: >
  Evaluate biomedical ML models with appropriate metrics and confidence
  intervals. Use when: (1) Computing classification metrics (AUC-ROC,
  balanced accuracy, sensitivity, specificity, F1) with confidence intervals,
  (2) Evaluating segmentation models (Dice, IoU, Hausdorff, surface Dice),
  (3) Survival analysis (C-index, Kaplan-Meier, Cox PH, time-dependent AUC),
  (4) Statistical comparison between models (Wilcoxon, paired t-test).
---

# Evaluation Metrics

## References

| File | Apply When |
|------|------------|
| [references/evaluation-metrics.md](references/evaluation-metrics.md) | Classification metrics, confidence intervals, statistical comparisons |
| [references/segmentation-metrics.md](references/segmentation-metrics.md) | Dice, IoU, Hausdorff (HD95), surface Dice, multi-class segmentation |
| [references/survival-metrics.md](references/survival-metrics.md) | C-index, Kaplan-Meier, log-rank, Cox PH, time-dependent AUC, bootstrap CIs |

Note: `segmentation-metrics.md` references `compute_ci()` from `evaluation-metrics.md`.
