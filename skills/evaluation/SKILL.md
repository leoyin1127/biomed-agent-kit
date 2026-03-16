---
name: evaluation
description: >
  Evaluate biomedical ML models with appropriate metrics and confidence
  intervals. Use when: (1) Computing classification metrics (AUC-ROC,
  balanced accuracy, sensitivity, specificity, F1) with confidence intervals,
  (2) Evaluating segmentation models (Dice, IoU, Hausdorff, surface Dice),
  (3) Survival analysis (C-index, Kaplan-Meier, Cox PH, time-dependent AUC),
  (4) Statistical comparison between models (Wilcoxon, paired t-test),
  (5) Calibration assessment (Brier score, ECE, reliability diagrams),
  (6) Regression metrics (MAE, RMSE, R-squared, Bland-Altman),
  (7) Multi-label classification metrics.
---

# Evaluation Metrics

## Workflow

Evaluating a biomedical model involves these steps:

1. **Identify the task type** -- classification, segmentation, survival, regression, or multi-label
2. **Select primary and secondary metrics** -- based on clinical relevance
3. **Compute metrics with confidence intervals** -- t-distribution for k-fold, bootstrap for held-out
4. **Assess calibration** -- if probability thresholds guide clinical decisions
5. **Compare models statistically** -- Wilcoxon or paired t-test with correction

## Decision Tree

**What is the task type?**
- Binary/multi-class classification → AUC-ROC, balanced accuracy, sensitivity, specificity, F1. See [evaluation-metrics.md](references/evaluation-metrics.md)
- Multi-label classification → Per-label AUC, macro/micro AUC, average precision. See [evaluation-metrics.md](references/evaluation-metrics.md)
- Segmentation → Dice, IoU, HD95, surface Dice. See [segmentation-metrics.md](references/segmentation-metrics.md)
- Survival / time-to-event → C-index, time-dependent AUC, Brier score, Kaplan-Meier. See [survival-metrics.md](references/survival-metrics.md)
- Regression / biomarker prediction → MAE, RMSE, R-squared, Bland-Altman. See [regression-metrics.md](references/regression-metrics.md)

**How to compute confidence intervals?**
- K-fold cross-validation → t-distribution CI (`compute_ci()` in evaluation-metrics.md)
- Single held-out test set → Bootstrap CI (`bootstrap_ci()` in evaluation-metrics.md)
- **ASK the user** which evaluation setup they have

**Is calibration important?**
- Model outputs risk scores / probabilities → Yes, assess ECE + reliability diagram. See [calibration-metrics.md](references/calibration-metrics.md)
- Ranking or comparison only → Calibration optional

**Comparing multiple models?**
- 2 models on same folds → Wilcoxon signed-rank test (non-parametric)
- Multiple models → Apply Bonferroni correction
- See [evaluation-metrics.md](references/evaluation-metrics.md) statistical comparison section

**ASK the user** before starting:
- What is the task type?
- Cross-validation or held-out test set?
- Is model calibration clinically relevant?

## References

| File | Read When |
|------|-----------|
| [references/evaluation-metrics.md](references/evaluation-metrics.md) | Classification metrics (AUC, sensitivity, specificity, F1), CIs (t-distribution + bootstrap), multi-label, statistical model comparison |
| [references/segmentation-metrics.md](references/segmentation-metrics.md) | Dice, IoU, Hausdorff (HD95), surface Dice, multi-class segmentation, per-subject aggregation |
| [references/survival-metrics.md](references/survival-metrics.md) | C-index, Kaplan-Meier, log-rank, Cox PH, time-dependent AUC, bootstrap CIs for survival |
| [references/calibration-metrics.md](references/calibration-metrics.md) | ECE, Brier score, reliability diagrams, temperature scaling for post-hoc calibration |
| [references/regression-metrics.md](references/regression-metrics.md) | MAE, RMSE, R-squared, Bland-Altman analysis and plots for quantitative biomarker prediction |

Note: `segmentation-metrics.md` references `compute_ci()` from `evaluation-metrics.md`.
