# Evaluation Metrics Reference

## Standard Classification Metrics

For binary classification in biomedical research:

| Metric | Use When | Formula |
|--------|----------|---------|
| **AUC-ROC** | Primary ranking metric; threshold-independent | `roc_auc_score(y_true, y_prob)` |
| **Balanced Accuracy** | Class-imbalanced datasets (common in clinical) | `balanced_accuracy_score(y_true, y_pred)` |
| **Sensitivity (Recall)** | Minimizing false negatives matters (screening) | `TP / (TP + FN)` |
| **Specificity** | Minimizing false positives matters (confirmatory) | `TN / (TN + FP)` |
| **F1 Score** | Balancing precision and recall | `f1_score(y_true, y_pred)` |
| **Accuracy** | Balanced datasets only; misleading otherwise | `accuracy_score(y_true, y_pred)` |

## Confidence Intervals via Student's t-distribution

For k-fold cross-validation with small k (typically 5):

```python
from scipy import stats
import numpy as np

def compute_ci(values, confidence=0.95):
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    se = std / np.sqrt(n)
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    return {
        "mean": mean,
        "std": std,
        "ci_low": mean - t_crit * se,
        "ci_high": mean + t_crit * se,
    }
```

Use t-distribution (not z/normal) because k-fold sample sizes are small (typically n=5).

## Multi-Class Extensions

- AUC-ROC: `roc_auc_score(y_true, y_probs, multi_class="ovr")`
- For >2 classes, sensitivity/specificity become per-class or use macro averaging

## Statistical Comparison Between Models

When comparing two models across folds:

```python
from scipy.stats import wilcoxon

# Paired Wilcoxon signed-rank test (non-parametric, suitable for small n)
stat, p_value = wilcoxon(model_a_aucs, model_b_aucs)

# Paired t-test (if normality assumption holds)
from scipy.stats import ttest_rel
stat, p_value = ttest_rel(model_a_aucs, model_b_aucs)
```

Apply Bonferroni correction when making multiple comparisons:
`corrected_alpha = 0.05 / num_comparisons`

## Publication-Quality Reporting

Report format: `metric (95% CI: [low, high])`

Example: `AUC-ROC: 0.847 (95% CI: [0.812, 0.882])`

For tables, include mean +/- std and CI:

| Encoder | AUC-ROC | 95% CI | Balanced Acc |
|---------|---------|--------|--------------|
| Model A | 0.847 +/- 0.032 | [0.812, 0.882] | 0.791 |
