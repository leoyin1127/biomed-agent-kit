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

## Bootstrap Confidence Intervals (Held-Out Test Sets)

When evaluating on a single held-out test set (not k-fold), use bootstrap:

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_ci(y_true, y_score, metric_fn, n_bootstraps=2000,
                 confidence=0.95, seed=42):
    """Bootstrap CI for any metric on a held-out test set.

    Use this instead of compute_ci() when you have a single test set
    (not k-fold cross-validation).
    """
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip if bootstrap sample has only one class
        scores.append(metric_fn(y_true[idx], y_score[idx]))
    alpha = 1 - confidence
    return {
        "mean": np.mean(scores),
        "ci_low": np.percentile(scores, 100 * alpha / 2),
        "ci_high": np.percentile(scores, 100 * (1 - alpha / 2)),
    }

# Usage
result = bootstrap_ci(y_true, y_prob, roc_auc_score)
```

**When to use which CI method:**
- `compute_ci()` (t-distribution): k-fold cross-validation with small k
- `bootstrap_ci()`: single held-out test set evaluation

**ASK the user** whether they're evaluating with cross-validation or a held-out test set to select the right CI method.

## Multi-Label Classification

For tasks with multiple non-exclusive labels (e.g., chest X-ray findings):

```python
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       label_names: list[str] | None = None) -> dict:
    """Compute per-label and macro metrics for multi-label classification.

    y_true: (N, L) binary matrix
    y_prob: (N, L) predicted probabilities
    """
    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"label_{i}" for i in range(n_labels)]

    per_label = {}
    for i, name in enumerate(label_names):
        if len(np.unique(y_true[:, i])) < 2:
            continue  # skip labels with no positive samples
        per_label[name] = {
            "auc_roc": roc_auc_score(y_true[:, i], y_prob[:, i]),
            "avg_precision": average_precision_score(y_true[:, i], y_prob[:, i]),
        }

    return {
        "per_label": per_label,
        "macro_auc_roc": roc_auc_score(y_true, y_prob, average="macro"),
        "micro_auc_roc": roc_auc_score(y_true, y_prob, average="micro"),
    }
```
