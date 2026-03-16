# Regression Metrics Reference

For quantitative biomarker prediction, treatment response measurement, and continuous outcome modeling.

## Standard Regression Metrics

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "median_ae": np.median(np.abs(y_true - y_pred)),
    }
```

## Bland-Altman Analysis

For comparing two measurement methods (common in clinical validation):

```python
import numpy as np
import matplotlib.pyplot as plt

def bland_altman_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute Bland-Altman statistics."""
    diff = y_pred - y_true
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    return {
        "mean_diff": mean_diff,  # bias
        "std_diff": std_diff,
        "loa_lower": mean_diff - 1.96 * std_diff,
        "loa_upper": mean_diff + 1.96 * std_diff,
    }


def plot_bland_altman(y_true: np.ndarray, y_pred: np.ndarray,
                      output_path: str, units: str = ""):
    """Bland-Altman plot with limits of agreement."""
    diff = y_pred - y_true
    mean_vals = (y_true + y_pred) / 2
    stats = bland_altman_analysis(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(mean_vals, diff, alpha=0.5, s=20)
    ax.axhline(stats["mean_diff"], color="red", linestyle="-",
               label=f"Mean diff: {stats['mean_diff']:.3f}")
    ax.axhline(stats["loa_upper"], color="gray", linestyle="--",
               label=f"+1.96 SD: {stats['loa_upper']:.3f}")
    ax.axhline(stats["loa_lower"], color="gray", linestyle="--",
               label=f"-1.96 SD: {stats['loa_lower']:.3f}")
    ax.set_xlabel(f"Mean of true and predicted{' (' + units + ')' if units else ''}")
    ax.set_ylabel(f"Difference (predicted - true){' (' + units + ')' if units else ''}")
    ax.set_title("Bland-Altman Plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## When to Use Which Metric

| Metric | Use When |
|--------|----------|
| **MAE** | Robust to outliers; interpretable in original units |
| **RMSE** | Penalizes large errors more; standard for optimization |
| **R-squared** | Proportion of variance explained; compare across studies |
| **Bland-Altman** | Comparing two measurement methods; clinical agreement |

**ASK the user** what units their target variable is in and whether outlier robustness matters.

## Common Pitfalls

- **R-squared can be negative**: If the model is worse than predicting the mean
- **Scale sensitivity**: MAE and RMSE are in target units; R-squared is unitless
- **Heteroscedasticity**: If error varies with magnitude, consider log-transforming or using percentage error
- **Bland-Altman assumes normal differences**: Check with a histogram first
