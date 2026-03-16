# Curve Plots Reference

## Contents

- ROC Curve
- Precision-Recall Curve
- Kaplan-Meier Survival Curve
- When to Use Which Curve


## ROC Curve

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(results: list[dict], output_path: str,
                   title: str = "ROC Curve"):
    """Plot ROC curves for one or more models.

    results: list of {"name": str, "y_true": array, "y_prob": array}
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    for r in results:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2,
                label=f"{r['name']} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_with_ci(y_true: np.ndarray, y_prob: np.ndarray,
                     output_path: str, n_bootstraps: int = 1000,
                     seed: int = 42):
    """ROC curve with bootstrap confidence band."""
    rng = np.random.RandomState(seed)
    base_fpr = np.linspace(0, 1, 100)
    tprs = []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[idx], y_prob[idx])
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    ci_low = np.percentile(tprs, 2.5, axis=0)
    ci_high = np.percentile(tprs, 97.5, axis=0)
    mean_auc = auc(base_fpr, mean_tpr)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.plot(base_fpr, mean_tpr, "b-", linewidth=2,
            label=f"Mean ROC (AUC = {mean_auc:.3f})")
    ax.fill_between(base_fpr, ci_low, ci_high, alpha=0.2, color="blue",
                    label="95% CI")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve with Bootstrap CI", fontsize=14)
    ax.legend(loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curve(results: list[dict], output_path: str,
                  title: str = "Precision-Recall Curve"):
    """Plot precision-recall curves. Preferred over ROC for imbalanced data.

    results: list of {"name": str, "y_true": array, "y_prob": array}
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    for r in results:
        precision, recall, _ = precision_recall_curve(r["y_true"], r["y_prob"])
        ap = average_precision_score(r["y_true"], r["y_prob"])
        ax.plot(recall, precision, linewidth=2,
                label=f"{r['name']} (AP = {ap:.3f})")

    prevalence = np.mean(results[0]["y_true"])
    ax.axhline(prevalence, color="gray", linestyle="--", alpha=0.5,
               label=f"Baseline (prevalence = {prevalence:.3f})")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## Kaplan-Meier Survival Curve

```python
import matplotlib.pyplot as plt

def plot_kaplan_meier(groups: list[dict], output_path: str,
                     xlabel: str = "Time", title: str = "Kaplan-Meier Curve"):
    """Plot Kaplan-Meier survival curves for multiple groups.

    groups: list of {"name": str, "times": array, "events": array}
        events: 1 = event occurred, 0 = censored
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)

    for g in groups:
        kmf = KaplanMeierFitter()
        kmf.fit(g["times"], event_observed=g["events"], label=g["name"])
        kmf.plot_survival_function(ax=ax, ci_show=True)

    if len(groups) == 2:
        result = logrank_test(
            groups[0]["times"], groups[1]["times"],
            event_observed_A=groups[0]["events"],
            event_observed_B=groups[1]["events"],
        )
        ax.text(0.6, 0.9, f"Log-rank p = {result.p_value:.4f}",
                transform=ax.transAxes, fontsize=11,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## When to Use Which Curve

| Curve | Use When |
|-------|----------|
| **ROC** | Binary classification; comparing models at all thresholds |
| **ROC with CI** | Single test set; want uncertainty quantification |
| **Precision-Recall** | Class-imbalanced data; focus on positive class |
| **Kaplan-Meier** | Survival/time-to-event; comparing patient groups |

**ASK the user** whether their dataset is balanced or imbalanced -- for imbalanced data, PR curves are more informative than ROC.
