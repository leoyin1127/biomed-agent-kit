# Scientific Reporting Patterns

## Figure Types for Benchmarking

### Heatmaps (Dimension A x Dimension B Performance)

Useful for comparing two experiment dimensions (e.g., feature set x model architecture):

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(df, metric_col, filter_col, filter_val,
                 row_col, col_col, output_path):
    """Generic heatmap: pivot df on two dimensions, color by metric.

    Example: plot_heatmap(df, "test_auc_roc_mean", "task", "classification",
                          "feature_set", "model", "fig.png")
    """
    subset = df[df[filter_col] == filter_val] if filter_col else df
    pivot = subset.pivot(index=row_col, columns=col_col, values=metric_col)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

### Forest Plots (Metric with Confidence Intervals)

Show point estimates with CI bars for each experiment:

```python
def plot_forest(df, metric_prefix, label_cols, output_path):
    """Generic forest plot for any metric with CI columns.

    metric_prefix: e.g. "test_auc_roc" (expects _mean, _ci_low, _ci_high cols)
    label_cols: list of column names to combine into labels
    """
    df_sorted = df.sort_values(f"{metric_prefix}_mean", ascending=True)
    labels = [" / ".join(str(r[c]) for c in label_cols) for _, r in df_sorted.iterrows()]
    means = df_sorted[f"{metric_prefix}_mean"]
    ci_low = df_sorted[f"{metric_prefix}_ci_low"]
    ci_high = df_sorted[f"{metric_prefix}_ci_high"]
    y = range(len(labels))

    fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.35)), dpi=150)
    ax.errorbar(means, y, xerr=[means - ci_low, ci_high - means],
                fmt="o", capsize=3, markersize=5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(metric_prefix.replace("_", " ").title())
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

### Radar/Spider Plots (Multi-Metric Comparison)

Compare top configurations across multiple metrics simultaneously.

### Sensitivity vs Specificity Scatter

Each point is one experiment; shows trade-off between clinical metrics.

## Dated Output Directory

Always write reports to `docs/<YYYYMMDD>/`:

```python
from datetime import date
import os

report_dir = os.path.join("docs", date.today().strftime("%Y%m%d"))
os.makedirs(report_dir, exist_ok=True)
```

## Markdown Report Structure

Adapt column headers to match your experiment dimensions:

```markdown
# Experiment Report - {DATE}

## Summary
{1-2 paragraph overview of key findings}

## Results

### {TASK_OR_SUBTASK}
| {Dim 1} | {Dim 2} | Primary Metric (95% CI) | Secondary Metric | ... |
|---------|---------|-------------------------|------------------|-----|
{rows}

### Best Configurations
{top-3 per task with discussion}

## Figures
![Heatmap](fig_heatmap.png)
![Forest Plot](fig_forest.png)

## Conclusions
{key takeaways, limitations, next steps}
```

## Figure Quality Guidelines

- DPI: 150+ for review, 300+ for publication
- Font size: 8-12pt for labels, 14pt for titles
- Color maps: sequential (YlOrRd) for performance, diverging (RdBu) for comparisons
- Always include axis labels and units
- Use `fig.tight_layout()` or `bbox_inches="tight"`
