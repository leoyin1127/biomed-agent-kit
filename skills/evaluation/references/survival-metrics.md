# Survival Analysis Metrics Reference

## Standard Survival Metrics

For time-to-event prediction in clinical/genomics research:

| Metric | Use When | Range | Notes |
|--------|----------|-------|-------|
| **C-index (Harrell's)** | Primary ranking metric for survival models | [0, 1] | 0.5 = random, 1.0 = perfect |
| **Time-dependent AUC** | Evaluating discrimination at specific time points | [0, 1] | AUC at t=1yr, 3yr, 5yr |
| **Brier Score** | Calibration + discrimination combined | [0, 1] | Lower is better |
| **Log-rank test** | Comparing survival curves between groups | p-value | Non-parametric |
| **Integrated Brier Score** | Overall calibration across time | [0, 0.25] | Lower is better |

## Concordance Index (C-index)

The most common metric for survival prediction models:

```python
from lifelines.utils import concordance_index

# From raw predictions
c_index = concordance_index(
    event_times=durations,       # observed survival times
    predicted_scores=-risk_scores,  # negate so higher = longer survival
    event_observed=events,       # 1 = event occurred, 0 = censored
)

# Using scikit-survival (alternative)
from sksurv.metrics import concordance_index_censored

c_index, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
    event_indicator=events.astype(bool),
    event_time=durations,
    estimate=-risk_scores,
)
```

## Kaplan-Meier Estimation

Non-parametric survival curve estimation:

```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed=events, label="All patients")

# Median survival time
median = kmf.median_survival_time_

# Plot
ax = kmf.plot_survival_function()
ax.set_xlabel("Time (months)")
ax.set_ylabel("Survival probability")
```

### Stratified Kaplan-Meier (Group Comparison)

```python
from lifelines.statistics import logrank_test

# Split into groups (e.g., high-risk vs low-risk)
mask_high = risk_scores > np.median(risk_scores)

kmf_high = KaplanMeierFitter()
kmf_high.fit(durations[mask_high], events[mask_high], label="High risk")

kmf_low = KaplanMeierFitter()
kmf_low.fit(durations[~mask_high], events[~mask_high], label="Low risk")

# Log-rank test for statistical significance
result = logrank_test(
    durations[mask_high], durations[~mask_high],
    events[mask_high], events[~mask_high],
)
print(f"Log-rank p-value: {result.p_value:.4f}")

# Plot both curves
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
kmf_high.plot_survival_function(ax=ax)
kmf_low.plot_survival_function(ax=ax)
ax.set_title(f"Log-rank p = {result.p_value:.4f}")
fig.tight_layout()
```

## Cox Proportional Hazards

Semi-parametric survival model:

```python
from lifelines import CoxPHFitter

# df must have duration_col, event_col, and feature columns
cph = CoxPHFitter(penalizer=0.1)  # L2 regularization
cph.fit(df, duration_col="time", event_col="event")

# Hazard ratios and significance
cph.print_summary()

# Predict risk scores for new data
risk_scores = cph.predict_partial_hazard(X_test)

# Concordance
print(f"C-index: {cph.concordance_index_:.3f}")
```

## Time-Dependent AUC

Evaluate discrimination at clinically meaningful time points:

```python
from sksurv.metrics import cumulative_dynamic_auc

# Structured array format required by sksurv
from sksurv.util import Surv
y_train = Surv.from_arrays(events_train.astype(bool), durations_train)
y_test = Surv.from_arrays(events_test.astype(bool), durations_test)

# AUC at specific time points (e.g., 1, 3, 5 years)
times = [12, 36, 60]  # in months
auc_values, mean_auc = cumulative_dynamic_auc(
    y_train, y_test, risk_scores, times
)
for t, auc in zip(times, auc_values):
    print(f"AUC at {t} months: {auc:.3f}")
print(f"Mean AUC: {mean_auc:.3f}")
```

## Confidence Intervals for C-index

Bootstrap confidence intervals (preferred for C-index):

```python
import numpy as np
from lifelines.utils import concordance_index

def bootstrap_cindex(durations, events, risk_scores,
                     n_bootstrap=1000, confidence=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(durations)
    c_indices = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            ci = concordance_index(
                durations[idx], -risk_scores[idx], events[idx]
            )
            c_indices.append(ci)
        except ZeroDivisionError:
            continue
    alpha = (1 - confidence) / 2
    return {
        "mean": np.mean(c_indices),
        "std": np.std(c_indices),
        "ci_low": np.percentile(c_indices, 100 * alpha),
        "ci_high": np.percentile(c_indices, 100 * (1 - alpha)),
    }
```

## Cross-Validation for Survival Models

Use the same stratified K-fold as classification, but stratify by event status:

```python
from sklearn.model_selection import StratifiedKFold

def survival_cv(durations, events, features, n_splits=5, seed=42):
    """Stratify by event status to ensure each fold has censored + uncensored."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_c_indices = []
    for train_idx, test_idx in skf.split(features, events):
        # Train model on train_idx, predict on test_idx
        # ...
        fold_c_indices.append(c_index)
    return fold_c_indices  # pass to compute_ci() from evaluation-metrics.md
```

## Publication-Quality Reporting

Report format: `C-index: 0.723 (95% CI: [0.681, 0.765])`

| Model | C-index (95% CI) | AUC@1yr | AUC@3yr | AUC@5yr | Log-rank p |
|-------|-------------------|---------|---------|---------|------------|
| Cox PH | 0.723 [0.681, 0.765] | 0.741 | 0.718 | 0.695 | < 0.001 |
| DeepSurv | 0.758 [0.712, 0.804] | 0.772 | 0.751 | 0.729 | < 0.001 |

## Common Pitfalls

- **Censoring handling**: Never drop censored observations. They carry information about survival up to the censoring time.
- **Time horizon**: C-index is sensitive to follow-up duration. Report the maximum follow-up time and censoring rate.
- **Risk score sign**: Some libraries expect higher score = higher risk, others the opposite. Always check and negate if needed.
- **Proportional hazards assumption**: Cox PH assumes constant hazard ratios over time. Test with `cph.check_assumptions(df)` in lifelines.
- **Competing risks**: If patients can experience multiple event types (e.g., death from cancer vs other causes), use Fine-Gray or cause-specific models instead of standard Cox.
