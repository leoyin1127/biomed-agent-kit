# Calibration Metrics Reference

Model calibration measures whether predicted probabilities match observed frequencies.
Critical for clinical decision-making where probability thresholds guide treatment.

## Expected Calibration Error (ECE)

```python
import numpy as np

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
    """Expected Calibration Error -- weighted average of per-bin calibration gap.

    Lower is better. A perfectly calibrated model has ECE = 0.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)
```

## Brier Score

```python
from sklearn.metrics import brier_score_loss

# Lower is better. Range [0, 1]. Decomposes into calibration + refinement.
brier = brier_score_loss(y_true, y_prob)
```

## Reliability Diagram

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray,
                             n_bins: int = 10, output_path: str = "reliability.png"):
    """Reliability diagram (calibration plot).

    Perfectly calibrated model follows the diagonal.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            bin_counts.append(0)
        else:
            bin_accs.append(y_true[mask].mean())
            bin_confs.append(y_prob[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), dpi=150,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.bar(bin_centers, bin_accs, width=1/n_bins, alpha=0.3, edgecolor="black")
    ax1.plot(bin_confs, bin_accs, "o-", color="C0", label="Model")
    ax1.set_ylabel("Observed frequency")
    ax1.set_xlabel("Predicted probability")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.hist(y_prob, bins=n_bins, range=(0, 1), edgecolor="black", alpha=0.5)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## Temperature Scaling (Post-Hoc Calibration)

```python
import torch
import torch.nn as nn
from torch.optim import LBFGS

class TemperatureScaling(nn.Module):
    """Learn a single temperature parameter on the validation set."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor,
            max_iter: int = 50):
        """Optimize temperature on validation set."""
        criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self.temperature.item()
```

## When to Assess Calibration

- **Always** for clinical decision support where thresholds matter
- When using predicted probabilities to rank or triage patients
- Before deploying a model that outputs risk scores

**ASK the user** whether their clinical application relies on probability thresholds -- if so, calibration assessment and correction are essential.

## Common Pitfalls

- **Class imbalance inflates ECE**: Use adaptive binning or classwise ECE for imbalanced datasets
- **ECE depends on bin count**: Report the number of bins used; 10-15 is typical
- **Calibration != discrimination**: A well-calibrated model can still have poor AUC. Evaluate both.
- **Post-hoc calibration needs held-out data**: Never calibrate and evaluate on the same set
