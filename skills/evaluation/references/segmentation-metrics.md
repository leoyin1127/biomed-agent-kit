# Segmentation Metrics Reference

## Contents

- Standard Segmentation Metrics
- Core Metric Implementations
- Multi-Class Segmentation
- Confidence Intervals for Segmentation
- Publication-Quality Reporting
- Common Pitfalls


## Standard Segmentation Metrics

For pixel/voxel-level segmentation in medical imaging:

| Metric | Use When | Range | Notes |
|--------|----------|-------|-------|
| **Dice (F1)** | Primary overlap metric; most commonly reported | [0, 1] | Sensitive to small structures |
| **IoU (Jaccard)** | Alternative overlap metric; stricter than Dice | [0, 1] | Dice = 2*IoU / (1+IoU) |
| **Hausdorff Distance** | Boundary accuracy matters (surgical planning) | [0, inf) | Sensitive to outliers |
| **HD95** | Robust boundary metric; ignores worst 5% | [0, inf) | Preferred over HD in practice |
| **Surface Dice (NSD)** | Clinically acceptable boundary tolerance | [0, 1] | Requires tolerance parameter (tau) |
| **Sensitivity (Recall)** | Under-segmentation is costly (tumor coverage) | [0, 1] | TP / (TP + FN) at pixel level |
| **Precision** | Over-segmentation is costly (false alarms) | [0, 1] | TP / (TP + FP) at pixel level |

## Core Metric Implementations

### Dice Coefficient and IoU

```python
import numpy as np

def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice coefficient for binary masks. Returns 1.0 if both are empty."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    intersection = (pred & target).sum()
    return 2.0 * intersection / (pred.sum() + target.sum())

def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """IoU (Jaccard index) for binary masks. Returns 1.0 if both are empty."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    intersection = (pred & target).sum()
    union = (pred | target).sum()
    return intersection / union
```

### Hausdorff Distance and HD95

```python
from scipy.ndimage import distance_transform_edt

def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Hausdorff distance between binary mask boundaries."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")  # undefined if either mask is empty
    dt_pred = distance_transform_edt(~pred)
    dt_target = distance_transform_edt(~target)
    return max(dt_pred[target].max(), dt_target[pred].max())

def hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th percentile Hausdorff distance (robust to outliers)."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")
    dt_pred = distance_transform_edt(~pred)
    dt_target = distance_transform_edt(~target)
    d_pred_to_target = dt_target[pred]
    d_target_to_pred = dt_pred[target]
    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return np.percentile(all_distances, 95)
```

### Surface Dice (Normalized Surface Distance)

```python
def surface_dice(pred: np.ndarray, target: np.ndarray,
                 tolerance: float = 2.0, spacing: tuple = (1.0, 1.0)) -> float:
    """Surface Dice at a given tolerance (in mm).

    tolerance: maximum acceptable distance (tau) in physical units.
    spacing: pixel/voxel spacing in mm (e.g., (0.5, 0.5) or (1.0, 1.0, 3.0) for 3D).
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0
    dt_pred = distance_transform_edt(~pred, sampling=spacing)
    dt_target = distance_transform_edt(~target, sampling=spacing)
    # Surface pixels: boundary of each mask
    from scipy.ndimage import binary_erosion
    pred_border = pred ^ binary_erosion(pred)
    target_border = target ^ binary_erosion(target)
    pred_on_target = (dt_target[pred_border] <= tolerance).sum()
    target_on_pred = (dt_pred[target_border] <= tolerance).sum()
    return (pred_on_target + target_on_pred) / (pred_border.sum() + target_border.sum())
```

## Multi-Class Segmentation

Compute per-class metrics and aggregate:

```python
def per_class_dice(pred: np.ndarray, target: np.ndarray,
                   classes: list[int]) -> dict[int, float]:
    """Dice per class, excluding background (class 0 by convention)."""
    scores = {}
    for c in classes:
        scores[c] = dice_score((pred == c), (target == c))
    return scores

def macro_dice(pred: np.ndarray, target: np.ndarray,
               classes: list[int]) -> float:
    """Mean Dice across classes (macro average)."""
    scores = per_class_dice(pred, target, classes)
    return np.mean(list(scores.values()))
```

## Confidence Intervals for Segmentation

Use `compute_ci()` from evaluation-metrics.md:

```python
# Per-subject Dice across test set
subject_dices = [dice_score(pred_i, target_i) for pred_i, target_i in zip(preds, targets)]
ci = compute_ci(subject_dices)
```

## Publication-Quality Reporting

Report format: `Dice: 0.847 (95% CI: [0.812, 0.882])`

| Structure | Dice | 95% CI | HD95 (mm) | IoU |
|-----------|------|--------|-----------|-----|
| Tumor | 0.847 +/- 0.032 | [0.812, 0.882] | 3.21 | 0.735 |
| Organ | 0.921 +/- 0.018 | [0.901, 0.941] | 1.45 | 0.854 |

## Common Pitfalls

- **Empty predictions**: Always handle the case where pred or target is empty; don't let 0/0 become NaN.
- **Spacing matters**: HD and NSD are in physical units (mm). Always pass voxel spacing or results are in pixel units.
- **Per-subject, not per-pixel**: Aggregate Dice per subject/image, then compute mean and CI across subjects. Don't compute a single global Dice across all pixels.
- **Background class**: Exclude background (class 0) from macro-averaged metrics; it inflates scores.
