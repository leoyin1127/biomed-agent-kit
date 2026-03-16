# Data Splitting Patterns Reference

## Patient-Level Splits (No Data Leakage)

In biomedical data, a single patient often has multiple samples (slices, tiles, time
points). Splitting at the sample level causes data leakage — the model sees the same
patient in both train and test.

**Always split at the patient level:**

```python
from sklearn.model_selection import GroupKFold

def patient_level_kfold(features, labels, patient_ids, n_splits=5):
    """K-fold where all samples from a patient stay in the same fold."""
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in gkf.split(features, labels, groups=patient_ids):
        splits.append({"train": train_idx, "test": test_idx})
    return splits
```

### Stratified Group K-Fold

Maintain both patient-level integrity and class balance:

```python
from sklearn.model_selection import StratifiedGroupKFold

def stratified_patient_kfold(features, labels, patient_ids, n_splits=5, seed=42):
    """Stratified by label, grouped by patient."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in sgkf.split(features, labels, groups=patient_ids):
        splits.append({"train": train_idx, "test": test_idx})
    return splits
```

## Site/Institution-Aware Splits

Multi-site studies should evaluate generalization across institutions:

```python
def leave_one_site_out(features, labels, site_ids):
    """Each fold holds out one institution for testing.

    Use when: multi-center studies where site-level confounders
    (scanner type, staining protocol, patient demographics) exist.
    """
    unique_sites = sorted(set(site_ids))
    splits = []
    for held_out in unique_sites:
        test_idx = [i for i, s in enumerate(site_ids) if s == held_out]
        train_idx = [i for i, s in enumerate(site_ids) if s != held_out]
        splits.append({
            "train": train_idx,
            "test": test_idx,
            "held_out_site": held_out,
        })
    return splits

def site_aware_stratified(features, labels, patient_ids, site_ids,
                          n_splits=5, seed=42):
    """Stratify by label, group by patient, and ensure each fold has
    representation from multiple sites (not leave-one-out).

    Use when: you want within-site diversity in every fold rather than
    strict site-level holdout.
    """
    # Combine patient + site into a composite group key
    # GroupKFold won't split a group across folds
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in sgkf.split(features, labels, groups=patient_ids):
        splits.append({"train": train_idx, "test": test_idx})
    return splits
```

## Temporal Splits (Longitudinal Data)

For studies with time-ordered data, prevent temporal leakage:

```python
def temporal_split(features, labels, timestamps, train_cutoff, val_cutoff=None):
    """Train on past data, test on future data.

    Use when: data has a natural time ordering (EHR records, longitudinal
    imaging, clinical trials). Prevents look-ahead bias.
    """
    train_idx = [i for i, t in enumerate(timestamps) if t < train_cutoff]
    if val_cutoff:
        val_idx = [i for i, t in enumerate(timestamps)
                   if train_cutoff <= t < val_cutoff]
        test_idx = [i for i, t in enumerate(timestamps) if t >= val_cutoff]
        return {"train": train_idx, "val": val_idx, "test": test_idx}
    test_idx = [i for i, t in enumerate(timestamps) if t >= train_cutoff]
    return {"train": train_idx, "test": test_idx}
```

## Train/Validation Split from Training Fold

Split the training fold into train + validation for early stopping:

```python
from sklearn.model_selection import StratifiedGroupKFold

def split_train_val(train_features, train_labels, train_patient_ids,
                    val_fraction=0.2, seed=42):
    """Split a training fold into train/val, respecting patient groups."""
    n_val_splits = max(2, round(1.0 / val_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=seed)
    # Take the first split: fold 0 as val, rest as train
    for sub_train_idx, val_idx in sgkf.split(
        train_features, train_labels, groups=train_patient_ids
    ):
        return sub_train_idx, val_idx
```

## Class Imbalance Strategies

### Weighted Loss

Apply class weights inversely proportional to frequency:

```python
import numpy as np
import torch
import torch.nn as nn

def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights."""
    classes, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(classes) * counts)
    return torch.tensor(weights, dtype=torch.float32)

# Usage
weights = compute_class_weights(train_labels)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

### Focal Loss

Down-weight easy examples, focus on hard ones (useful for extreme imbalance):

```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal loss for class-imbalanced classification."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()
```

### Oversampling (WeightedRandomSampler)

```python
from torch.utils.data import WeightedRandomSampler

def balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Oversample minority classes to balance each batch."""
    class_counts = np.bincount(labels)
    sample_weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
```

## Validation Checklist

Before running experiments, verify your splitting strategy:

- [ ] **No patient leakage**: All samples from one patient are in the same fold
- [ ] **Class balance**: Each fold has similar label distribution (check with `np.bincount`)
- [ ] **Site diversity**: Each fold has samples from multiple sites (if multi-center)
- [ ] **Sufficient test size**: Each fold has enough samples for stable metric estimates
- [ ] **Reproducibility**: All splits use fixed seeds and are deterministic

```python
def validate_splits(splits, labels, patient_ids):
    """Sanity check that no patient appears in both train and test."""
    for i, split in enumerate(splits):
        train_patients = set(patient_ids[j] for j in split["train"])
        test_patients = set(patient_ids[j] for j in split["test"])
        overlap = train_patients & test_patients
        assert len(overlap) == 0, f"Fold {i}: patient leakage! {overlap}"
        # Check label distribution
        train_dist = np.bincount(labels[split["train"]])
        test_dist = np.bincount(labels[split["test"]])
        print(f"Fold {i}: train={train_dist}, test={test_dist}")
```

## Common Pitfalls

- **Slide-level != Patient-level**: In histopathology, one patient may have multiple slides. Split by patient, not slide.
- **Data augmentation after splitting**: Never augment before splitting; augmented copies of the same sample would leak into test.
- **Information leakage through normalization**: Fit scalers/normalizers on training data only, then transform validation/test.
- **Ignoring class imbalance in validation**: Use stratified splits for validation too, not just test.
- **Multiple comparisons**: When evaluating multiple models on the same splits, apply Bonferroni or FDR correction.
