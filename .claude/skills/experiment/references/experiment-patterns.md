# Experiment Orchestration Patterns

## Experiment Grid Generation

Generate all combinations from experiment dimensions (e.g., tasks, feature sets, model architectures):

```python
from itertools import product

def generate_experiments(dimensions: dict[str, list], train_cfg):
    """Generate experiment grid from arbitrary dimensions.

    dimensions: {"task": [...], "feature_set": [...], "model": [...]}
    """
    keys = list(dimensions.keys())
    experiments = []
    for combo in product(*dimensions.values()):
        config = dict(zip(keys, combo))
        experiments.append(ExperimentConfig(**config, train=train_cfg))
    # Sort for cache efficiency (group by whichever dim is most expensive to load)
    experiments.sort(key=lambda e: tuple(getattr(e, k) for k in keys))
    return experiments
```

## Resumable Completion Tracking

Use a JSON file with file locking to track completed experiments:

```python
import fcntl, json, os

def load_completed(output_dir: str) -> set[str]:
    path = os.path.join(output_dir, "results", "_completed.json")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return set(json.load(f))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def mark_completed(output_dir: str, experiment_id: str) -> None:
    path = os.path.join(output_dir, "results", "_completed.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read().strip()
            completed = set(json.loads(content)) if content else set()
            completed.add(experiment_id)
            f.seek(0)
            f.truncate()
            f.write(json.dumps(sorted(completed), indent=2))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
```

## Cross-Validation Split Generation

Stratified K-fold with reproducible seeds:

```python
from sklearn.model_selection import StratifiedKFold

def create_splits(labels, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for train_idx, test_idx in skf.split(range(len(labels)), labels):
        # Train/val sub-split: see data-splitting.md split_train_val()
        splits.append({"train": train_idx, "test": test_idx})
    return splits
```

## Results Aggregation

Collect per-fold results into a flat DataFrame with confidence intervals:

```python
def aggregate_results(summaries: list[dict], dim_keys: list[str]) -> pd.DataFrame:
    """Aggregate experiment summaries into a flat table.

    dim_keys: column names for experiment dimensions, e.g. ["task", "feature_set", "model"]
    """
    rows = []
    for s in summaries:
        row = {k: s[k] for k in dim_keys}
        for metric, data in s["test"].items():
            for stat in ("mean", "std", "ci_low", "ci_high"):
                row[f"test_{metric}_{stat}"] = data[stat]
        rows.append(row)
    return pd.DataFrame(rows)
```

## Experiment Tracking with Wandb

Use wandb for experiment logging and metric visualization:

```python
import wandb

run_name = "/".join(dim_values) + f"/fold{fold}"
wandb.init(
    project="my-project",
    name=run_name,
    config=experiment_config.to_dict(),
    reinit=True,
)
wandb.log({"epoch": epoch, "train_loss": loss, "val_metric": val_score})
wandb.finish()
```

## Per-Experiment Logging (Multi-GPU)

Redirect stdout/stderr per experiment to avoid interleaved output:

```python
import sys

log_path = os.path.join(output_dir, "logs", f"{experiment_id}.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout
```

## Directory Structure for Results

Organize by experiment dimensions (adapt directory depth to your grid):

```
results/
├── _completed.json
├── aggregated_results.csv
├── aggregated_results.json
├── <dim_1_value>/
│   ├── <dim_2_value>/
│   │   ├── <dim_3_value>/
│   │   │   ├── summary.json
│   │   │   ├── fold_0/ (checkpoints, metrics)
│   │   │   └── fold_1/
│   │   └── <dim_3_other>/
│   └── <dim_2_other>/
└── <dim_1_other>/
```
