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
import json
import os

from filelock import FileLock  # pip install filelock (cross-platform)

def load_completed(output_dir: str) -> set[str]:
    path = os.path.join(output_dir, "results", "_completed.json")
    lock_path = path + ".lock"
    if not os.path.exists(path):
        return set()
    with FileLock(lock_path):
        with open(path) as f:
            return set(json.load(f))

def mark_completed(output_dir: str, experiment_id: str) -> None:
    path = os.path.join(output_dir, "results", "_completed.json")
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with FileLock(lock_path):
        completed = set()
        if os.path.exists(path):
            with open(path) as f:
                completed = set(json.load(f))
        completed.add(experiment_id)
        with open(path, "w") as f:
            json.dump(sorted(completed), f, indent=2)
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

## Experiment Tracking

### Wandb

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

### MLflow

Use MLflow when data must stay on-premises (common in clinical/regulated settings):

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlruns.db")  # or a remote server
mlflow.set_experiment("my-experiment")

with mlflow.start_run(run_name=run_name):
    mlflow.log_params(experiment_config.to_dict())
    mlflow.log_metrics({"val_auc": val_score, "val_loss": val_loss}, step=epoch)
    mlflow.log_artifact(model_path)
```

**ASK the user** which tracking tool they prefer -- wandb (cloud-hosted, richer UI) or MLflow (self-hosted, better for regulated environments where data cannot leave the organization).

## Per-Experiment Logging (Multi-GPU)

Use Python's logging module instead of redirecting stdout:

```python
import logging
import os

def setup_experiment_logger(output_dir: str, experiment_id: str) -> logging.Logger:
    """Create a per-experiment logger that writes to a dedicated log file."""
    log_path = os.path.join(output_dir, "logs", f"{experiment_id}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(experiment_id)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    return logger

# Usage in worker process
logger = setup_experiment_logger(output_dir, experiment_id)
logger.info(f"Starting experiment {experiment_id}")
logger.info(f"Epoch {epoch}: train_loss={loss:.4f}, val_auc={val_auc:.4f}")
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
