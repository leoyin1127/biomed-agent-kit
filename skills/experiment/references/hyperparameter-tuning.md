# Hyperparameter Tuning Patterns

## Optuna Study Setup

```python
import optuna
import numpy as np
import torch

def objective(trial: optuna.Trial) -> float:
    """Define hyperparameter search space and return metric to optimize."""
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    n_layers = trial.suggest_int("n_layers", 1, 4)

    model = build_model(n_layers=n_layers, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    val_metric = train_and_evaluate(model, optimizer, batch_size)
    return val_metric


study = optuna.create_study(
    direction="maximize",  # or "minimize" for loss
    study_name="my-experiment",
    storage="sqlite:///optuna.db",  # persist across restarts
    load_if_exists=True,
)
study.optimize(objective, n_trials=100, timeout=3600 * 8)

print(f"Best value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

## Pruning (Early Termination of Bad Trials)

```python
def objective_with_pruning(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, optimizer)
        val_metric = evaluate(model)

        trial.report(val_metric, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_metric

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
)
study.optimize(objective_with_pruning, n_trials=100)
```

## Multi-Objective Optimization

```python
def multi_objective(trial: optuna.Trial) -> tuple[float, float]:
    n_layers = trial.suggest_int("n_layers", 1, 8)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512)

    model = build_model(n_layers=n_layers, hidden_dim=hidden_dim)
    val_auc = train_and_evaluate(model)
    inference_time = benchmark_inference(model)
    return val_auc, inference_time

study = optuna.create_study(directions=["maximize", "minimize"])
study.optimize(multi_objective, n_trials=50)

for trial in study.best_trials:
    print(f"AUC={trial.values[0]:.4f}, time={trial.values[1]:.2f}s")
```

## Search Space Guidelines

| Hyperparameter | Typical Range | Scale |
|---------------|---------------|-------|
| Learning rate | 1e-5 to 1e-2 | log |
| Weight decay | 1e-6 to 1e-2 | log |
| Dropout | 0.0 to 0.5 | linear |
| Batch size | 8, 16, 32, 64 | categorical |
| Hidden dim | 64 to 1024 | int (step=64) |
| Number of layers | 1 to 6 | int |

**ASK the user** what hyperparameters to tune and their compute budget before setting up the study.

## Integration with Cross-Validation

```python
def objective_cv(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        model = build_model(dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        score = train_fold(model, optimizer, train_idx, val_idx)
        fold_scores.append(score)

        trial.report(np.mean(fold_scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)
```

## Common Pitfalls

- **Overfitting the validation set**: Many trials risk overfitting to validation. Use a held-out test set for final evaluation.
- **Forgetting to persist**: Always use `storage="sqlite:///..."` to resume interrupted studies.
- **Too narrow search space**: Start broad, then narrow based on initial results.
- **Ignoring pruning**: Pruning saves 50%+ compute. Always use it for expensive training.
