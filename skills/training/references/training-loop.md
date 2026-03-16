# Training Loop Reference

## Contents

- Standard Training Loop
- Early Stopping
- LR Scheduling
- Checkpoint Management
- Complete Training Function
- Common Pitfalls


**ASK the user** which metric to monitor for early stopping and whether they want to minimize (loss) or maximize (AUC, accuracy) it.

## Standard Training Loop

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device,
                    accumulation_steps: int = 1) -> float:
    """Train for one epoch with optional gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(tqdm(loader, desc="Training")):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        all_preds.append(outputs.cpu())
        all_targets.append(targets.cpu())

    return {
        "loss": total_loss / len(loader),
        "preds": torch.cat(all_preds),
        "targets": torch.cat(all_targets),
    }
```

## Early Stopping

```python
from dataclasses import dataclass

@dataclass
class EarlyStopping:
    """Early stopping with patience and best model restoration.

    ASK the user: which metric to monitor and whether to minimize or maximize.
    """
    patience: int = 10
    mode: str = "max"  # "min" for loss, "max" for AUC/accuracy
    min_delta: float = 1e-4

    def __post_init__(self):
        self.best_score: float | None = None
        self.counter: int = 0
        self.best_state: dict | None = None
        self.should_stop: bool = False

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Update state. Returns True if improved."""
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return True

        improved = (
            score > self.best_score + self.min_delta if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def restore_best(self, model: torch.nn.Module):
        """Restore the best model weights."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
```

## LR Scheduling

```python
import torch.optim as optim

# ── ReduceLROnPlateau (most common for research) ───────────────
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-7
)
# Call after validation: scheduler.step(val_metric)

# ── Cosine Annealing ───────────────────────────────────────────
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-7
)
# Call per epoch: scheduler.step()

# ── OneCycleLR (aggressive, good for fine-tuning) ──────────────
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3,
    steps_per_epoch=len(train_loader), epochs=num_epochs
)
# Call per batch: scheduler.step()

# ── Warmup + Cosine Decay ──────────────────────────────────────
def warmup_cosine_scheduler(optimizer, warmup_epochs: int,
                            total_epochs: int) -> optim.lr_scheduler.SequentialLR:
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-7
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
```

## Checkpoint Management

```python
import os
import torch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler, epoch: int, metric: float,
                    output_dir: str, is_best: bool = False):
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metric": metric,
    }
    torch.save(state, os.path.join(output_dir, "last_checkpoint.pt"))
    if is_best:
        torch.save(state, os.path.join(output_dir, "best_model.pt"))


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler=None) -> int:
    """Load checkpoint and return the epoch to resume from."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"]
```

## Complete Training Function

```python
def train(model, train_loader, val_loader, optimizer, criterion,
          scheduler, device, num_epochs, output_dir,
          early_stopping_patience=10, monitor_metric="val_auc",
          monitor_mode="max", accumulation_steps=1):
    """Full training loop with early stopping, checkpointing, and scheduling."""

    stopper = EarlyStopping(patience=early_stopping_patience, mode=monitor_mode)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, accumulation_steps
        )
        val_result = evaluate(model, val_loader, criterion, device)

        # Compute validation metric (adapt to your task)
        val_metric = compute_metric(val_result["preds"], val_result["targets"])

        # LR scheduling
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metric)
        else:
            scheduler.step()

        # Early stopping + checkpointing
        improved = stopper(val_metric, model)
        save_checkpoint(model, optimizer, scheduler, epoch, val_metric,
                        output_dir, is_best=improved)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} "
              f"val_loss={val_result['loss']:.4f} "
              f"{monitor_metric}={val_metric:.4f} lr={lr:.2e}")

        if stopper.should_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    stopper.restore_best(model)
    return model
```

## Common Pitfalls

- **Forgetting `model.train()` / `model.eval()`**: BatchNorm and Dropout behave differently. Always set the mode.
- **Not zeroing gradients**: `optimizer.zero_grad()` must be called. With gradient accumulation, call it after the accumulation step, not every batch.
- **LR scheduler timing**: `ReduceLROnPlateau.step(metric)` is called per-epoch with the metric value. `OneCycleLR.step()` is called per-batch. Getting this wrong silently breaks training.
- **Saving optimizer state**: If you want to resume training, save optimizer and scheduler states too, not just model weights.
- **GPU memory leak in evaluation**: Always wrap evaluation in `@torch.no_grad()` or `with torch.no_grad():`. Without it, computation graphs accumulate and cause OOM.
