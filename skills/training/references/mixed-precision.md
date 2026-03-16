# Mixed Precision Training Reference

## Contents

- Standard AMP Training (fp16)
- bf16 Training (Ampere+ GPUs)
- fp16 vs bf16 Comparison
- Disabling AMP for Specific Operations
- Checking GPU bf16 Support
- Common Pitfalls


**ASK the user** what GPU they have -- bf16 requires Ampere (A100, RTX 3090) or newer; older GPUs must use fp16.

## Standard AMP Training (fp16)

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch_amp(model, loader, optimizer, criterion, device,
                        scaler: GradScaler, accumulation_steps: int = 1):
    """Training loop with automatic mixed precision (fp16)."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(loader)


# Setup
scaler = GradScaler()
# Pass scaler to training loop
train_one_epoch_amp(model, loader, optimizer, criterion, device, scaler)
```

## bf16 Training (Ampere+ GPUs)

```python
import torch

def train_one_epoch_bf16(model, loader, optimizer, criterion, device,
                         accumulation_steps: int = 1):
    """Training loop with bf16 -- no GradScaler needed.

    bf16 has the same exponent range as fp32, so no overflow/underflow issues.
    Requires Ampere GPU (A100, RTX 3090) or newer.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accumulation_steps

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    return total_loss / len(loader)
```

## fp16 vs bf16 Comparison

| Property | fp16 | bf16 |
|----------|------|------|
| Exponent bits | 5 | 8 (same as fp32) |
| Mantissa bits | 10 | 7 |
| Dynamic range | Narrow (needs GradScaler) | Wide (no scaler needed) |
| Precision | Higher | Lower |
| GPU requirement | Any CUDA GPU | Ampere+ (A100, RTX 3090+) |
| NaN risk | Higher (overflow) | Lower |
| Speed | ~2x fp32 | ~2x fp32 |

**Recommendation**: Use bf16 if your GPU supports it. Fall back to fp16 + GradScaler otherwise.

## Disabling AMP for Specific Operations

Some operations should stay in fp32 for numerical stability:

```python
with autocast():
    outputs = model(inputs)

# Compute loss in fp32 (more stable for some loss functions)
with autocast(enabled=False):
    loss = criterion(outputs.float(), targets)

scaler.scale(loss).backward()
```

## Checking GPU bf16 Support

```python
def supports_bf16() -> bool:
    """Check if current GPU supports bf16."""
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability()
    return capability >= (8, 0)  # Ampere = compute capability 8.0+

# Usage
dtype = torch.bfloat16 if supports_bf16() else torch.float16
use_scaler = dtype == torch.float16  # only needed for fp16
```

## Common Pitfalls

- **NaN losses with fp16**: fp16 has limited dynamic range. GradScaler handles this, but if you still see NaN, try increasing `init_scale` or reducing learning rate.
- **Forgetting `scaler.update()`**: Must be called after every `scaler.step()`. Without it, the loss scale never adjusts and training degrades.
- **AMP + gradient accumulation**: Scale the loss by `accumulation_steps` before `scaler.scale()`. Call `scaler.step()` and `scaler.update()` only at the accumulation boundary.
- **Model evaluation**: Don't use autocast during evaluation/inference. It can cause subtle numerical differences. Use `model.eval()` + `torch.no_grad()` in fp32.
- **Saving/loading with AMP**: Save GradScaler state alongside model/optimizer for reproducible resumption.
