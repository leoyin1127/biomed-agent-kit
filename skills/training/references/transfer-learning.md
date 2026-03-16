# Transfer Learning Reference

## Contents

- Decision Guide
- Freeze / Unfreeze Backbone
- Progressive Unfreezing
- Discriminative Learning Rates
- Layer-Wise LR Decay
- Loading Pretrained Weights
- Common Pitfalls


**ASK the user** about their dataset size relative to pretraining data -- this determines whether to freeze, fine-tune, or train from scratch.

## Decision Guide

| Dataset Size | Similarity to Pretrained | Strategy |
|-------------|--------------------------|----------|
| Small (<1K) | High | Freeze backbone, train head only |
| Small (<1K) | Low | Freeze backbone, train head (risk of overfitting) |
| Medium (1K-10K) | High | Fine-tune with low LR |
| Medium (1K-10K) | Low | Fine-tune with discriminative LR |
| Large (>10K) | Any | Fine-tune all or train from scratch |

## Freeze / Unfreeze Backbone

```python
import torch.nn as nn

def freeze_backbone(model: nn.Module, freeze_bn: bool = True):
    """Freeze all backbone parameters, keep head trainable.

    freeze_bn: also freeze BatchNorm statistics (recommended when frozen).
    """
    for name, param in model.named_parameters():
        if "head" not in name and "classifier" not in name and "fc" not in name:
            param.requires_grad = False

    if freeze_bn:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()  # freeze running stats


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
```

## Progressive Unfreezing

```python
def progressive_unfreeze(model: nn.Module, epoch: int,
                         unfreeze_schedule: dict[int, list[str]]):
    """Unfreeze layers according to a schedule.

    unfreeze_schedule: {epoch: [layer_name_prefixes_to_unfreeze]}
    Example: {0: ["head"], 5: ["layer4"], 10: ["layer3"], 15: ["layer2"]}

    Call at the start of each epoch.
    """
    if epoch in unfreeze_schedule:
        prefixes = unfreeze_schedule[epoch]
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in prefixes):
                param.requires_grad = True
        print(f"Epoch {epoch}: unfroze {prefixes}")
```

## Discriminative Learning Rates

```python
import torch.optim as optim

def create_param_groups(model: nn.Module, base_lr: float = 1e-4,
                        head_lr: float = 1e-3) -> list[dict]:
    """Different LR for backbone (pretrained) vs head (new).

    Backbone uses lower LR to preserve pretrained features.
    Head uses higher LR to learn task-specific mapping.
    """
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name or "classifier" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": head_lr},
    ]

# Usage
param_groups = create_param_groups(model, base_lr=1e-5, head_lr=1e-3)
optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
```

## Layer-Wise LR Decay

```python
def layerwise_lr_decay(model: nn.Module, base_lr: float = 1e-4,
                       decay_factor: float = 0.8,
                       head_lr: float = 1e-3) -> list[dict]:
    """Assign decreasing LR to deeper (earlier) layers.

    Layer closest to output gets base_lr, each layer back gets
    base_lr * decay_factor^depth.

    Works with models that have numbered layers (ResNet, ViT, etc.).
    """
    param_groups = []
    layer_names = []

    # Identify layer groups (adapt pattern to your model)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Extract layer depth from name (e.g., "layer3.1.conv2" -> 3)
        parts = name.split(".")
        if "head" in name or "classifier" in name or "fc" in name:
            param_groups.append({"params": [param], "lr": head_lr})
        else:
            # Deeper (lower index) layers get lower LR
            depth = 0
            for p in parts:
                if p.startswith("layer") or p.startswith("block"):
                    try:
                        depth = int("".join(filter(str.isdigit, p)))
                    except ValueError:
                        pass
                    break
            lr = base_lr * (decay_factor ** (4 - depth))  # adjust 4 to max depth
            param_groups.append({"params": [param], "lr": lr})

    return param_groups
```

## Loading Pretrained Weights

```python
import timm
import torchvision.models as tv_models

# ── From timm ───────────────────────────────────────────────────
model = timm.create_model("resnet50", pretrained=True, num_classes=2)

# ── From torchvision ────────────────────────────────────────────
weights = tv_models.ResNet50_Weights.DEFAULT
model = tv_models.resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# ── From HuggingFace ───────────────────────────────────────────
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/swin-base-patch4-window7-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True,
)
```

## Common Pitfalls

- **Not updating optimizer after unfreezing**: When you unfreeze new layers, you must add them to the optimizer or recreate it. Newly unfrozen parameters won't be updated otherwise.
- **BatchNorm in frozen backbone**: Even if parameters are frozen, BN running stats update during `model.train()`. Call `module.eval()` on frozen BN layers to prevent this.
- **Wrong requires_grad**: Setting `requires_grad = False` doesn't remove the parameter from the optimizer. You need to either filter parameters when creating the optimizer or use `param_groups` with `lr=0`.
- **Head initialization**: The new classification head is randomly initialized. It often helps to train only the head for a few epochs before unfreezing the backbone.
