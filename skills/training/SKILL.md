---
name: training
description: >
  Train and fine-tune biomedical ML models with production-grade patterns.
  Use when: (1) Writing training loops with early stopping and LR scheduling,
  (2) Transfer learning and fine-tuning pretrained models (freeze/unfreeze,
  discriminative LR), (3) Mixed precision training (fp16/bf16),
  (4) Gradient accumulation for large effective batch sizes, (5) Checkpoint
  management (save/load best model, resume training).
---

# Training

## Workflow

Training a biomedical ML model involves these steps:

1. **Choose training strategy** -- from scratch, fine-tune, or frozen feature extraction
2. **Configure the training loop** -- loss, optimizer, LR scheduler, early stopping
3. **Set up mixed precision** -- fp16 or bf16 based on GPU hardware
4. **Run training** -- with checkpointing and metric logging
5. **Evaluate** -- on held-out validation set each epoch

## Decision Tree

**Is there a pretrained model available?**
- Yes, dataset is small (<1K samples) → Freeze backbone, train head only. See [transfer-learning.md](references/transfer-learning.md)
- Yes, dataset is medium (1K-10K) → Fine-tune with discriminative LR. See [transfer-learning.md](references/transfer-learning.md)
- Yes, dataset is large (>10K) → Fine-tune all layers or train from scratch
- No → Train from scratch with full training loop

**What LR scheduler?**
- Research / general use → `ReduceLROnPlateau` (adapts to val metric)
- Fine-tuning with warm start → Warmup + Cosine decay
- Short aggressive training → `OneCycleLR`

**Mixed precision?**
- Ampere+ GPU (A100, RTX 3090+) → bf16 (no GradScaler needed)
- Older GPU → fp16 + GradScaler
- CPU only → skip AMP

**ASK the user** before starting:
- Are they training from scratch or fine-tuning? This determines the entire strategy.
- What metric to monitor for early stopping (and minimize vs maximize)?
- What GPU hardware is available? This determines mixed precision strategy.

## References

| File | Read When |
|------|-----------|
| [references/training-loop.md](references/training-loop.md) | Writing training/evaluation loops, early stopping, LR scheduling, gradient accumulation, checkpoint save/load |
| [references/transfer-learning.md](references/transfer-learning.md) | Fine-tuning pretrained models: freeze/unfreeze, progressive unfreezing, discriminative LR, loading weights from timm/torchvision/HuggingFace |
| [references/mixed-precision.md](references/mixed-precision.md) | AMP with fp16 or bf16, GradScaler, checking GPU bf16 support, common numerical pitfalls |
