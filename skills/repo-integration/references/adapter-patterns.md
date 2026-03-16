# Adapter Patterns

## Contents

- Core Adapter Class
- Pattern 1: Feature Extractor
- Pattern 2: Pretrained Classifier/Segmenter
- Pattern 3: Preprocessing Pipeline
- Pattern 4: Batch Processing with Memory Management
- Checkpoint Loading Recipes
- File Organization
- Common Pitfalls


Reusable patterns for wrapping external repos into your project's interface.

## Core Adapter Class

All adapters follow the same structure — a thin wrapper that handles loading,
format conversion, and cleanup:

```python
from __future__ import annotations

import torch
import torch.nn as nn


class ExternalModelAdapter:
    """Adapter template. Subclass or modify for each external repo."""

    def __init__(self, weights_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = self._load_model(weights_path)
        self.model.eval()

    def _load_model(self, weights_path: str) -> nn.Module:
        """Load the external model. Override per repo."""
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference. Override if pre/post-processing is needed."""
        return self.model(x.to(self.device))

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
```

## Pattern 1: Feature Extractor

Extract embeddings from a pretrained model (most common use case):

```python
class FeatureExtractor(ExternalModelAdapter):
    """Extract feature vectors from a pretrained encoder.

    Usage:
        extractor = FeatureExtractor("weights.pth", layer="avgpool")
        features = extractor(batch)  # (B, D)
    """

    def __init__(self, weights_path: str, layer: str = "avgpool",
                 device: str = "cuda"):
        self.layer = layer
        self._features = None
        super().__init__(weights_path, device)

    def _load_model(self, weights_path: str) -> nn.Module:
        from vendor.REPO.models import build_model  # adapt import

        model = build_model()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        # Handle common key patterns
        if "state_dict" in state:
            state = state["state_dict"]
        if "model" in state:
            state = state["model"]
        # Strip "module." prefix from DataParallel checkpoints
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.to(self.device)

        # Register hook to capture intermediate features
        target = dict(model.named_modules())[self.layer]
        target.register_forward_hook(self._hook)
        return model

    def _hook(self, module, input, output):
        self._features = output

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns feature tensor of shape (B, D)."""
        self.model(x.to(self.device))
        feat = self._features
        if feat.dim() > 2:
            feat = feat.flatten(1)  # (B, C, H, W) → (B, C*H*W)
        return feat.cpu()
```

## Pattern 2: Pretrained Classifier/Segmenter

Use an external model for inference, converting between your data format and theirs:

```python
import numpy as np


class ExternalSegmenter(ExternalModelAdapter):
    """Wrap an external segmentation model.

    Handles format conversion: your numpy/DICOM → their expected input → your output format.
    """

    def __init__(self, weights_path: str, device: str = "cuda",
                 input_size: tuple[int, int] = (512, 512)):
        self.input_size = input_size
        super().__init__(weights_path, device)

    def _load_model(self, weights_path: str) -> nn.Module:
        from vendor.REPO.models import SegNet  # adapt import

        model = SegNet(num_classes=2)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        return model

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert from your format to the model's expected input.

        Adapt normalization, resizing, and channel ordering to match
        the external repo's preprocessing.
        """
        import torch.nn.functional as F

        # Example: numpy HxW or HxWxC → (1, C, H, W) tensor
        if image.ndim == 2:
            image = image[None, None, ...]  # (1, 1, H, W)
        elif image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))[None, ...]  # (1, C, H, W)
        x = torch.from_numpy(image).float()
        # Match the repo's normalization (check their data loader)
        x = x / 255.0
        x = F.interpolate(x, size=self.input_size, mode="bilinear",
                          align_corners=False)
        return x

    def _postprocess(self, output: torch.Tensor,
                     original_size: tuple[int, int]) -> np.ndarray:
        """Convert model output back to your format."""
        import torch.nn.functional as F

        mask = output.argmax(dim=1)  # (1, H, W)
        mask = F.interpolate(mask.unsqueeze(1).float(), size=original_size,
                             mode="nearest").squeeze().byte()
        return mask.cpu().numpy()

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Full pipeline: numpy in → numpy mask out."""
        original_size = image.shape[:2]
        x = self._preprocess(image)
        output = self.model(x.to(self.device))
        return self._postprocess(output, original_size)
```

## Pattern 3: Preprocessing Pipeline

Wrap an external repo's preprocessing (transforms, normalization, stain normalization):

```python
class ExternalPreprocessor:
    """Wrap external preprocessing as a callable.

    Does NOT require GPU or PyTorch — works with numpy/PIL.
    """

    def __init__(self, config: dict | None = None):
        from vendor.REPO.preprocessing import build_pipeline  # adapt import
        self.pipeline = build_pipeline(**(config or {}))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply the external preprocessing pipeline."""
        return self.pipeline(image)

    @classmethod
    def from_repo_defaults(cls) -> "ExternalPreprocessor":
        """Use the exact same config as the published paper."""
        return cls(config={
            # Copy from their default config or argparse defaults
        })
```

## Pattern 4: Batch Processing with Memory Management

Process a dataset through an external model efficiently:

```python
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def extract_features_batched(
    extractor: FeatureExtractor,
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> np.ndarray:
    """Extract features for an entire dataset.

    Returns array of shape (N, D).
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)
    all_features = []
    for batch in tqdm(loader, desc="Extracting features"):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]  # discard labels if present
        features = extractor(batch)  # (B, D)
        all_features.append(features.numpy())
    return np.concatenate(all_features, axis=0)
```

## Checkpoint Loading Recipes

Published repos use inconsistent checkpoint formats. Common patterns:

```python
def load_checkpoint_flexible(path: str, model: nn.Module) -> nn.Module:
    """Handle common checkpoint format variations."""
    state = torch.load(path, map_location="cpu", weights_only=True)

    # Unwrap nested dicts
    for key in ("state_dict", "model", "model_state_dict", "net"):
        if key in state:
            state = state[key]
            break

    # Strip DataParallel "module." prefix
    state = {k.removeprefix("module."): v for k, v in state.items()}

    # Strip DDP "_orig_mod." prefix (torch.compile)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

    # Load with relaxed strictness, report mismatches
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Unexpected keys: {result.unexpected_keys}")
    return model
```

## File Organization

Place adapters alongside vendored code or in a dedicated module:

```
src/<pkg>/
├── external/
│   ├── __init__.py
│   ├── repo_a_adapter.py     ← adapter for repo A
│   └── repo_b_adapter.py     ← adapter for repo B
├── vendor/                    ← vendored source files (if not pip-installed)
│   ├── repo_a/
│   │   ├── __init__.py
│   │   ├── models.py          ← copied from external repo
│   │   └── transforms.py
│   └── repo_b/
│       └── ...
```

## Common Pitfalls

- **Weight key mismatches**: Always inspect `state_dict.keys()` before loading. Print and compare if `strict=False` reports missing keys.
- **Input normalization mismatch**: The #1 cause of silently wrong results. Check the external repo's data loader for mean/std normalization, pixel value range (0-1 vs 0-255), and channel order (RGB vs BGR).
- **Eval mode**: Always call `model.eval()` before inference. BatchNorm and Dropout behave differently in training mode.
- **Half precision**: Some pretrained weights are in float16. Cast to float32 if you see numerical instability.
- **torch.compile**: Models saved with `torch.compile` have `_orig_mod.` prefix in state dict keys. Strip it when loading into a non-compiled model.
