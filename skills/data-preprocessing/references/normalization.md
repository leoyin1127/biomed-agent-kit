# Normalization Reference

**ASK the user** about their imaging modality and clinical task -- CT windowing parameters depend on the anatomy of interest.

## CT Windowing

Apply a window (center, width) to map HU values to display range:

```python
import numpy as np

def ct_window(volume: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply CT windowing and rescale to [0, 1].

    volume: array in Hounsfield Units (after RescaleSlope/Intercept)
    center: window center (HU)
    width: window width (HU)
    """
    lower = center - width / 2
    upper = center + width / 2
    windowed = np.clip(volume, lower, upper)
    return (windowed - lower) / (upper - lower)
```

### Common CT Windows

| Anatomy | Center (HU) | Width (HU) |
|---------|-------------|------------|
| Soft tissue | 40 | 400 |
| Lung | -600 | 1500 |
| Bone | 400 | 1800 |
| Brain | 40 | 80 |
| Liver | 60 | 150 |
| Mediastinum | 50 | 350 |
| Abdomen | 40 | 350 |

**ASK the user** what anatomy they're studying -- using the wrong window makes pathology invisible.

## MRI Z-Score Normalization

MRI intensity values are arbitrary (no physical units). Normalize per volume:

```python
import numpy as np

def zscore_normalize(volume: np.ndarray,
                     mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score normalization, optionally using foreground mask.

    mask: binary mask of foreground voxels (excludes air/background).
          If None, uses non-zero voxels as foreground.
    """
    if mask is None:
        mask = volume > 0  # simple foreground: non-zero voxels

    foreground = volume[mask]
    mean = foreground.mean()
    std = foreground.std()

    if std < 1e-8:
        return volume - mean  # avoid division by zero

    normalized = (volume - mean) / std
    return normalized


def percentile_normalize(volume: np.ndarray,
                         lower_pct: float = 1.0,
                         upper_pct: float = 99.0) -> np.ndarray:
    """Percentile-based normalization to [0, 1].

    Clips outliers and rescales. Robust to intensity artifacts.
    """
    p_low = np.percentile(volume, lower_pct)
    p_high = np.percentile(volume, upper_pct)
    clipped = np.clip(volume, p_low, p_high)
    return (clipped - p_low) / (p_high - p_low + 1e-8)
```

## Histopathology Stain Normalization

### Macenko Method (via torchstain)

Normalizes H&E staining variation across slides from different scanners/labs:

```python
import torch
from torchstain import MacenkoNormalizer
from torchvision import transforms
from PIL import Image

def create_stain_normalizer(reference_image_path: str) -> MacenkoNormalizer:
    """Fit a Macenko normalizer to a reference image.

    Choose a representative, well-stained image as reference.
    """
    normalizer = MacenkoNormalizer(backend="torch")
    ref_img = Image.open(reference_image_path).convert("RGB")
    ref_tensor = transforms.ToTensor()(ref_img) * 255  # torchstain expects [0, 255]
    normalizer.fit(ref_tensor)
    return normalizer


def normalize_stain(normalizer: MacenkoNormalizer,
                    image: np.ndarray) -> np.ndarray:
    """Normalize staining of an H&E image.

    image: RGB numpy array (H, W, 3), values in [0, 255]
    """
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    try:
        result, _, _ = normalizer.normalize(tensor)
        return result.permute(1, 2, 0).numpy().astype(np.uint8)
    except Exception:
        # Some tiles (mostly background) fail normalization
        return image
```

### Alternative: StainTools

```python
# pip install staintools
import staintools

def normalize_with_staintools(reference_path: str, target_image: np.ndarray) -> np.ndarray:
    """Stain normalization using StainTools (Vahadane method)."""
    normalizer = staintools.StainNormalizer(method="vahadane")
    reference = staintools.read_image(reference_path)
    reference = staintools.LuminosityStandardizer.standardize(reference)
    normalizer.fit(reference)

    target = staintools.LuminosityStandardizer.standardize(target_image)
    return normalizer.transform(target)
```

## Generic Intensity Normalization

```python
import numpy as np

def minmax_normalize(array: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 1]."""
    amin, amax = array.min(), array.max()
    if amax - amin < 1e-8:
        return np.zeros_like(array)
    return (array - amin) / (amax - amin)
```

## Common Pitfalls

- **CT without windowing**: Raw HU range (~-1000 to +3000) has very low contrast. Always apply a task-specific window.
- **MRI normalization scope**: Normalize per-volume, not per-dataset. MRI intensities vary between scans even for the same patient.
- **Stain normalization failures**: Background tiles and poorly stained regions can cause Macenko to fail. Always handle exceptions gracefully.
- **Integer vs float**: DICOM pixel data is often uint16. Convert to float32 before arithmetic to avoid overflow.
- **Normalization leakage**: When using dataset-level statistics (global mean/std), compute them on the training set only.
