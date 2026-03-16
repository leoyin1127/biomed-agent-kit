# Augmentation Reference

**ASK the user** which augmentation framework they prefer (MONAI, albumentations, torchvision, or custom) and their data dimensionality (2D vs 3D) before writing augmentation code.

## Spatial Augmentations

```python
import numpy as np
from scipy.ndimage import affine_transform, map_coordinates

def random_affine_2d(image: np.ndarray, max_rotation: float = 15.0,
                     max_scale: float = 0.1, max_shear: float = 0.1,
                     seed: int | None = None) -> np.ndarray:
    """Random affine transformation for 2D images.

    image: (H, W) or (H, W, C)
    """
    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]
    center = np.array([h / 2, w / 2])

    angle = rng.uniform(-max_rotation, max_rotation) * np.pi / 180
    scale = 1 + rng.uniform(-max_scale, max_scale)
    shear = rng.uniform(-max_shear, max_shear)

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    scale_mat = np.diag([scale, scale])
    shear_mat = np.array([[1, shear], [0, 1]])

    transform = rotation @ scale_mat @ shear_mat
    offset = center - transform @ center

    if image.ndim == 3:
        result = np.stack([
            affine_transform(image[:, :, c], transform, offset=offset, order=1)
            for c in range(image.shape[2])
        ], axis=2)
    else:
        result = affine_transform(image, transform, offset=offset, order=1)

    return result


def elastic_deformation(image: np.ndarray, alpha: float = 100,
                        sigma: float = 10, seed: int | None = None) -> np.ndarray:
    """Elastic deformation for 2D images.

    Common in medical imaging -- simulates tissue deformation.
    alpha: deformation intensity
    sigma: smoothness of deformation field
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.RandomState(seed)
    h, w = image.shape[:2]

    dx = gaussian_filter(rng.randn(h, w) * alpha, sigma)
    dy = gaussian_filter(rng.randn(h, w) * alpha, sigma)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [y + dy, x + dx]

    if image.ndim == 3:
        result = np.stack([
            map_coordinates(image[:, :, c], coords, order=1, mode="reflect")
            for c in range(image.shape[2])
        ], axis=2)
    else:
        result = map_coordinates(image, coords, order=1, mode="reflect")

    return result
```

## Intensity Augmentations

```python
import numpy as np

def random_brightness_contrast(image: np.ndarray,
                                brightness_range: float = 0.1,
                                contrast_range: float = 0.1,
                                seed: int | None = None) -> np.ndarray:
    """Random brightness and contrast adjustment."""
    rng = np.random.RandomState(seed)
    brightness = rng.uniform(-brightness_range, brightness_range)
    contrast = rng.uniform(1 - contrast_range, 1 + contrast_range)
    return np.clip(image * contrast + brightness, 0, 1)


def random_gaussian_noise(image: np.ndarray, std_range: tuple = (0.01, 0.05),
                          seed: int | None = None) -> np.ndarray:
    """Add random Gaussian noise."""
    rng = np.random.RandomState(seed)
    std = rng.uniform(*std_range)
    noise = rng.randn(*image.shape).astype(np.float32) * std
    return np.clip(image + noise, 0, 1)


def random_gaussian_blur(image: np.ndarray, sigma_range: tuple = (0.5, 1.5),
                         seed: int | None = None) -> np.ndarray:
    """Apply random Gaussian blur."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.RandomState(seed)
    sigma = rng.uniform(*sigma_range)
    if image.ndim == 3:
        return np.stack([gaussian_filter(image[:, :, c], sigma)
                         for c in range(image.shape[2])], axis=2)
    return gaussian_filter(image, sigma)
```

## Domain-Specific Augmentations

### Histopathology: HED Color Augmentation

```python
import numpy as np
from skimage.color import rgb2hed, hed2rgb

def hed_color_augmentation(image: np.ndarray, sigma: float = 0.05,
                           seed: int | None = None) -> np.ndarray:
    """Augment H&E stained images in HED color space.

    Perturbs Hematoxylin, Eosin, and DAB channels independently.
    More biologically meaningful than RGB color jitter.
    image: RGB float array in [0, 1]
    """
    rng = np.random.RandomState(seed)
    hed = rgb2hed(image)

    # Perturb each channel independently
    for i in range(3):
        hed[:, :, i] *= 1 + rng.uniform(-sigma, sigma)
        hed[:, :, i] += rng.uniform(-sigma, sigma)

    augmented = hed2rgb(hed)
    return np.clip(augmented, 0, 1).astype(np.float32)
```

### Radiology: Random Intensity Shift

```python
def random_intensity_shift(volume: np.ndarray, max_shift: float = 0.1,
                           max_scale: float = 0.1,
                           seed: int | None = None) -> np.ndarray:
    """Random intensity shift and scale for CT/MRI volumes.

    Simulates scanner variation and contrast differences.
    """
    rng = np.random.RandomState(seed)
    shift = rng.uniform(-max_shift, max_shift)
    scale = rng.uniform(1 - max_scale, 1 + max_scale)
    return np.clip(volume * scale + shift, 0, 1)
```

## 3D Augmentations

```python
import numpy as np
from scipy.ndimage import rotate, zoom

def random_rotate_3d(volume: np.ndarray, max_angle: float = 15.0,
                     axes: tuple = (0, 1), seed: int | None = None) -> np.ndarray:
    """Random rotation in 3D around specified axes.

    volume: (D, H, W) or (D, H, W, C)
    axes: plane of rotation, e.g. (0,1) for axial, (0,2) for coronal
    """
    rng = np.random.RandomState(seed)
    angle = rng.uniform(-max_angle, max_angle)
    return rotate(volume, angle, axes=axes, reshape=False, order=1, mode="nearest")


def random_anisotropic_scale(volume: np.ndarray,
                              scale_range: tuple = (0.9, 1.1),
                              seed: int | None = None) -> np.ndarray:
    """Random anisotropic scaling for 3D volumes.

    Simulates variation in voxel spacing across scanners.
    """
    rng = np.random.RandomState(seed)
    scales = [rng.uniform(*scale_range) for _ in range(3)]
    if volume.ndim == 4:
        scales.append(1.0)  # don't scale channel dim
    return zoom(volume, scales, order=1, mode="nearest")
```

## Framework Comparison

| Framework | Best For | 2D | 3D | GPU |
|-----------|----------|----|----|-----|
| **MONAI** | Medical imaging (2D/3D), dictionary transforms | Yes | Yes | Yes |
| **albumentations** | Fast 2D augmentations, bounding box support | Yes | No | No |
| **torchvision** | Standard CV transforms, simple pipelines | Yes | No | No |
| **Custom (scipy)** | Full control, no extra dependencies | Yes | Yes | No |

### MONAI Example

```python
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandAffined,
    RandGaussianNoised, RandGaussianSmoothd, RandScaleIntensityd,
)

train_transforms = Compose([
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandAffined(keys=["image", "label"], prob=0.5,
                rotate_range=(0.26,), scale_range=(0.1,),
                mode=("bilinear", "nearest")),
    RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
])
```

### albumentations Example

```python
import albumentations as A

train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ElasticTransform(alpha=100, sigma=10, p=0.3),
    A.GaussNoise(var_limit=(0.001, 0.01), p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
])

# Usage: result = train_transforms(image=image)["image"]
```

## Key Rules

- **Never augment validation/test data** (except for TTA at inference -- see deployment skill)
- **Apply augmentation after splitting** -- augmented copies of the same sample must never leak across splits
- **For paired data (image + mask)**: apply the same spatial transform to both; intensity transforms to image only
- **Augmentation probability**: not every sample needs every augmentation. Use `p=0.3-0.5` for most transforms
