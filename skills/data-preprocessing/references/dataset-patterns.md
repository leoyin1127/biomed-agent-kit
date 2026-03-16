# Dataset Patterns Reference

## Contents

- Basic Medical Imaging Dataset
- Cached Dataset (In-Memory)
- HDF5-Backed Dataset
- Patch-Based WSI Dataset
- Tabular Clinical Data Dataset
- Common Pitfalls


**ASK the user** about their dataset size and available RAM before recommending a caching strategy.

## Basic Medical Imaging Dataset

```python
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MedicalImageDataset(Dataset):
    """Basic dataset that loads images from disk on each access.

    Use when: dataset is too large to fit in RAM, or quick prototyping.
    """

    def __init__(self, image_paths: list[str], labels: list[int],
                 transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = np.load(self.image_paths[idx]).astype(np.float32)

        if self.transform:
            image = self.transform(image)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Ensure channel dimension: (H, W) -> (1, H, W)
        if image.dim() == 2:
            image = image.unsqueeze(0)

        return image, self.labels[idx]
```

## Cached Dataset (In-Memory)

```python
from functools import lru_cache

class CachedDataset(Dataset):
    """Cache loaded images in memory after first access.

    Use when: dataset fits in RAM and I/O is the bottleneck.
    Warning: with num_workers > 0, each worker has its own cache copy.
    For shared caching, use HDF5 or memory-mapped arrays.
    """

    def __init__(self, image_paths: list[str], labels: list[int],
                 transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    @lru_cache(maxsize=None)
    def _load(self, idx: int) -> np.ndarray:
        return np.load(self.image_paths[idx]).astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = self._load(idx).copy()  # copy to avoid mutation issues
        if self.transform:
            image = self.transform(image)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        return image, self.labels[idx]
```

## HDF5-Backed Dataset

```python
import h5py
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    """Dataset backed by an HDF5 file.

    Use when: dataset is too large for RAM but you want fast random access.
    HDF5 supports compression and chunked storage.

    HDF5 file structure:
        /images  -> (N, H, W) or (N, C, H, W)
        /labels  -> (N,)
    """

    def __init__(self, hdf5_path: str, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        # Read length without keeping file open (important for multiprocessing)
        with h5py.File(hdf5_path, "r") as f:
            self.length = len(f["labels"])
        self._file = None

    def _open(self):
        """Lazy-open file handle (one per worker process)."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        self._open()
        image = self._file["images"][idx].astype(np.float32)
        label = int(self._file["labels"][idx])

        if self.transform:
            image = self.transform(image)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if image.dim() == 2:
            image = image.unsqueeze(0)
        return image, label

    def __del__(self):
        if self._file is not None:
            self._file.close()


def create_hdf5(image_arrays: list[np.ndarray], labels: list[int],
                output_path: str, compression: str = "gzip"):
    """Create an HDF5 file from numpy arrays."""
    with h5py.File(output_path, "w") as f:
        f.create_dataset("images", data=np.stack(image_arrays),
                         compression=compression, chunks=True)
        f.create_dataset("labels", data=np.array(labels))
```

## Patch-Based WSI Dataset

```python
import openslide
import numpy as np
import torch
from torch.utils.data import Dataset

class WSIPatchDataset(Dataset):
    """On-the-fly patch extraction from whole-slide images.

    Use when: working with whole-slide images where pre-extracting
    all patches would require too much disk space.
    """

    def __init__(self, slide_path: str, coordinates: list[tuple[int, int]],
                 labels: list[int], patch_size: int = 256, level: int = 0,
                 transform=None):
        self.slide_path = slide_path
        self.coordinates = coordinates  # (x, y) in level-0 space
        self.labels = labels
        self.patch_size = patch_size
        self.level = level
        self.transform = transform
        self._slide = None

    def _open_slide(self):
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.slide_path)

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        self._open_slide()
        x, y = self.coordinates[idx]
        tile = self._slide.read_region(
            (x, y), self.level, (self.patch_size, self.patch_size)
        )
        image = np.array(tile.convert("RGB")).astype(np.float32) / 255.0

        if self.transform:
            image = self.transform(image)
        if isinstance(image, np.ndarray):
            # HWC -> CHW
            image = torch.from_numpy(image).permute(2, 0, 1)
        return image, self.labels[idx]
```

## Tabular Clinical Data Dataset

```python
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    """Dataset for tabular clinical data (e.g., lab values, demographics).

    Use when: features are structured/tabular, not images.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list[str],
                 label_col: str):
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df[label_col].values

        # Handle missing values
        col_means = np.nanmean(self.features, axis=0)
        nan_mask = np.isnan(self.features)
        self.features[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.features[idx]), int(self.labels[idx])
```

## Common Pitfalls

- **HDF5 + multiprocess DataLoader**: Don't open the HDF5 file in `__init__`. Open it lazily in `__getitem__` so each worker gets its own file handle. Otherwise you get corrupted reads.
- **Worker memory duplication**: With `num_workers > 0`, each worker copies the dataset object. If you cache data in a Python list, memory usage multiplies by `num_workers`. Use memory-mapped files or HDF5 instead.
- **Forgetting `.copy()`**: When caching numpy arrays and applying in-place transforms, always `.copy()` before transforming to avoid corrupting the cache.
- **Channel ordering**: PyTorch expects (C, H, W), but most image libraries return (H, W, C). Always convert.
- **Label dtype**: Ensure labels are Python `int` or `torch.long` for `CrossEntropyLoss`. Float labels cause cryptic errors.
