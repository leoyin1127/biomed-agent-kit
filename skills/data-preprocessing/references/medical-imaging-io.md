# Medical Imaging I/O Reference

## Contents

- DICOM Loading (pydicom)
- NIfTI Loading (nibabel)
- Whole-Slide Image Loading (OpenSlide)
- Common Pitfalls


**ASK the user** which imaging modality they are working with before selecting a loading pattern.

## DICOM Loading (pydicom)

### Single File

```python
import pydicom
import numpy as np

def load_dicom(path: str) -> tuple[np.ndarray, dict]:
    """Load a single DICOM file and extract metadata.

    Returns pixel array and metadata dict.
    """
    ds = pydicom.dcmread(path)

    # Apply rescale slope/intercept for correct HU values (CT)
    pixels = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        pixels = pixels * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    metadata = {
        "patient_id": str(getattr(ds, "PatientID", "")),
        "study_date": str(getattr(ds, "StudyDate", "")),
        "modality": str(getattr(ds, "Modality", "")),
        "pixel_spacing": [float(x) for x in ds.PixelSpacing] if hasattr(ds, "PixelSpacing") else None,
        "slice_thickness": float(ds.SliceThickness) if hasattr(ds, "SliceThickness") else None,
        "rows": ds.Rows,
        "columns": ds.Columns,
    }
    return pixels, metadata
```

### Loading a DICOM Series (3D Volume)

```python
import os
import pydicom
import numpy as np

def load_dicom_series(directory: str) -> tuple[np.ndarray, dict]:
    """Load all DICOM slices in a directory into a 3D volume.

    Slices are sorted by ImagePositionPatient or InstanceNumber.
    Returns volume (D, H, W) and metadata.
    """
    slices = []
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        try:
            ds = pydicom.dcmread(fpath)
            slices.append(ds)
        except Exception:
            continue  # skip non-DICOM files

    if not slices:
        raise ValueError(f"No DICOM files found in {directory}")

    # Sort by slice position
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except (AttributeError, IndexError):
        slices.sort(key=lambda s: int(s.InstanceNumber))

    # Build volume
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices])

    # Apply rescale
    ds0 = slices[0]
    if hasattr(ds0, "RescaleSlope") and hasattr(ds0, "RescaleIntercept"):
        volume = volume * float(ds0.RescaleSlope) + float(ds0.RescaleIntercept)

    # Compute spacing
    pixel_spacing = [float(x) for x in ds0.PixelSpacing] if hasattr(ds0, "PixelSpacing") else [1.0, 1.0]
    if len(slices) > 1:
        try:
            slice_spacing = abs(float(slices[1].ImagePositionPatient[2]) -
                                float(slices[0].ImagePositionPatient[2]))
        except (AttributeError, IndexError):
            slice_spacing = float(ds0.SliceThickness) if hasattr(ds0, "SliceThickness") else 1.0
    else:
        slice_spacing = float(ds0.SliceThickness) if hasattr(ds0, "SliceThickness") else 1.0

    metadata = {
        "patient_id": str(getattr(ds0, "PatientID", "")),
        "spacing": [slice_spacing, pixel_spacing[0], pixel_spacing[1]],  # (D, H, W)
        "shape": volume.shape,
        "n_slices": len(slices),
    }
    return volume, metadata
```

## NIfTI Loading (nibabel)

```python
import nibabel as nib
import numpy as np

def load_nifti(path: str) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Load a NIfTI file.

    Returns: (volume, affine_matrix, voxel_spacing)
    """
    img = nib.load(path)
    volume = img.get_fdata().astype(np.float32)
    affine = img.affine
    spacing = tuple(img.header.get_zooms()[:3])
    return volume, affine, spacing


def resample_to_isotropic(volume: np.ndarray, spacing: tuple,
                          target_spacing: float = 1.0) -> np.ndarray:
    """Resample volume to isotropic voxel spacing.

    Uses scipy for trilinear interpolation.
    """
    from scipy.ndimage import zoom

    scale_factors = tuple(s / target_spacing for s in spacing)
    resampled = zoom(volume, scale_factors, order=1)  # bilinear
    return resampled
```

## Whole-Slide Image Loading (OpenSlide)

```python
import openslide
import numpy as np
from PIL import Image

def open_slide(path: str) -> openslide.OpenSlide:
    """Open a whole-slide image (SVS, TIFF, NDPI, etc.)."""
    return openslide.OpenSlide(path)


def get_slide_info(slide: openslide.OpenSlide) -> dict:
    """Extract slide metadata."""
    return {
        "dimensions": slide.dimensions,  # (width, height) at level 0
        "level_count": slide.level_count,
        "level_dimensions": slide.level_dimensions,
        "level_downsamples": slide.level_downsamples,
        "mpp": float(slide.properties.get("openslide.mpp-x", 0)),  # microns per pixel
    }


def extract_tile(slide: openslide.OpenSlide, x: int, y: int,
                 size: int = 256, level: int = 0) -> np.ndarray:
    """Extract a tile at given coordinates.

    x, y: top-left corner in level-0 coordinates
    size: tile size in pixels at the target level
    level: magnification level (0 = highest resolution)
    """
    tile = slide.read_region((x, y), level, (size, size))
    return np.array(tile.convert("RGB"))


def extract_tiles_grid(slide: openslide.OpenSlide, tile_size: int = 256,
                       level: int = 0, overlap: int = 0,
                       tissue_threshold: float = 0.5) -> list[dict]:
    """Extract all tiles on a grid, filtering by tissue content.

    Returns list of {"x": int, "y": int, "tile": np.ndarray}.
    """
    w, h = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    step = tile_size - overlap
    tiles = []

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            # Coordinates in level-0 space
            x0 = int(x * downsample)
            y0 = int(y * downsample)
            tile = extract_tile(slide, x0, y0, tile_size, level)

            # Tissue detection via Otsu thresholding
            if _tissue_fraction(tile) >= tissue_threshold:
                tiles.append({"x": x0, "y": y0, "tile": tile})

    return tiles


def _tissue_fraction(tile: np.ndarray) -> float:
    """Estimate tissue fraction using Otsu thresholding on grayscale."""
    from skimage.filters import threshold_otsu

    gray = np.mean(tile, axis=2)  # RGB to grayscale
    if gray.std() < 5:  # near-uniform = background
        return 0.0
    try:
        thresh = threshold_otsu(gray)
        tissue_mask = gray < thresh  # tissue is darker than background
        return tissue_mask.mean()
    except ValueError:
        return 0.0
```

## Common Pitfalls

- **DICOM pixel value interpretation**: Raw pixel values are NOT Hounsfield Units. Always apply `RescaleSlope * pixel + RescaleIntercept`. Without this, CT windowing will be wrong.
- **DICOM series sorting**: Never rely on filename order. Sort by `ImagePositionPatient[2]` (z-coordinate) or `InstanceNumber`.
- **NIfTI orientation**: NIfTI uses RAS (Right-Anterior-Superior) by default. Some tools use LPS. Check with `nib.aff2axcodes(affine)` and reorient if needed with `nib.as_closest_canonical(img)`.
- **WSI coordinate systems**: `read_region()` always takes level-0 coordinates, even when reading at lower magnification levels. This is a common source of off-by-one-magnification bugs.
- **Memory**: A single WSI at 40x can be 100,000+ x 100,000+ pixels. Never load the full image into memory. Always work with tiles or lower magnification levels.
- **Multi-frame DICOM**: Enhanced DICOM stores all frames in a single file. Use `ds.pixel_array` which returns shape `(frames, H, W)` for multi-frame.
