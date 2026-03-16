---
name: data-preprocessing
description: >
  Load and preprocess biomedical data (DICOM, NIfTI, whole-slide images, tabular
  clinical data). Use when: (1) Loading DICOM/NIfTI/WSI files and extracting
  metadata, (2) Intensity normalization for CT/MRI, (3) Stain normalization for
  histopathology, (4) Building PyTorch Dataset classes with caching or HDF5,
  (5) Designing augmentation pipelines for medical imaging (2D or 3D).
---

# Data Preprocessing

## Workflow

Preprocessing biomedical data involves these steps:

1. **Identify data modality** -- determine the file format and imaging type
2. **Load and validate** -- read files, extract metadata, verify integrity
3. **Normalize** -- apply modality-appropriate intensity normalization
4. **Build dataset** -- create a PyTorch Dataset class for training
5. **Add augmentation** -- apply augmentation (training only)

## Decision Tree

Determine the data type first, then follow the appropriate path:

**What is the data format?**
- DICOM (.dcm) → See [medical-imaging-io.md](references/medical-imaging-io.md) for loading, then [normalization.md](references/normalization.md) for CT windowing or MRI z-score
- NIfTI (.nii/.nii.gz) → See [medical-imaging-io.md](references/medical-imaging-io.md) for loading + resampling
- Whole-slide images (.svs/.tiff/.ndpi) → See [medical-imaging-io.md](references/medical-imaging-io.md) for tile extraction
- PNG/TIFF patches → Load directly with PIL/numpy
- Tabular clinical data (.csv) → See [dataset-patterns.md](references/dataset-patterns.md) TabularDataset

**What normalization is needed?**
- CT imaging → CT windowing (soft tissue, lung, bone -- **ask user which anatomy**)
- MRI → Z-score normalization with foreground masking
- Histopathology → Stain normalization (Macenko or Vahadane)
- General → Percentile clipping + min-max

**What dataset pattern fits?**
- Dataset fits in RAM → CachedDataset
- Too large for RAM, need fast access → HDF5Dataset
- Whole-slide images → WSIPatchDataset (on-the-fly tile extraction)
- Tabular features → TabularDataset

**ASK the user** which data modality and format they are working with before selecting patterns. The choice of normalization, dataset class, and augmentation all depend on the modality.

## References

| File | Read When |
|------|-----------|
| [references/medical-imaging-io.md](references/medical-imaging-io.md) | Loading DICOM series, NIfTI volumes, or whole-slide images; extracting metadata and pixel spacing |
| [references/normalization.md](references/normalization.md) | Applying CT windowing, MRI z-score, stain normalization, or generic intensity normalization |
| [references/dataset-patterns.md](references/dataset-patterns.md) | Building PyTorch Dataset classes (basic, cached, HDF5, WSI patch-based, tabular) |
| [references/augmentation.md](references/augmentation.md) | Spatial and intensity augmentations for 2D/3D medical images; framework comparison (MONAI vs albumentations vs torchvision) |
