# Inference Pipelines Reference

## Contents

- Sliding Window Inference (3D Volumes)
- Patch-Based WSI Inference
- Test-Time Augmentation (TTA)
- Memory-Efficient Inference
- Batch Inference CLI
- Common Pitfalls


**ASK the user** about their input data format and size -- sliding window is for 3D volumes, patch-based is for WSI, simple batching is for standard 2D images.

## Sliding Window Inference (3D Volumes)

```python
import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def sliding_window_inference(model: torch.nn.Module, volume: torch.Tensor,
                             window_size: tuple[int, int, int],
                             overlap: float = 0.5,
                             device: str = "cuda",
                             use_gaussian: bool = True) -> torch.Tensor:
    """Sliding window inference with overlap and Gaussian weighting.

    volume: (1, C, D, H, W) input tensor
    window_size: (d, h, w) patch size
    overlap: fraction of overlap between patches (0.5 = 50%)
    use_gaussian: weight predictions with Gaussian to reduce boundary artifacts

    Returns: (1, num_classes, D, H, W) prediction tensor
    """
    model.eval()
    _, C, D, H, W = volume.shape
    d, h, w = window_size
    step_d = max(1, int(d * (1 - overlap)))
    step_h = max(1, int(h * (1 - overlap)))
    step_w = max(1, int(w * (1 - overlap)))

    # Pad volume if needed
    pad_d = max(0, d - D)
    pad_h = max(0, h - H)
    pad_w = max(0, w - W)
    volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d))
    _, _, D_p, H_p, W_p = volume.shape

    # Create Gaussian importance map for blending
    if use_gaussian:
        importance = _gaussian_kernel_3d(window_size)
    else:
        importance = torch.ones(window_size)
    importance = importance.to(device)

    # Accumulators
    output_shape = None
    output_sum = None
    weight_sum = None

    for z in range(0, D_p - d + 1, step_d):
        for y in range(0, H_p - h + 1, step_h):
            for x in range(0, W_p - w + 1, step_w):
                patch = volume[:, :, z:z+d, y:y+h, x:x+w].to(device)
                pred = model(patch)  # (1, num_classes, d, h, w)

                if output_sum is None:
                    n_classes = pred.shape[1]
                    output_sum = torch.zeros(1, n_classes, D_p, H_p, W_p, device=device)
                    weight_sum = torch.zeros(1, 1, D_p, H_p, W_p, device=device)

                output_sum[:, :, z:z+d, y:y+h, x:x+w] += pred * importance
                weight_sum[:, :, z:z+d, y:y+h, x:x+w] += importance

    # Normalize
    result = output_sum / (weight_sum + 1e-8)
    # Remove padding
    result = result[:, :, :D, :H, :W]
    return result.cpu()


def _gaussian_kernel_3d(size: tuple[int, int, int],
                        sigma: float = 0.125) -> torch.Tensor:
    """Create a 3D Gaussian importance map."""
    coords = [torch.linspace(-1, 1, s) for s in size]
    grid = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)
    kernel = torch.exp(-torch.sum(grid ** 2, dim=-1) / (2 * sigma ** 2))
    return kernel
```

## Patch-Based WSI Inference

```python
import openslide
import numpy as np
import torch
from tqdm import tqdm

@torch.no_grad()
def predict_wsi(model: torch.nn.Module, slide_path: str,
                patch_size: int = 256, level: int = 0,
                batch_size: int = 32, device: str = "cuda",
                tissue_threshold: float = 0.5) -> dict:
    """Run patch-level inference on a whole-slide image.

    Returns dict with coordinates and predictions.
    """
    model.eval()
    slide = openslide.OpenSlide(slide_path)
    w, h = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]

    # Collect patches
    patches = []
    coords = []
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            x0, y0 = int(x * downsample), int(y * downsample)
            tile = np.array(slide.read_region((x0, y0), level,
                           (patch_size, patch_size)).convert("RGB"))
            if _tissue_fraction(tile) >= tissue_threshold:
                patches.append(tile)
                coords.append((x0, y0))

    # Batch inference
    all_preds = []
    for i in tqdm(range(0, len(patches), batch_size), desc="WSI inference"):
        batch = np.stack(patches[i:i+batch_size])
        batch = torch.from_numpy(batch).float().permute(0, 3, 1, 2) / 255.0
        preds = model(batch.to(device)).cpu()
        all_preds.append(preds)

    slide.close()
    return {
        "coordinates": coords,
        "predictions": torch.cat(all_preds) if all_preds else torch.tensor([]),
    }


def _tissue_fraction(tile: np.ndarray) -> float:
    gray = np.mean(tile, axis=2)
    return (gray < 220).mean()  # simple threshold for tissue
```

## Test-Time Augmentation (TTA)

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def predict_with_tta(model: torch.nn.Module, image: torch.Tensor,
                     device: str = "cuda") -> torch.Tensor:
    """Test-time augmentation: average predictions over flips and rotations.

    image: (1, C, H, W) tensor
    Returns: averaged prediction
    """
    model.eval()
    preds = []

    # Original
    preds.append(model(image.to(device)))

    # Horizontal flip
    preds.append(model(torch.flip(image, dims=[3]).to(device)))

    # Vertical flip
    preds.append(model(torch.flip(image, dims=[2]).to(device)))

    # 90-degree rotation
    rotated = torch.rot90(image, k=1, dims=[2, 3])
    preds.append(model(rotated.to(device)))

    # Average predictions
    avg_pred = torch.stack(preds).mean(dim=0)
    return avg_pred.cpu()


@torch.no_grad()
def predict_segmentation_with_tta(model: torch.nn.Module,
                                   image: torch.Tensor,
                                   device: str = "cuda") -> torch.Tensor:
    """TTA for segmentation -- must reverse spatial transforms on output masks.

    image: (1, C, H, W) tensor
    Returns: averaged soft prediction (1, num_classes, H, W)
    """
    model.eval()
    preds = []

    # Original
    preds.append(model(image.to(device)))

    # Horizontal flip: flip input, predict, flip output back
    flipped_h = torch.flip(image, dims=[3])
    pred_h = model(flipped_h.to(device))
    preds.append(torch.flip(pred_h, dims=[3]))

    # Vertical flip
    flipped_v = torch.flip(image, dims=[2])
    pred_v = model(flipped_v.to(device))
    preds.append(torch.flip(pred_v, dims=[2]))

    avg_pred = torch.stack(preds).mean(dim=0)
    return avg_pred.cpu()
```

## Memory-Efficient Inference

```python
import torch

@torch.no_grad()
def efficient_inference(model, inputs, device="cuda", half=True):
    """Memory-efficient inference with optional half precision."""
    model.eval()
    if half:
        model = model.half()
        inputs = inputs.half()
    return model(inputs.to(device)).cpu()
```

## Batch Inference CLI

```python
import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm

def batch_inference_cli():
    parser = argparse.ArgumentParser(description="Batch model inference")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--input-dir", required=True, help="Directory of input files")
    parser.add_argument("--output-dir", required=True, help="Directory for results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.model)
    model.to(args.device).eval()

    files = sorted(f for f in os.listdir(args.input_dir)
                   if f.endswith((".npy", ".npz", ".png")))

    results = {}
    for fname in tqdm(files, desc="Inference"):
        input_path = os.path.join(args.input_dir, fname)
        data = load_input(input_path)  # implement per-project
        with torch.no_grad():
            pred = model(data.unsqueeze(0).to(args.device))
        results[fname] = pred.cpu().numpy().tolist()

    with open(os.path.join(args.output_dir, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    batch_inference_cli()
```

## Common Pitfalls

- **Boundary artifacts**: Sliding window without overlap creates visible seams. Use at least 50% overlap + Gaussian weighting.
- **Forgetting `torch.no_grad()`**: Without it, computation graphs accumulate and cause OOM on large volumes/WSIs.
- **OOM on large volumes**: Process one patch at a time or reduce batch size. Don't load the full volume on GPU.
- **TTA for segmentation**: Must reverse spatial transforms on the output mask before averaging. Forgetting this produces blurred/wrong masks.
- **WSI coordinate systems**: OpenSlide's `read_region()` always takes level-0 coordinates, even when reading at lower levels.
