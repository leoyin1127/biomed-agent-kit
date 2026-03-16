# Hub Integrations

## Contents

- HuggingFace Hub
- MONAI (Medical Open Network for AI)
- timm (PyTorch Image Models)
- Common Pitfalls


Patterns for loading models from popular model hubs in biomedical ML.

## HuggingFace Hub

### Loading Biomedical Models

```python
from transformers import AutoModel, AutoTokenizer

# ── Biomedical Text Models ──────────────────────────────────────
# PubMedBERT
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
model = AutoModel.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)

# ClinicalBERT
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
```

### Feature Extraction from HuggingFace Models

```python
import torch
from transformers import AutoModel, AutoTokenizer

class HuggingFaceFeatureExtractor:
    """Extract embeddings from any HuggingFace model."""

    def __init__(self, model_name: str, device: str = "cuda",
                 pooling: str = "cls"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.pooling = pooling

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 512) -> torch.Tensor:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            return hidden[:, 0, :]
        elif self.pooling == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1)
            return (hidden * mask).sum(1) / mask.sum(1)
        raise ValueError(f"Unknown pooling: {self.pooling}")

    def cleanup(self):
        del self.model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
```

### Downloading Weights

```python
from huggingface_hub import hf_hub_download, snapshot_download

weights_path = hf_hub_download(
    repo_id="username/model-name",
    filename="pytorch_model.bin",
    cache_dir="./weights",
)

model_dir = snapshot_download(
    repo_id="username/model-name",
    cache_dir="./weights",
    ignore_patterns=["*.md", "*.txt"],
)
```

## MONAI (Medical Open Network for AI)

### Pretrained Models

```python
import torch
from monai.networks.nets import DenseNet121, SwinUNETR, UNet

# 3D Classification
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2, pretrained=True)

# 3D Segmentation (SwinUNETR)
model = SwinUNETR(img_size=(96, 96, 96), in_channels=1, out_channels=14, feature_size=48)
weight_url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
state = torch.hub.load_state_dict_from_url(weight_url, map_location="cpu")
model.load_state_dict(state["state_dict"], strict=False)

# Simple 3D UNet
model = UNet(spatial_dims=3, in_channels=1, out_channels=2,
             channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2))
```

### MONAI Transforms

```python
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped,
)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0),
             mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250,
                         b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=4),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    EnsureTyped(keys=["image", "label"]),
])
```

### MONAI Bundle API

```python
from monai.bundle import download, load

bundle_dir = download(name="spleen_ct_segmentation", bundle_dir="./bundles")
model = load(name="spleen_ct_segmentation", bundle_dir="./bundles", source="ngc")
```

## timm (PyTorch Image Models)

```python
import timm

models = timm.list_models("*vit*", pretrained=True)
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
# num_classes=0 -> feature extractor (removes classification head)

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
```

## Common Pitfalls

- **HuggingFace cache**: Models cached in `~/.cache/huggingface/`. Set `HF_HOME` to control location on shared clusters.
- **MONAI version pinning**: Bundles may require specific MONAI versions. Check `configs/metadata.json`.
- **Model card**: Always check the model card for training data, intended use, and limitations.

**ASK the user** whether their model is on HuggingFace Hub, MONAI Model Zoo, timm, or a standalone GitHub repo.
