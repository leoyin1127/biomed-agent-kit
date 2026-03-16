---
name: deployment
description: >
  Deploy and optimize trained biomedical ML models for inference. Use when:
  (1) Exporting models to ONNX or TorchScript, (2) Building inference pipelines
  with sliding window for 3D volumes or patch-based prediction for WSI,
  (3) Test-time augmentation (TTA), (4) Optimizing inference speed with
  torch.compile or quantization, (5) Packaging models as REST APIs or Docker
  containers for serving.
---

# Deployment & Inference

## Workflow

Deploying a trained model involves these steps:

1. **Choose deployment target** -- batch processing, REST API, or edge device
2. **Export the model** -- to ONNX, TorchScript, or use torch.compile
3. **Build the inference pipeline** -- with appropriate tiling/windowing strategy
4. **Optimize** -- quantization, half precision, or compilation
5. **Package** -- Docker container, FastAPI server, or batch CLI

## Decision Tree

**What is the deployment target?**
- Local batch processing → Batch inference CLI. See [serving.md](references/serving.md)
- REST API → FastAPI server + Docker. See [serving.md](references/serving.md)
- Cross-framework (e.g., to C++/ONNX Runtime) → ONNX export. See [export.md](references/export.md)
- PyTorch-only, speed optimization → torch.compile or TorchScript. See [export.md](references/export.md)

**What is the input format?**
- 3D volumes (CT/MRI) → Sliding window inference with Gaussian blending. See [inference-pipelines.md](references/inference-pipelines.md)
- Whole-slide images → Patch-based inference with tissue detection. See [inference-pipelines.md](references/inference-pipelines.md)
- Standard 2D images → Simple batched inference
- Want uncertainty estimates → Test-time augmentation (TTA). See [inference-pipelines.md](references/inference-pipelines.md)

**ASK the user** before starting:
- What is the deployment target (local, API, edge)?
- What input data format and typical sizes?
- Any latency or throughput requirements?

## References

| File | Read When |
|------|-----------|
| [references/export.md](references/export.md) | Exporting to ONNX (with dynamic axes, verification), TorchScript (tracing vs scripting), torch.compile, quantization |
| [references/inference-pipelines.md](references/inference-pipelines.md) | Sliding window for 3D volumes (overlap + Gaussian), patch-based WSI inference, TTA for classification and segmentation |
| [references/serving.md](references/serving.md) | FastAPI model server, health checks, Dockerfile with CUDA, batch inference CLI |
