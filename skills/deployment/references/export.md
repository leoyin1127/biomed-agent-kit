# Model Export Reference

**ASK the user** whether they need cross-framework compatibility (use ONNX) or PyTorch-only deployment (use TorchScript/torch.compile).

## ONNX Export

```python
import torch
import torch.nn as nn

def export_to_onnx(model: nn.Module, input_shape: tuple,
                   output_path: str, opset_version: int = 17,
                   dynamic_batch: bool = True):
    """Export a PyTorch model to ONNX format.

    input_shape: e.g. (1, 3, 224, 224) for a single image
    dynamic_batch: allow variable batch size at inference
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"ONNX model saved to {output_path}")
```

### Verifying ONNX Model

```python
import numpy as np
import onnxruntime as ort

def verify_onnx(onnx_path: str, pytorch_model: nn.Module,
                input_shape: tuple, atol: float = 1e-5):
    """Verify ONNX model produces same outputs as PyTorch."""
    pytorch_model.eval()
    dummy = torch.randn(*input_shape)

    # PyTorch output
    with torch.no_grad():
        pt_output = pytorch_model(dummy).numpy()

    # ONNX output
    session = ort.InferenceSession(onnx_path)
    ort_output = session.run(None, {"input": dummy.numpy()})[0]

    np.testing.assert_allclose(pt_output, ort_output, atol=atol)
    print(f"Verification passed (atol={atol})")


def onnx_inference(onnx_path: str, input_array: np.ndarray) -> np.ndarray:
    """Run inference with an ONNX model."""
    session = ort.InferenceSession(onnx_path,
                                   providers=["CUDAExecutionProvider",
                                              "CPUExecutionProvider"])
    return session.run(None, {"input": input_array})[0]
```

## TorchScript

### Tracing (Recommended for Most Models)

```python
def export_torchscript_trace(model: nn.Module, input_shape: tuple,
                             output_path: str):
    """Export via tracing -- records operations on example input.

    Use when: model has no data-dependent control flow (if/else on input values).
    """
    model.eval()
    dummy = torch.randn(*input_shape)
    traced = torch.jit.trace(model, dummy)
    traced.save(output_path)

# Load and run
loaded = torch.jit.load(output_path)
output = loaded(input_tensor)
```

### Scripting (For Models with Control Flow)

```python
def export_torchscript_script(model: nn.Module, output_path: str):
    """Export via scripting -- compiles Python to TorchScript IR.

    Use when: model has if/else or loops that depend on input.
    More restrictive: not all Python is supported.
    """
    model.eval()
    scripted = torch.jit.script(model)
    scripted.save(output_path)
```

## torch.compile (PyTorch 2.0+)

```python
import torch

# Compile for faster inference (no export needed)
model = torch.compile(model, mode="reduce-overhead")

# Modes:
# "default" - balanced compile time vs speedup
# "reduce-overhead" - minimize runtime overhead (best for inference)
# "max-autotune" - maximum optimization (longer compile time)
```

## Quantization

### Dynamic Quantization (CPU)

```python
import torch.quantization

def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Dynamic quantization for CPU inference.

    Quantizes weights to int8, activations are quantized on-the-fly.
    Best for: models with large linear layers (transformers, MLPs).
    """
    quantized = torch.quantization.quantize_dynamic(
        model.cpu(), {nn.Linear}, dtype=torch.qint8
    )
    return quantized
```

## Common Pitfalls

- **Dynamic control flow breaks tracing**: `if x.shape[0] > 1` in forward() will be traced as a constant. Use scripting instead.
- **Custom ops not in ONNX**: Some PyTorch ops don't have ONNX equivalents. Check the ONNX op set coverage.
- **Input shape assumptions**: Traced models may hardcode input shapes. Use `dynamic_axes` in ONNX export.
- **Model must be in eval mode**: Always call `model.eval()` before export. BatchNorm and Dropout in train mode will give wrong results.
- **torch.compile warmup**: First inference is slow (compilation). Subsequent calls are fast.
