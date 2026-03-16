# Model Serving Reference

**ASK the user** about their serving requirements (single request vs batch, GPU vs CPU, authentication needs) before designing the API.

## FastAPI Model Server

```python
import io
import numpy as np
import torch
import torch.nn as nn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel

# Global model reference
model: nn.Module | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model
    model = load_model("weights/best_model.pt")  # implement per-project
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    yield
    del model
    torch.cuda.empty_cache()


app = FastAPI(title="Model API", lifespan=lifespan)


class PredictionResponse(BaseModel):
    prediction: list[float]
    label: str
    confidence: float


@app.get("/health")
async def health():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile):
    """Run inference on an uploaded image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = load_image_from_bytes(contents)  # implement per-project
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        tensor = preprocess(image).unsqueeze(0).to(device)  # implement
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    label_map = {0: "negative", 1: "positive"}  # adapt per-project

    return PredictionResponse(
        prediction=probs.tolist(),
        label=label_map.get(pred_idx, str(pred_idx)),
        confidence=float(probs[pred_idx]),
    )
```

Run with: `uvicorn serve:app --host 0.0.0.0 --port 8000`

## Dockerfile

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Copy application
COPY src/ src/
COPY weights/ weights/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t model-server .
docker run --gpus all -p 8000:8000 model-server
```

## Common Pitfalls

- **Loading model per request**: Load the model ONCE at startup (in `lifespan`), not in each request handler. Model loading takes seconds; inference takes milliseconds.
- **Multi-worker GPU conflicts**: With multiple uvicorn workers, each loads a copy of the model. Either use 1 worker with async, or manage GPU memory carefully.
- **Missing error handling**: Always validate input format and catch model errors. Return 400 for bad input, 500 for model errors.
- **No health check**: Always include `/health` for load balancers and orchestrators.
- **Large file uploads**: Set upload limits. Medical images can be very large (WSIs = GBs). Stream or reject oversized files.
