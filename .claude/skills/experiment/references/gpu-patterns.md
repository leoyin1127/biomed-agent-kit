# GPU Management Patterns

## Memory Cleanup Between Models

Always free GPU memory when switching between models to prevent OOM:

```python
import gc
import torch

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# Usage: between model runs in a loop
for model_name in models:
    encoder = load_model(model_name)
    results = run_inference(encoder, data)
    save_results(results)
    del encoder
    cleanup_gpu()
```

## Multi-GPU with ProcessPoolExecutor

Pin each worker to a specific GPU using `CUDA_VISIBLE_DEVICES`:

```python
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def gpu_init(gpu_id: int):
    """Worker initializer: pin to a single GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)  # "cuda:0" now maps to gpu_id

def run_on_gpu(experiment):
    """Runs in worker process, sees only its pinned GPU."""
    device = torch.device("cuda:0")
    # ... training/inference logic ...

# One pool per GPU, each with N workers
ctx = multiprocessing.get_context("spawn")
for gpu_id in gpu_ids:
    pool = ProcessPoolExecutor(
        max_workers=workers_per_gpu,
        mp_context=ctx,
        initializer=gpu_init,
        initargs=(gpu_id,),
    )
```

## VRAM-Based Worker Scaling

Auto-detect how many concurrent experiments a GPU can handle:

```python
def detect_experiments_per_gpu(gpu_ids: list[int], mem_per_exp_gb: float = 3.0) -> int:
    min_free = float("inf")
    for gid in gpu_ids:
        free, _ = torch.cuda.mem_get_info(gid)
        min_free = min(min_free, free)
    return max(1, int(min_free / (mem_per_exp_gb * 1024**3)))
```

## OOM Protection

Wrap GPU operations with OOM retry and reduced batch fallback:

```python
def safe_train(experiment, device):
    try:
        return train(experiment, device)
    except torch.cuda.OutOfMemoryError:
        cleanup_gpu()
        # Log failure, don't retry with same params
        return None
```

## Reproducibility Seeds

Set all seeds before any stochastic operation:

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

## Multi-GPU with a Single Model

Prefer `DistributedDataParallel` (DDP) over `DataParallel` — DDP uses one process
per GPU, avoids the GIL bottleneck, and scales better:

```python
# ── DDP (preferred) ──────────────────────────────────────────────
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Launch with: torchrun --nproc_per_node=NUM_GPUS script.py
```

`DataParallel` is acceptable for quick prototyping or inference only:

```python
# ── DataParallel (simple but slower) ─────────────────────────────
if torch.cuda.device_count() > 1 and use_all_gpus:
    model = torch.nn.DataParallel(model)
model = model.to(device)
```
