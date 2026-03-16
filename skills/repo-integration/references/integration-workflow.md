# Integration Workflow

## Contents

- Step 1: Evaluate the Repo
- Step 2: Choose Install Strategy
- Step 3: Dependency Conflict Resolution
- Step 4: Test the Integration
- Common Pitfalls

## Step 1: Evaluate the Repo

Before writing any code, check these in order. Stop early if a blocker is found.

### 1a. License Compatibility

```bash
# Check license file
gh repo view OWNER/REPO --json licenseInfo --jq '.licenseInfo.name'
```

| License | Can use in research? | Can redistribute? | Notes |
|---------|---------------------|-------------------|-------|
| MIT, BSD, Apache 2.0 | Yes | Yes | Permissive — safe |
| GPL v3 | Yes | Derivative must be GPL | Problematic if your code is proprietary |
| No license | Legally unclear | No | Contact authors or avoid |

### 1b. Python and CUDA Compatibility

```bash
# Clone and inspect requirements
gh repo clone OWNER/REPO /tmp/REPO --depth 1

# Check Python version
cat /tmp/REPO/setup.py /tmp/REPO/setup.cfg /tmp/REPO/pyproject.toml 2>/dev/null | grep -i python

# Check PyTorch/CUDA version
grep -r "torch" /tmp/REPO/requirements*.txt /tmp/REPO/setup.py /tmp/REPO/pyproject.toml 2>/dev/null
```

Common conflicts in biomedical repos:
- **PyTorch version**: repo pins `torch==1.x` but your project uses `torch>=2.0`
- **CUDA version**: repo requires CUDA 11.x but your system has 12.x
- **Python version**: repo uses 3.7/3.8 syntax or dependencies
- **numpy/scipy**: major version breaks (numpy 1.x vs 2.x)

### 1c. Identify What You Actually Need

Most published repos contain far more than you need. Identify the minimum:

```
Typical repo structure:
├── train.py              ← usually NOT needed
├── evaluate.py           ← maybe useful for reference
├── models/               ← NEEDED: model definitions
│   ├── __init__.py
│   └── network.py
├── utils/                ← PARTIALLY needed
│   ├── data_loader.py    ← usually replace with yours
│   └── transforms.py     ← may be needed
├── pretrained/           ← NEEDED: weights
│   └── best_model.pth
├── configs/              ← reference only
└── requirements.txt      ← check for conflicts
```

Focus on: **model definition + pretrained weights + any custom transforms**.
Skip: training scripts, data loaders, configs, visualization code.

## Step 2: Choose Install Strategy

| Strategy | When to use | Command |
|----------|------------|---------|
| **pip install** | Repo is a proper package on PyPI or installable via git | `uv add git+https://github.com/OWNER/REPO` |
| **git submodule** | Need the full repo, want to track upstream updates | `git submodule add https://github.com/OWNER/REPO vendor/REPO` |
| **Vendor (copy)** | Only need a few files, repo is not pip-installable, or has dependency conflicts | Copy specific files into `src/<pkg>/vendor/REPO/` |

### pip install (preferred when possible)

```bash
# From PyPI
uv add package-name

# From GitHub (specific tag/commit for reproducibility)
uv add "package-name @ git+https://github.com/OWNER/REPO@v1.0.0"
```

### git submodule

```bash
git submodule add https://github.com/OWNER/REPO vendor/REPO
git submodule update --init

# Pin to specific commit
cd vendor/REPO && git checkout COMMIT_HASH && cd ../..
git add vendor/REPO && git commit -m "pin REPO to COMMIT_HASH"
```

Add to `.gitignore` if weights are large:
```
vendor/REPO/pretrained/*.pth
```

### Vendor (copy specific files)

```bash
mkdir -p src/<pkg>/vendor/REPO
cp /tmp/REPO/models/network.py src/<pkg>/vendor/REPO/
cp /tmp/REPO/utils/transforms.py src/<pkg>/vendor/REPO/
```

Always add a provenance comment at the top of vendored files:

```python
# Vendored from https://github.com/OWNER/REPO (commit abc1234)
# License: MIT
# Modified: <describe changes, or "none">
```

Install only the vendored code's direct dependencies:

```bash
uv add dependency-a dependency-b  # only what the vendored files import
```

## Step 3: Dependency Conflict Resolution

### Version Pinning Conflicts

If the external repo pins a version that conflicts with yours:

```bash
# Check what version you currently have
uv pip show torch

# Try installing the repo — uv will report conflicts
uv add git+https://github.com/OWNER/REPO
```

Resolution strategies (in order of preference):

1. **Test with your version** — often the pin is overly strict and the code works fine
2. **Vendor and patch** — copy the files, fix the import/API differences
3. **Subprocess isolation** — run the external tool in a separate venv (last resort)

### Subprocess Isolation (for irreconcilable conflicts)

When the external repo absolutely cannot coexist with your dependencies:

```python
import subprocess
import json

def run_external_model(input_path: str, output_path: str) -> dict:
    """Run external model in its own venv via subprocess."""
    result = subprocess.run(
        [
            "/path/to/external_venv/bin/python",
            "/path/to/external_repo/inference.py",
            "--input", input_path,
            "--output", output_path,
        ],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"External model failed: {result.stderr}")
    return json.loads(result.stdout)
```

Set up the isolated venv:

```bash
uv venv /path/to/external_venv --python 3.10
VIRTUAL_ENV=/path/to/external_venv uv pip install -r /path/to/external_repo/requirements.txt
```

## Step 4: Test the Integration

Before using the integration in experiments, verify it works:

```python
import torch


def test_external_model_loads():
    """Verify model loads and produces expected output shape."""
    from your_pkg.vendor.repo_adapter import ExternalModel

    model = ExternalModel(weights_path="path/to/weights.pth")
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    expected_features = 512  # replace with your adapter's true output size
    assert output.shape == (1, expected_features), f"Unexpected shape: {output.shape}"


def test_external_model_deterministic():
    """Verify reproducible outputs."""
    model = ExternalModel(weights_path="path/to/weights.pth")
    x = torch.randn(1, 3, 224, 224)
    torch.manual_seed(42)
    out1 = model(x)
    torch.manual_seed(42)
    out2 = model(x)
    assert torch.allclose(out1, out2)
```

### Checklist

- [ ] Model loads without errors
- [ ] Output shape matches expectations
- [ ] Output is deterministic with fixed seeds
- [ ] Works with your data format (not just their example data)
- [ ] GPU memory usage is acceptable
- [ ] No dependency conflicts at import time

## Common Pitfalls

- **Weights not included in repo**: Many repos host weights on Google Drive, Hugging Face, or Zenodo. Check the README for download links and automate the download in a setup script.
- **Hardcoded paths**: Published code often has hardcoded absolute paths. Search for them: `grep -rn "/home\|/data\|/mnt\|C:\\" models/`.
- **Global state**: Some repos set global seeds, modify `sys.path`, or change matplotlib backends at import time. Vendor and remove these side effects.
- **Missing `__init__.py`**: Many research repos don't have proper package structure. Add `__init__.py` files when vendoring.
- **Relative imports**: Research code often uses relative imports that break when you move files. Convert to absolute imports when vendoring.
