---
name: repo-integration
description: >
  Integrate published GitHub repositories into the current project. Use when:
  (1) Adding a pretrained model or method from a paper's GitHub repo,
  (2) Wrapping an external tool as a feature extractor or preprocessing step,
  (3) Resolving dependency conflicts between an external repo and the current
  project, (4) Adapting an external repo's data format to match the current
  pipeline, (5) Vendoring or submoduling code from another repository,
  (6) Loading models from HuggingFace Hub, MONAI Model Zoo, or timm.
---

# Repo Integration

## Workflow

Integrating external research code involves these steps:

1. **Evaluate** the external repo (license, compatibility, identify needed files)
2. **Choose install strategy** (pip, submodule, or vendor)
3. **Write an adapter** to bridge their API to your project
4. **Resolve dependency conflicts** if any
5. **Test** the integration with your data

## Decision Tree

**Where is the model hosted?**
- HuggingFace Hub → Use transformers/huggingface_hub API. See [hub-integrations.md](references/hub-integrations.md)
- MONAI Model Zoo → Use MONAI bundle API. See [hub-integrations.md](references/hub-integrations.md)
- timm (PyTorch Image Models) → Use `timm.create_model()`. See [hub-integrations.md](references/hub-integrations.md)
- Standalone GitHub repo → Follow the 5-step integration workflow. See [integration-workflow.md](references/integration-workflow.md)

**What install strategy?**
- Repo is a proper package (PyPI or pip-installable) → `uv add`
- Need full repo, want upstream updates → git submodule
- Only need a few files, or dependency conflicts → Vendor (copy files)

**What adapter pattern?**
- Extract embeddings from pretrained model → Feature extractor pattern. See [adapter-patterns.md](references/adapter-patterns.md)
- Run inference and get predictions → Classifier/segmenter pattern. See [adapter-patterns.md](references/adapter-patterns.md)
- Wrap preprocessing transforms → Preprocessor pattern. See [adapter-patterns.md](references/adapter-patterns.md)

**ASK the user** before starting:
- Is the model on HuggingFace, MONAI, timm, or a standalone GitHub repo?
- Do they need the full model or just feature extraction?
- Are there known dependency conflicts?

## References

| File | Read When |
|------|-----------|
| [references/integration-workflow.md](references/integration-workflow.md) | Full 5-step process: evaluating repos (license, compatibility), choosing install strategy, dependency conflicts, testing |
| [references/adapter-patterns.md](references/adapter-patterns.md) | Adapter class template, feature extractor, classifier/segmenter, preprocessor, batch processing, checkpoint loading recipes |
| [references/hub-integrations.md](references/hub-integrations.md) | Loading from HuggingFace Hub (biomedical vision + text models), MONAI (pretrained nets, transforms, bundles), timm |
