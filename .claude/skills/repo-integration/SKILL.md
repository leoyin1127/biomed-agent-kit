---
name: repo-integration
description: >
  Integrate published GitHub repositories into the current project. Use when:
  (1) Adding a pretrained model or method from a paper's GitHub repo,
  (2) Wrapping an external tool as a feature extractor or preprocessing step,
  (3) Resolving dependency conflicts between an external repo and the current
  project, (4) Adapting an external repo's data format to match the current
  pipeline, (5) Vendoring or submoduling code from another repository.
---

# Repo Integration

Systematic workflow for incorporating published research code into your project.

## Workflow

1. **Evaluate** the external repo before writing any code
2. **Choose an install strategy** (pip, submodule, or vendor)
3. **Write an adapter** to bridge their API to your project
4. **Isolate dependencies** if there are conflicts
5. **Test** the integration with your data

See [references/integration-workflow.md](references/integration-workflow.md) for the full step-by-step process.

## Adapter Patterns

Reusable code patterns for wrapping external repos as model inference, feature
extraction, or preprocessing components.

See [references/adapter-patterns.md](references/adapter-patterns.md) for implementations.

## References

| File | Apply When |
|------|------------|
| [references/integration-workflow.md](references/integration-workflow.md) | Evaluating, installing, and testing an external repo |
| [references/adapter-patterns.md](references/adapter-patterns.md) | Writing wrappers for models, feature extractors, preprocessors |
