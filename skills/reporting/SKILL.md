---
name: reporting
description: >
  Generate publication-quality figures and markdown reports for biomedical
  ML experiments. Use when: (1) Creating heatmaps, forest plots, radar
  charts, or sensitivity-specificity scatter plots, (2) Plotting ROC curves,
  precision-recall curves, or Kaplan-Meier survival curves, (3) Writing
  structured experiment reports, (4) Exporting results to LaTeX tables for
  paper submission, (5) Producing figures meeting publication DPI and
  formatting standards.
---

# Scientific Reporting

## Workflow

Generating a report involves these steps:

1. **Choose figure types** -- based on what dimensions you're comparing
2. **Generate figures** -- at appropriate DPI and formatting
3. **Export tables** -- to markdown and/or LaTeX
4. **Write report** -- structured markdown with figures and tables
5. **Save to dated directory** -- `docs/<YYYYMMDD>/` to prevent overwrites

## Decision Tree

**What are you visualizing?**
- Performance across 2 dimensions (e.g., model x feature set) → Heatmap. See [reporting-patterns.md](references/reporting-patterns.md)
- Point estimates with confidence intervals → Forest plot. See [reporting-patterns.md](references/reporting-patterns.md)
- Multiple metrics for top configurations → Radar/spider plot. See [reporting-patterns.md](references/reporting-patterns.md)
- Sensitivity vs specificity trade-off → Scatter plot. See [reporting-patterns.md](references/reporting-patterns.md)
- Model discrimination (classification) → ROC curve or PR curve. See [curve-plots.md](references/curve-plots.md)
- Survival analysis → Kaplan-Meier curve with log-rank test. See [curve-plots.md](references/curve-plots.md)
- Results for paper submission → LaTeX table. See [latex-tables.md](references/latex-tables.md)

**What is the target venue?**
- Internal review → DPI 150, PNG format
- Journal/conference submission → DPI 300+, PDF/SVG format
- **ASK the user** about target venue and formatting requirements

**ASK the user** before starting:
- What figures do they need?
- Target venue (determines DPI, formatting)?
- Color-blind-friendly palette needed?
- What file format (PNG, PDF, SVG)?

## References

| File | Read When |
|------|-----------|
| [references/reporting-patterns.md](references/reporting-patterns.md) | Heatmaps, forest plots, radar/spider charts, sensitivity-specificity scatter, dated output directories, markdown report template, figure quality guidelines |
| [references/curve-plots.md](references/curve-plots.md) | ROC curves (single and with bootstrap CI), precision-recall curves, Kaplan-Meier survival curves with log-rank test |
| [references/latex-tables.md](references/latex-tables.md) | Exporting results to LaTeX tables with booktabs formatting, standalone preview files |
