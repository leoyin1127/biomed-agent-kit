# LaTeX Table Export

## Results Table

```python
import pandas as pd

def to_latex_results_table(df: pd.DataFrame, metric_prefixes: list[str],
                           dim_cols: list[str], caption: str = "",
                           label: str = "tab:results") -> str:
    """Convert aggregated results DataFrame to a LaTeX table.

    metric_prefixes: e.g. ["test_auc_roc", "test_balanced_acc"]
        Each prefix expects _mean and _ci_low, _ci_high columns.
    dim_cols: experiment dimension columns, e.g. ["model", "feature_set"]
    """
    rows = []
    for _, r in df.iterrows():
        row = [str(r[c]).replace("_", r"\_") for c in dim_cols]
        for prefix in metric_prefixes:
            mean = r[f"{prefix}_mean"]
            ci_lo = r[f"{prefix}_ci_low"]
            ci_hi = r[f"{prefix}_ci_high"]
            row.append(f"{mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        rows.append(row)

    metric_headers = [p.replace("test_", "").replace("_", " ").title()
                      for p in metric_prefixes]
    headers = [c.replace("_", " ").title() for c in dim_cols] + metric_headers

    col_spec = "l" * len(dim_cols) + "c" * len(metric_prefixes)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}" if caption else "",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(line for line in lines if line)
```

## Saving LaTeX Tables

```python
def save_latex_table(latex: str, output_path: str):
    """Write LaTeX table to a .tex file for inclusion in papers."""
    with open(output_path, "w") as f:
        f.write(latex)
    # Also save standalone compilable file for preview
    standalone = (
        r"\documentclass{article}" + "\n"
        r"\usepackage{booktabs}" + "\n"
        r"\begin{document}" + "\n"
        + latex + "\n"
        r"\end{document}" + "\n"
    )
    preview_path = output_path.replace(".tex", "_preview.tex")
    with open(preview_path, "w") as f:
        f.write(standalone)
```

## Common Pitfalls

- **Underscore escaping**: LaTeX treats `_` as subscript. Escape with `\_` in text columns.
- **Table width**: Wide tables need `\resizebox{\textwidth}{!}{...}` or `tabularx`.
- **Booktabs**: Always use `\toprule`, `\midrule`, `\bottomrule` (requires `\usepackage{booktabs}`).

**ASK the user** which LaTeX packages their paper uses -- some venues have specific table formatting requirements.
