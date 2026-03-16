# Paper Analysis Patterns

## Structured Paper Reading Template

```markdown
## Paper: {Title}

**Authors**: {author list}
**Venue**: {journal/conference} ({year})
**DOI/arXiv**: {identifier}
**Code**: {GitHub URL or "not available"}

### Problem
{What problem does this paper solve? 1-2 sentences}

### Method
- Architecture: {model architecture}
- Training: {training strategy, loss functions}
- Key innovation: {what makes this different}

### Data
- Dataset(s): {names, sizes, modalities}
- Split strategy: {train/val/test}
- Preprocessing: {key steps}

### Results
| Metric | Value | Baseline | Improvement |
|--------|-------|----------|-------------|
| {metric} | {value} | {baseline} | {delta} |

### Strengths
- {strength 1}

### Limitations
- {limitation 1}

### Relevance to Our Work
{How does this relate? What can we adopt?}
```

## Literature Comparison Table

```python
import pandas as pd

def create_literature_table(papers: list[dict]) -> pd.DataFrame:
    """Create a structured comparison of papers."""
    rows = []
    for p in papers:
        rows.append({
            "Paper": p.get("short_name", p["title"][:40] + "..."),
            "Year": p["year"],
            "Venue": p["venue"],
            "Method": p.get("method_summary", ""),
            "Dataset": p.get("dataset", ""),
            "Primary Metric": p.get("primary_metric", ""),
            "Result": p.get("primary_result", ""),
            "Code": "Yes" if p.get("code_url") else "No",
            "Citations": p.get("citations", 0),
        })
    return pd.DataFrame(rows).sort_values("Year", ascending=False)
```

## Literature Review Outline

```markdown
# Related Work: {Topic}

## 1. {Subtopic A} (e.g., "Traditional Approaches")
{Overview of foundational methods. 3-5 key papers.}

## 2. {Subtopic B} (e.g., "Deep Learning Methods")
{How DL changed the field. Key architectures.}

## 3. {Subtopic C} (e.g., "Foundation Models")
{Recent paradigm shifts. Emerging methods.}

## 4. Gaps and Opportunities
{What hasn't been addressed? Where does your work fit?}

## Summary Table
| Category | Representative Work | Contribution | Limitation |
|----------|-------------------|--------------|------------|
```

## Systematic Search Reporting

```markdown
# Literature Search Report -- {Date}

## Research Question
{Specific question}

## Search Strategy
- PubMed: query = `{exact query}`
- Semantic Scholar: query = `{exact query}`, filters = {filters}
- arXiv: query = `{exact query}`, categories = {categories}

## Inclusion Criteria
- {criterion 1}

## Exclusion Criteria
- {criterion 1}

## Results
- Total found: {N}
- After deduplication: {N}
- After screening: {N}
- Final selected: {N}
```

## Finding Code for Papers

```python
import requests

def find_paper_code(title: str) -> list[dict]:
    """Search Papers With Code for implementations."""
    url = "https://paperswithcode.com/api/v1/papers/"
    params = {"q": title}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    repos = []
    for paper in results[:5]:
        paper_id = paper.get("id")
        if paper_id:
            repo_url = f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
            repo_resp = requests.get(repo_url, timeout=30)
            if repo_resp.ok:
                for repo in repo_resp.json().get("results", []):
                    repos.append({
                        "paper_title": paper.get("title", ""),
                        "github_url": repo.get("url", ""),
                        "stars": repo.get("stars", 0),
                        "framework": repo.get("framework", ""),
                    })
    return repos
```

## Common Pitfalls

- **Cherry-picking**: Include competing methods, not just papers that support your approach.
- **Outdated baselines**: Compare against latest SOTA, not just convenient old baselines.
- **Missing ablations**: Note whether papers include ablation studies -- they indicate rigor.
- **Preprint vs published**: Note peer-review status. Preprints may change significantly.
- **Reproducibility**: Prioritize papers with public code and clear experimental details.

**ASK the user** whether they want a broad survey or deep dive before starting.
