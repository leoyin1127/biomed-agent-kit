# Search Strategy Reference

## Step 1: Define Search Scope

Before searching, clarify with the user:

- **Topic**: Specific problem (e.g., "cell segmentation in histopathology" not just "segmentation")
- **Modality**: imaging, text, genomics, EHR, etc.
- **Task type**: classification, segmentation, detection, generation, survival prediction, etc.
- **Time range**: recent advances (1-2 years) vs comprehensive review (5+ years)

**ASK the user** to be as specific as possible about their research question -- broad queries return too many irrelevant results.

## Step 2: Search Sources

### PubMed (Biomedical Literature)

Best for: clinical studies, medical imaging, biomedical methods published in medical journals.

```python
from Bio import Entrez

Entrez.email = "your.email@institution.edu"

def search_pubmed(query: str, max_results: int = 50,
                  sort: str = "relevance") -> list[dict]:
    """Search PubMed and return structured results.

    sort: "relevance", "pub_date", or "most_recent"
    """
    handle = Entrez.esearch(db="pubmed", term=query,
                            retmax=max_results, sort=sort)
    record = Entrez.read(handle)
    ids = record["IdList"]

    if not ids:
        return []

    handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml")
    records = Entrez.read(handle)

    results = []
    for article in records["PubmedArticle"]:
        info = article["MedlineCitation"]["Article"]
        results.append({
            "pmid": str(article["MedlineCitation"]["PMID"]),
            "title": str(info["ArticleTitle"]),
            "journal": str(info["Journal"]["Title"]),
            "year": str(info["Journal"]["JournalIssue"].get("PubDate", {}).get("Year", "N/A")),
            "abstract": str(info.get("Abstract", {}).get("AbstractText", [""])[0]),
        })
    return results
```

Useful PubMed query syntax:
- `"deep learning"[Title] AND "pathology"[Title/Abstract]` -- field-specific search
- `"2023/01/01"[Date - Publication] : "2025/12/31"[Date - Publication]` -- date range
- `"Nature Medicine"[Journal]` -- journal-specific
- `AND (hasabstract[text])` -- only papers with abstracts

### Semantic Scholar API

Best for: ML/AI papers, citation analysis, finding influential papers.

```python
import requests

def search_semantic_scholar(query: str, max_results: int = 50,
                            year_range: str | None = None,
                            venue: str | None = None,
                            fields_of_study: list[str] | None = None,
                            open_access_only: bool = False) -> list[dict]:
    """Search Semantic Scholar with filtering.

    year_range: e.g. "2023-2025"
    venue: e.g. "MICCAI" or "Nature Medicine"
    fields_of_study: e.g. ["Medicine", "Computer Science"]
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "title,authors,year,venue,citationCount,abstract,externalIds,isOpenAccess,url",
    }
    if year_range:
        params["year"] = year_range
    if venue:
        params["venue"] = venue
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)
    if open_access_only:
        params["openAccessPdf"] = ""

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    return [{
        "title": p.get("title", ""),
        "authors": [a["name"] for a in p.get("authors", [])],
        "year": p.get("year"),
        "venue": p.get("venue", ""),
        "citations": p.get("citationCount", 0),
        "abstract": p.get("abstract", ""),
        "doi": p.get("externalIds", {}).get("DOI", ""),
        "arxiv_id": p.get("externalIds", {}).get("ArXiv", ""),
        "url": p.get("url", ""),
        "open_access": p.get("isOpenAccess", False),
    } for p in data.get("data", [])]


def get_paper_details(paper_id: str) -> dict:
    """Get detailed info about a paper (by Semantic Scholar ID, DOI, or ArXiv ID).

    paper_id formats: "DOI:10.1234/...", "ARXIV:2301.12345", "CorpusId:12345"
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    params = {"fields": "title,authors,year,venue,citationCount,abstract,references,citations,tldr,externalIds,fieldsOfStudy"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_citations(paper_id: str, max_results: int = 100) -> list[dict]:
    """Get papers that cite this paper (find follow-up work)."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
    params = {"fields": "title,authors,year,venue,citationCount", "limit": max_results}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return [c["citingPaper"] for c in resp.json().get("data", [])]


def get_references(paper_id: str, max_results: int = 100) -> list[dict]:
    """Get papers referenced by this paper (trace foundational work)."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
    params = {"fields": "title,authors,year,venue,citationCount", "limit": max_results}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return [r["citedPaper"] for r in resp.json().get("data", [])]
```

### arXiv API

Best for: preprints, cutting-edge ML methods not yet published.

```python
import arxiv

def search_arxiv(query: str, max_results: int = 50,
                 categories: list[str] | None = None) -> list[dict]:
    """Search arXiv for preprints.

    categories: e.g. ["cs.CV", "cs.LG", "cs.AI", "q-bio.QM", "eess.IV"]
    """
    cat_filter = ""
    if categories:
        cat_filter = " AND (" + " OR ".join(f"cat:{c}" for c in categories) + ")"

    search = arxiv.Search(
        query=query + cat_filter,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    return [{
        "title": p.title,
        "authors": [a.name for a in p.authors],
        "published": p.published.strftime("%Y-%m-%d"),
        "arxiv_id": p.entry_id.split("/")[-1],
        "abstract": p.summary,
        "categories": p.categories,
        "pdf_url": p.pdf_url,
    } for p in search.results()]
```

### Google Scholar

Google Scholar doesn't have a free API. Use Semantic Scholar or PubMed as primary sources. For Google Scholar, use Claude Code's WebSearch tool with queries like `site:scholar.google.com {your query}`.

## Step 3: Filter by Quality

```python
def filter_by_quality(papers: list[dict], min_citations: int = 0,
                      top_venues: set[str] | None = None,
                      min_year: int | None = None) -> list[dict]:
    """Filter and rank papers by quality indicators."""
    filtered = []
    for p in papers:
        if min_year and p.get("year", 0) and p["year"] < min_year:
            continue
        if min_citations and p.get("citations", 0) < min_citations:
            continue
        if top_venues and p.get("venue", "") and p["venue"] not in top_venues:
            continue
        filtered.append(p)
    filtered.sort(key=lambda x: x.get("citations", 0), reverse=True)
    return filtered
```

### Quality Heuristics

| Signal | High Quality | Lower Quality |
|--------|-------------|---------------|
| Venue | Top-tier journal/conference | Workshop, preprint-only |
| Citations | >50 in 2 years, >200 all-time | <5 after 2+ years |
| Code | Open-source, reproducible | No code available |
| Dataset | Public benchmark, multi-site | Single-site, private |
| Evaluation | Multiple metrics, CI, ablations | Single metric, no CI |

**ASK the user** what quality threshold they want -- strict (top venues only) or inclusive (preprints, workshops).

## Step 4: Extract and Organize

```python
import pandas as pd
import json

def save_search_results(papers: list[dict], output_path: str):
    """Save search results to CSV and JSON."""
    df = pd.DataFrame(papers)
    df.to_csv(output_path.replace(".json", ".csv"), index=False)
    with open(output_path, "w") as f:
        json.dump(papers, f, indent=2, default=str)
```

## Common Pitfalls

- **Query too broad**: "deep learning medical imaging" returns thousands of results. Narrow to specific modality + task.
- **Recency bias**: Sorting by date misses foundational papers. Also search by citation count.
- **Venue bias**: Not all great work appears in top venues. Check preprints and workshops for cutting-edge methods.
- **Missing code**: Check Papers With Code (paperswithcode.com) for implementations.
- **API rate limits**: Semantic Scholar allows ~100 requests/5 minutes without an API key.
