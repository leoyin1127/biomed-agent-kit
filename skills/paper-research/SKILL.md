---
name: paper-research
description: >
  Research biomedical and ML papers across the internet, focusing on top venues
  and publishers. Use when: (1) Searching for state-of-the-art methods in a
  biomedical domain, (2) Finding papers from top venues (Nature Medicine, Lancet,
  MICCAI, NeurIPS, ICML, CVPR, TMI, MedIA), (3) Reviewing related work for a
  research topic, (4) Finding benchmark datasets or baselines for a task,
  (5) Comparing methods across papers, (6) Summarizing recent advances in a
  biomedical subfield.
---

# Paper Research

## Workflow

Researching biomedical papers involves these steps:

1. **Define scope** -- narrow the research question with the user
2. **Search** -- query PubMed, Semantic Scholar, and arXiv
3. **Filter** -- by venue quality, citations, recency, and code availability
4. **Analyze** -- extract methods, datasets, metrics, and results
5. **Synthesize** -- compare approaches, identify gaps, create comparison tables

## Decision Tree

**What is the search goal?**
- Find SOTA methods → Search by citation count + recent years, filter by top venues. See [search-strategy.md](references/search-strategy.md)
- Comprehensive literature review → Systematic search across all sources, document inclusion/exclusion criteria. See [paper-analysis.md](references/paper-analysis.md)
- Find datasets/benchmarks → Search Papers With Code, check datasets sections of top papers
- Find implementations → Search Papers With Code + GitHub. See [paper-analysis.md](references/paper-analysis.md)

**Which search source?**
- Clinical / medical journals → PubMed (Entrez API)
- ML / AI conferences → Semantic Scholar or arXiv
- Preprints / cutting-edge → arXiv (cs.CV, cs.LG, cs.CL, eess.IV)
- Broadest coverage → Google Scholar (via WebSearch)

**What quality threshold?**
- Strict (established methods) → Top venues only, >50 citations
- Inclusive (emerging work) → Include workshops, preprints, low citations

**ASK the user** before starting:
- What is the specific research question or topic?
- What subfield (medical imaging, clinical NLP, drug discovery, genomics)?
- What time range (last 1 year, last 3 years, all time)?
- Are they looking for methods, datasets, benchmarks, or all?
- Want a broad survey or deep dive on fewer papers?

## References

| File | Read When |
|------|-----------|
| [references/search-strategy.md](references/search-strategy.md) | Using PubMed, Semantic Scholar, and arXiv APIs; filtering by quality; query syntax; organizing results |
| [references/top-venues.md](references/top-venues.md) | Identifying top venues by subfield: medical imaging, ML/AI, bioinformatics, clinical informatics, pathology |
| [references/paper-analysis.md](references/paper-analysis.md) | Structured paper reading template, literature comparison tables, systematic search reporting, finding code on Papers With Code |
