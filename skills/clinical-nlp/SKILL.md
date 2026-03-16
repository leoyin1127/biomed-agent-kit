---
name: clinical-nlp
description: >
  Process and analyze biomedical text data (clinical notes, radiology reports,
  PubMed abstracts). Use when: (1) Loading biomedical language models (PubMedBERT,
  ClinicalBERT, BioBERT, BioGPT) and extracting embeddings, (2) Named entity
  recognition for medical concepts with scispaCy, (3) UMLS concept linking,
  (4) Text classification of clinical documents, (5) Extracting structured data
  from unstructured clinical text (vitals, lab values, findings).
---

# Clinical NLP

## Workflow

Processing biomedical text involves these steps:

1. **Assess the data** -- determine text type, check de-identification status
2. **Preprocess** -- clean text, segment sections, handle abbreviations
3. **Choose a model** -- select appropriate biomedical LM for the task
4. **Execute the task** -- NER, classification, embedding extraction, or information extraction
5. **Evaluate** -- with appropriate metrics (macro F1 for multi-class, per-label for multi-label)

## Decision Tree

**What type of text?**
- Clinical notes (EHR) → Use ClinicalBERT. **Check de-identification first.**
- PubMed abstracts / literature → Use PubMedBERT
- General biomedical text → Use BioBERT
- Need text generation → Use BioGPT

**What is the task?**
- Extract medical entities → scispaCy NER + UMLS linking. See [clinical-text.md](references/clinical-text.md)
- Classify documents → Fine-tune BERT (>500 samples) or few-shot LLM (<500). See [text-classification.md](references/text-classification.md)
- Extract embeddings for downstream ML → See [biomedical-lm.md](references/biomedical-lm.md)
- Extract structured data (vitals, labs) → Regex patterns. See [clinical-text.md](references/clinical-text.md)

**How large is the labeled dataset?**
- Large (>1K samples) → Fine-tune biomedical BERT
- Small (100-1K) → Fine-tune with aggressive regularization
- Very small (<100) → Few-shot LLM prompting

**ASK the user** before starting:
- What type of text are they working with?
- Is the data de-identified? If PHI is present, advise on de-identification first.
- What is the end goal (classification, NER, embeddings, extraction)?

## References

| File | Read When |
|------|-----------|
| [references/biomedical-lm.md](references/biomedical-lm.md) | Loading PubMedBERT, ClinicalBERT, BioBERT, BioGPT from HuggingFace; extracting embeddings (CLS, mean pooling); sentence similarity |
| [references/clinical-text.md](references/clinical-text.md) | Preprocessing clinical notes, section segmentation, scispaCy NER, UMLS linking, extracting vitals/lab values, de-identification |
| [references/text-classification.md](references/text-classification.md) | Fine-tuning BERT for classification (HuggingFace Trainer), multi-label classification, label extraction from radiology reports, few-shot LLM |
