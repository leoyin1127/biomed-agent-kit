# Biomedical Language Models Reference

**ASK the user** what type of biomedical text they're working with -- PubMedBERT is best for literature, ClinicalBERT for clinical notes, BioBERT for general biomedical text.

## Model Comparison

| Model | Training Data | Params | Best For |
|-------|--------------|--------|----------|
| PubMedBERT | PubMed abstracts + full text | 110M | Literature search, PubMed text |
| BioBERT | PubMed + PMC | 110M | General biomedical text |
| ClinicalBERT | MIMIC-III clinical notes | 110M | Clinical notes, EHR text |
| BioGPT | PubMed literature | 347M | Text generation, QA |
| SciBERT | Semantic Scholar papers | 110M | Scientific text (broad) |
| GatorTron | >90B words clinical text | 8.9B | Large-scale clinical NLP |

## Loading Models from HuggingFace

```python
from transformers import AutoModel, AutoTokenizer

# ── PubMedBERT (best for PubMed/literature) ─────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)
model = AutoModel.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)

# ── BioBERT (general biomedical) ────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# ── ClinicalBERT (clinical notes) ──────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# ── BioGPT (generative) ────────────────────────────────────────
from transformers import BioGptTokenizer, BioGptForCausalLM
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
```

## Extracting Embeddings

```python
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

class BiomedicalEmbedder:
    """Extract embeddings from biomedical BERT models."""

    def __init__(self, model_name: str, device: str = "cuda",
                 pooling: str = "cls"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.pooling = pooling  # "cls", "mean", "last_n_mean"

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int = 512,
               batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings. Returns (N, D) array."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)

            if self.pooling == "cls":
                emb = hidden[:, 0, :]
            elif self.pooling == "mean":
                emb = (hidden * mask).sum(1) / mask.sum(1)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            all_embeddings.append(emb.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def cleanup(self):
        del self.model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
```

## Sentence Similarity

```python
from sentence_transformers import SentenceTransformer

# Biomedical sentence embeddings
model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

sentences = [
    "Patient presents with shortness of breath",
    "The subject exhibits dyspnea on exertion",
    "Blood glucose levels were within normal range",
]
embeddings = model.encode(sentences)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)
```

## Common Pitfalls

- **Max token length**: BERT models truncate at 512 tokens. Clinical notes are often much longer. Use chunking with overlap or hierarchical approaches.
- **Wrong tokenizer**: Always use the tokenizer that matches the model. Mixing tokenizers silently produces garbage.
- **Domain mismatch**: PubMedBERT trained on literature != clinical notes. ClinicalBERT may not understand genetics papers. Match model to data.
- **Cased vs uncased**: Some biomedical models are uncased (lowercase everything). Check the model card.
- **Subword tokenization**: Medical terms like "cardiomyopathy" get split into subwords. This is normal but affects token-level tasks (NER).
