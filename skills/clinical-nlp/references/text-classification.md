# Text Classification Reference

## Contents

- Fine-Tuning BERT for Classification
- Multi-Label Classification
- Label Extraction from Radiology Reports
- Few-Shot Classification with LLMs
- Common Pitfalls


**ASK the user** about their label schema and dataset size -- small datasets (<500 samples) may benefit from few-shot LLM approaches rather than fine-tuning.

## Fine-Tuning BERT for Classification

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def fine_tune_classifier(train_texts: list[str], train_labels: list[int],
                         val_texts: list[str], val_labels: list[int],
                         model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                         num_labels: int = 2, output_dir: str = "./output",
                         epochs: int = 5, batch_size: int = 16,
                         learning_rate: float = 2e-5):
    """Fine-tune a biomedical BERT model for text classification."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True,
                         padding="max_length", max_length=512)

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        return {
            "f1_macro": f1_score(labels, preds, average="macro"),
            "accuracy": accuracy_score(labels, preds),
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer
```

## Multi-Label Classification

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultiLabelClassifier(nn.Module):
    """Multi-label classifier on top of a pretrained BERT model.

    Use for: ICD code prediction, multi-finding radiology reports, etc.
    """

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits  # apply sigmoid for probabilities


# Loss: BCEWithLogitsLoss (handles multi-label naturally)
criterion = nn.BCEWithLogitsLoss()
# Labels: (batch_size, num_labels) binary matrix
```

## Label Extraction from Radiology Reports

```python
import re

def extract_radiology_labels(report: str,
                             finding_patterns: dict[str, list[str]]) -> dict[str, bool]:
    """Rule-based label extraction from radiology reports.

    finding_patterns: {"finding_name": ["pattern1", "pattern2", ...]}
    Example: {"pneumonia": ["pneumonia", "consolidation", "airspace opacity"]}

    Use as a baseline or to generate weak labels for model training.
    """
    report_lower = report.lower()
    labels = {}

    # Negation patterns
    negation_pattern = re.compile(
        r"(no |no evidence of |without |negative for |rule out |"
        r"unlikely |absent |denies )",
        re.IGNORECASE,
    )

    for finding, patterns in finding_patterns.items():
        found = False
        negated = False
        for pattern in patterns:
            for match in re.finditer(re.escape(pattern), report_lower):
                found = True
                # Check for negation in the 50 chars before the match
                context = report_lower[max(0, match.start()-50):match.start()]
                if negation_pattern.search(context):
                    negated = True
        labels[finding] = found and not negated

    return labels
```

## Few-Shot Classification with LLMs

```python
def few_shot_classify(text: str, label_options: list[str],
                      examples: list[dict] | None = None) -> str:
    """Build a few-shot classification prompt.

    examples: [{"text": "...", "label": "..."}, ...]
    Returns the prompt string to send to an LLM.
    """
    prompt_parts = [
        "Classify the following clinical text into one of these categories: "
        + ", ".join(label_options) + ".",
        "",
    ]

    if examples:
        prompt_parts.append("Examples:")
        for ex in examples:
            prompt_parts.append(f"Text: {ex['text']}")
            prompt_parts.append(f"Category: {ex['label']}")
            prompt_parts.append("")

    prompt_parts.append(f"Text: {text}")
    prompt_parts.append("Category:")

    return "\n".join(prompt_parts)

# Usage with any LLM API:
# prompt = few_shot_classify(note, ["Positive", "Negative", "Uncertain"],
#                            examples=labeled_examples[:5])
# response = llm.generate(prompt)
```

## Common Pitfalls

- **Small dataset overfitting**: With <500 labeled samples, BERT fine-tuning often overfits. Use aggressive dropout (0.3-0.5), weight decay, and early stopping. Consider few-shot LLM approaches instead.
- **Label noise**: Clinical labels are noisy (inter-annotator disagreement, automated label extraction errors). Use label smoothing or noise-robust losses.
- **Long documents**: Clinical notes often exceed 512 tokens. Options: (a) truncate (loses information), (b) chunk and aggregate, (c) use Longformer/BigBird, (d) hierarchical model.
- **Class imbalance**: Clinical datasets are typically imbalanced. Use weighted loss, oversampling, or focal loss. Report macro F1, not accuracy.
- **Evaluation**: Always report per-class metrics for clinical tasks. Aggregate metrics can hide poor performance on rare but important classes.
