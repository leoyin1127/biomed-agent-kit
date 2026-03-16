# Clinical Text Processing Reference

## Contents

- Preprocessing Clinical Notes
- Medical NER with scispaCy
- UMLS Concept Linking
- Extracting Clinical Patterns
- De-Identification Considerations
- Common Pitfalls


**ASK the user** whether their clinical text is de-identified. If it contains PHI, advise on de-identification before any processing.

## Preprocessing Clinical Notes

```python
import re

def preprocess_clinical_note(text: str, lowercase: bool = False) -> str:
    """Basic preprocessing for clinical notes.

    Be careful with lowercasing -- medical abbreviations are case-sensitive
    (e.g., "BID" = twice daily, "bid" could be ambiguous).
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize common patterns
    text = re.sub(r"(\d+)\s*/\s*(\d+)", r"\1/\2", text)  # "120 / 80" -> "120/80"

    if lowercase:
        text = text.lower()

    return text


def segment_sections(text: str) -> dict[str, str]:
    """Split a clinical note into sections.

    Common sections: CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS,
    PAST MEDICAL HISTORY, MEDICATIONS, ASSESSMENT, PLAN, etc.
    """
    # Common section header patterns
    section_pattern = re.compile(
        r"^(CHIEF COMPLAINT|CC|HPI|HISTORY OF PRESENT ILLNESS|"
        r"PAST MEDICAL HISTORY|PMH|MEDICATIONS|MEDS|"
        r"ALLERGIES|PHYSICAL EXAM|PE|ASSESSMENT|PLAN|"
        r"IMPRESSION|FINDINGS|RECOMMENDATIONS|"
        r"SOCIAL HISTORY|FAMILY HISTORY|REVIEW OF SYSTEMS|ROS)"
        r"\s*[:\-]?\s*",
        re.MULTILINE | re.IGNORECASE,
    )

    sections = {}
    matches = list(section_pattern.finditer(text))

    for i, match in enumerate(matches):
        section_name = match.group(1).upper().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[section_name] = text[start:end].strip()

    return sections
```

## Medical NER with scispaCy

```python
import spacy

def setup_clinical_ner():
    """Load scispaCy models for clinical NER.

    Install: pip install scispacy
    Models: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
    """
    nlp = spacy.load("en_core_sci_lg")
    return nlp


def extract_entities(nlp, text: str) -> list[dict]:
    """Extract biomedical entities from text."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })
    return entities
```

## UMLS Concept Linking

```python
def setup_umls_linker():
    """Set up UMLS entity linking with scispaCy.

    Links extracted entities to UMLS concepts (CUI codes).
    """
    import spacy
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker",
                 config={"resolve_abbreviations": True,
                         "linker_name": "umls"})
    return nlp


def link_to_umls(nlp, text: str) -> list[dict]:
    """Extract entities and link to UMLS concepts."""
    doc = nlp(text)
    linked = []
    linker = nlp.get_pipe("scispacy_linker")

    for ent in doc.ents:
        concepts = []
        for cui, score in ent._.kb_ents[:3]:  # top 3 matches
            concept = linker.kb.cui_to_entity[cui]
            concepts.append({
                "cui": cui,
                "name": concept.canonical_name,
                "score": round(score, 3),
                "types": list(concept.types),
            })
        linked.append({
            "text": ent.text,
            "concepts": concepts,
        })
    return linked
```

## Extracting Clinical Patterns

```python
import re

def extract_vitals(text: str) -> dict:
    """Extract vital signs from clinical text."""
    vitals = {}
    patterns = {
        "blood_pressure": r"(?:BP|blood pressure)\s*[:\s]*(\d{2,3})\s*/\s*(\d{2,3})",
        "heart_rate": r"(?:HR|heart rate|pulse)\s*[:\s]*(\d{2,3})",
        "temperature": r"(?:temp|temperature)\s*[:\s]*(\d{2,3}\.?\d?)\s*(?:F|C|°)?",
        "spo2": r"(?:SpO2|O2\s*sat|oxygen)\s*[:\s]*(\d{2,3})\s*%?",
        "respiratory_rate": r"(?:RR|respiratory rate)\s*[:\s]*(\d{1,2})",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            vitals[name] = match.groups()
    return vitals


def extract_lab_values(text: str) -> list[dict]:
    """Extract lab values with units."""
    pattern = re.compile(
        r"(\w[\w\s]{2,30}?)\s*[:\s]+\s*"
        r"(\d+\.?\d*)\s*"
        r"(mg/dL|g/dL|mmol/L|mEq/L|U/L|IU/L|ng/mL|%|x10\^?\d)?",
        re.IGNORECASE,
    )
    return [{"name": m[0].strip(), "value": float(m[1]), "unit": m[2] or ""}
            for m in pattern.findall(text)]
```

## De-Identification Considerations

- **Never train on data containing PHI** (Protected Health Information)
- **Use de-identified datasets** for development (MIMIC-III/IV, n2c2 shared tasks)
- **Common de-identification tools**:
  - [Philter](https://github.com/BCHSI/philter-ucsf) -- rule-based, high recall
  - [scrubadub](https://github.com/LeapBeyond/scrubadub) -- general PII removal
  - [AWS Comprehend Medical](https://aws.amazon.com/comprehend/medical/) -- cloud-based PHI detection

**ASK the user** about their data governance requirements before processing any clinical text.

## Common Pitfalls

- **Clinical abbreviations**: Highly ambiguous. "PT" = patient, physical therapy, prothrombin time, or part-time. Context matters.
- **Negation detection**: "No evidence of malignancy" means the OPPOSITE of "evidence of malignancy." Use negation detection (NegEx, negspaCy) before extracting findings.
- **Section context**: A medication listed in "Allergies" means something very different from one in "Current Medications."
- **Misspellings**: Clinical notes are full of typos. Consider fuzzy matching for entity extraction.
- **Temporal expressions**: "History of" vs "currently has" vs "rule out" -- temporal and certainty context changes meaning entirely.
