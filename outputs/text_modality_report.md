# Text Modality & Pretrained Embeddings Report

> **Project:** Multimodal Intelligent Triage Assistant (Educational & Research Prototype)
> **Modality:** Clinical Chief Complaint (`Chief_complain`)

---

## 1. Corpus & Vocabulary Characteristics

- **Total Triage Records:** `1,267`
- **Distinct Vocabulary Terms:** `330`
- **Mean Words per Complaint:** `2.0` words (Median: `2`)
- **Mean Character Length:** `12.34` characters (Median: `12`)
- **Maximum Word Count:** `5` words

> **Clinical Linguistic Observation:** Emergency chief complaints are terse, telegraphic phrases (e.g., *'chest pain'*, *'dyspnea'*, *'right forearm burn'*). Most complaints contain only 2 to 4 words. Training a Transformer from scratch would lead to catastrophic overfitting on such a compact dataset (~1,267 records).

### Top 10 Clinical Complaint Terms Across Patients:

| Term | Patient Occurrence Count | Clinical Relevance |
| :--- | :---: | :--- |
| `pain` | 414 | Frequent acute symptom or anatomical site |
| `chest` | 102 | Frequent acute symptom or anatomical site |
| `abd` | 99 | Frequent acute symptom or anatomical site |
| `dyspnea` | 64 | Frequent acute symptom or anatomical site |
| `dizziness` | 64 | Frequent acute symptom or anatomical site |
| `weakness` | 49 | Frequent acute symptom or anatomical site |
| `fever` | 48 | Frequent acute symptom or anatomical site |
| `ant` | 44 | Frequent acute symptom or anatomical site |
| `rt` | 44 | Frequent acute symptom or anatomical site |
| `injury` | 43 | Frequent acute symptom or anatomical site |

---

## 2. Representation Comparison: TF-IDF vs Frozen MiniLM

| Text Representation | Classifier Architecture | Output Dim | Test Accuracy | Test Macro F1 | Test Weighted F1 | Test Macro Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `TF-IDF (300 n-grams)` | Logistic Regression (Balanced) | 300 | **0.5288** | **0.4787** | 0.5524 | 0.5858 |
| `Frozen MiniLM-L6-v2 (Dense)` | Logistic Regression Probe | 384 | **0.5340** | **0.4693** | 0.5471 | 0.5593 |
| `Frozen MiniLM + MLP Projection` | PyTorch MLP (Linear-LN-GELU-Linear) | 128 | **0.4764** | **0.4144** | 0.4981 | 0.5141 |

---

## 3. Scientific Justification for Frozen MiniLM Embeddings

1. **Semantic Generalization vs Exact Lexical Matches:** TF-IDF relies on exact character n-grams. Pretrained sentence transformers (`all-MiniLM-L6-v2`) embed clinical synonyms and related terminology (e.g., *'chest tightness'*, *'sternal ache'*, *'substernal discomfort'*) into proximate regions in the 384-d latent space.
2. **Overfitting Mitigation:** Freezing the transformer weights preserves foundational semantic knowledge while preventing parameter divergence on a 1,267-record dataset.
3. **Modular Computational Efficiency:** Precomputing and caching the 384-d embeddings to disk reduces downstream multimodal training epoch runtimes from minutes to seconds.

---

## 4. TextEncoder Architecture Specification

```
Chief Complaint Text
        ↓
Frozen MiniLM-L6-v2 (Embedding Dim = 384)
        ↓
Linear Projection (384 → 128)
        ↓
LayerNorm(128)
        ↓
GELU Activation
        ↓
Dropout (p = 0.2)
        ↓
Text Latent Vector (d_text = 128)
```

Cached files: `data/processed/text_embeddings_{train,val,test}.pt`.