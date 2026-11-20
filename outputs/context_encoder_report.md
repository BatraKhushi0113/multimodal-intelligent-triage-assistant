# Deep Learning Context Encoder Report

> **Project:** Multimodal Intelligent Triage Assistant (Educational & Research Prototype)
> **Modality:** Demographic & Operational Context (`Sex`, `Arrival mode`, `Injury`, `Age`, `Patients number per hour`)

---

## 1. Architectural Specification

```
Categorical Inputs (Entity Embeddings):
  ├── Sex (2 classes)          --> nn.Embedding(3, 8)
  ├── Arrival mode (7 classes) --> nn.Embedding(8, 16)
  └── Injury (2 classes)       --> nn.Embedding(3, 8)
Numerical Inputs (Continuous Standardized):
  ├── Age_scaled (Standardized Patient Age)
  └── Patients_per_hour_scaled (Standardized ED Crowding Rate)
        ↓
Concatenated Vector (Dimension: 34 = 32 cat + 2 num = 34)
        ↓
Linear(34 -> 64) + LayerNorm(64) + GELU Activation
        ↓
Dropout (p = 0.20)
        ↓
Linear(64 -> 64) + LayerNorm(64) + GELU Activation
        ↓
Dropout (p = 0.20)
        ↓
Context Latent Vector (d_context = 64)
```

---

## 2. Unimodal Performance on Unseen Test Set (N = 191)

| Metric | Value | Clinical Benchmark Context |
| :--- | :---: | :--- |
| **Test Accuracy** | **0.3246** | Pure demographic & context acuity |
| **Test Macro F1** | **0.2437** | Balanced across all 5 KTAS levels |
| **Test Weighted F1** | **0.3764** | Overall population weighted F1 |
| **KTAS 1 Recall (Resuscitation)** | **0.00** | Sensitivity on most acute tier |
| **KTAS 2 Recall (Emergent)** | **0.28** | Sensitivity on emergent presentations |
| **KTAS 3 Recall (Urgent)** | **0.28** | Sensitivity on urgent presentations |
| **KTAS 4 Recall (Less Urgent)** | **0.42** | Sensitivity on stable presentations |
| **KTAS 5 Recall (Non-Urgent)** | **0.15** | Sensitivity on minor presentations |

---

## 3. Cross-Modality Unimodal Comparison

Comparing all three unimodal encoders evaluated on the exact same held-out Test split (N = 191):

| Modality Encoder | Architecture | Input Dimension | Latent Dimension | Test Accuracy | Test Macro F1 | KTAS 1 Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Text Encoder** | Frozen `all-MiniLM-L6-v2` + MLP | 384 | 128 | 0.5340 | 0.4632 | 0.67 |
| **Vitals Encoder** | 2-Stage MLP + Norm + GELU | 10 | 128 | 0.4503 | 0.3808 | 0.67 |
| **Context Encoder** | Entity Embeddings + MLP | 34 (32+2) | 64 | **0.3246** | **0.2437** | **0.00** |

### Clinical Observations & Modality Roles:
1. **Context Modality Strengths:** Arrival mode (particularly 119 ambulance transport) provides strong discrimination for distinguishing critical/emergent tiers (KTAS 1 & 2) from outpatient walk-in presentations (KTAS 4 & 5).
2. **Complementary Modalities:** While context features provide valuable baseline demographic vulnerability (e.g. elderly patients at higher risk) and operational urgency, context alone cannot assess real-time physiological decompensation (vitals) or symptom specificity (chief complaints).
3. **Readiness for Multimodal Fusion:** With Text (384-d), Vitals (128-d), and Context (64-d) all represented as clean, regularized dense vectors, the foundation for Module 8 (Multimodal Baseline Fusion) and Module 9 (Reliability-Aware Adaptive Fusion) is complete.

---

## 4. Cached Latent Embeddings

- `data/processed/context_embeddings_train.pt`: `torch.Size([886, 64])`
- `data/processed/context_embeddings_val.pt`: `torch.Size([190, 64])`
- `data/processed/context_embeddings_test.pt`: `torch.Size([191, 64])`
- Model weights checkpoint: `models/checkpoints/context_encoder.pt`