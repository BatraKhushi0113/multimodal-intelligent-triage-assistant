# Deep Learning Vitals Encoder Report

> **Project:** Multimodal Intelligent Triage Assistant (Educational & Research Prototype)
> **Modality:** Static Physiological Vital Signs (`SBP`, `DBP`, `HR`, `RR`, `BT`, `Saturation`, `Mental`, `Pain`, `NRS_pain`)

---

## 1. Architectural Specification

```
Input Features (Dimension: 10)
  ├── Continuous Vitals (7): SBP, DBP, HR, RR, BT, Saturation, NRS_pain
  ├── Missingness Indicator (1): Saturation_is_missing
  └── Clinical Discrete Indicators (2): Mental_raw (AVPU), Pain_flag
        ↓
Linear Layer (10 → 64)
        ↓
BatchNorm1d(64) + GELU Activation
        ↓
Dropout (p = 0.20)
        ↓
Linear Projection (64 → 128)
        ↓
LayerNorm(128) + GELU Activation
        ↓
Dropout (p = 0.20)
        ↓
Vitals Latent Vector (d_vitals = 128)
```

---

## 2. Unimodal Performance on Unseen Test Set (N = 191)

| Metric | Value | Clinical Benchmark Context |
| :--- | :---: | :--- |
| **Test Accuracy** | **0.4503** | Pure vitals classification acuity |
| **Test Macro F1** | **0.3808** | Balanced across all 5 KTAS levels |
| **Test Weighted F1** | **0.4570** | Overall population weighted F1 |
| **KTAS 1 Recall (Resuscitation)** | **0.67** | Critical sensitivity on most acute tier |
| **KTAS 2 Recall (Emergent)** | **0.53** | Sensitivity on emergent presentations |
| **KTAS 3 Recall (Urgent)** | **0.30** | Sensitivity on urgent presentations |
| **KTAS 4 Recall (Less Urgent)** | **0.57** | Sensitivity on stable presentations |
| **KTAS 5 Recall (Non-Urgent)** | **0.31** | Sensitivity on minor presentations |

---

## 3. Scientific Justification for Static MLP vs Temporal Recurrent Models

1. **Nature of Emergency Intake Vitals:** Emergency triage records in this dataset represent **single snapshot measurements** taken during initial triage arrival, not continuous bedside ICU time-series telemetry.
2. **No Temporal Dimension:** Applying RNN, LSTM, or temporal Transformers to static 1-point snapshots would impose fictitious sequential assumptions.
3. **Regularization for Compact Datasets:** The 2-stage MLP incorporates BatchNorm, LayerNorm, GELU, and 20% Dropout to constrain parameter complexity on 886 training records.

---

## 4. Cached Latent Embeddings

- `data/processed/vitals_embeddings_train.pt`: `torch.Size([886, 128])`
- `data/processed/vitals_embeddings_val.pt`: `torch.Size([190, 128])`
- `data/processed/vitals_embeddings_test.pt`: `torch.Size([191, 128])`
- Model weights checkpoint: `models/checkpoints/vitals_encoder.pt`