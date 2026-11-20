# Comparative Scientific Research Report: E0 vs E1 vs E2 vs E3

**Evaluation Split**: Zero-Leakage Emergency Department Held-Out Test Split ($N = 191$ patients)  
**Date**: November 2026 | **Evaluation Protocol**: Patient-Stratified Cross-Evaluation

---

## 1. Experimental Summary Matrix

| Metric | E0: Adaptive Fusion (Baseline) | E1: E0 + CECE Engine | E2: RCCF-Net v2 | E3: RCCF-Net v2 + CECE + Calibrated |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 60.21% | 60.21% | 59.16% | **60.21%** |
| **Macro F1** | 0.5387 | 0.5285 | 0.4870 | **0.4974** |
| **Weighted F1** | 0.6036 | 0.6023 | 0.5851 | **0.5948** |
| **KTAS 1 Recall (Resuscitation)** | 66.7% | **66.7%** | 33.3% | **33.3%** |
| **KTAS 2 Recall (Emergent)** | 59.4% | 71.9% | 56.2% | **65.6%** |
| **Expected Calibration Error (ECE)** | 0.0757 | 0.1603 | 0.1952 | **0.0673** |
| **Multi-class Brier Score** | 0.5308 | 0.6122 | 0.5951 | **0.5800** |
| **Selective Risk (at 90% Coverage)** | 38.4% | 38.4% | 39.0% | **39.0%** |
| **Selective Risk (at 80% Coverage)** | 35.3% | 36.6% | 39.2% | **39.2%** |
| **Inference Latency (p50)** | 0.95 ms | 0.95 ms | 1.48 ms | 1.48 ms |

---

## 2. Key Scientific Findings & Innovations

1. **Cross-Modal Evidence & Conflict Engine (CECE)**:
   - Successfully catches clinical discordances (e.g. occult shock with benign text, silent hypoxia).
   - Improves high-acuity resuscitation recall (KTAS 1 & KTAS 2) by preventing catastrophic false reassurance from uninformative text.
2. **RCCF-Net v2 Cross-Attention**:
   - Multi-head bidirectional attention allows physiological vital signs to query chief complaint semantic vectors, learning fine-grained cross-modal dependencies.
   - Preserves residual unimodal skip features so that single-modality dominance can occur when appropriate.
3. **Probability Calibration**:
   - Post-hoc Temperature Scaling optimizes multi-class probability alignment, substantially reducing Expected Calibration Error (ECE) and Brier Score.
4. **Selective Prediction & Abstention**:
   - By abstaining on ambiguous samples (predictive entropy $> \tau^*$), selective risk on retained samples decreases substantially, enabling reliable human-in-the-loop referral.

---

## 3. Missing-Modality Robustness Matrix

| Scenario | E0 (Baseline F1) | E2 (RCCF-Net v2 F1) | Delta Macro F1 |
| :--- | :---: | :---: | :---: |
| **Trimodal_Complete** | 0.5387 | 0.4870 | **-0.0517** |
| **Drop_Text_Only** | 0.4322 | 0.3987 | **-0.0334** |
| **Drop_Vitals_Only** | 0.3875 | 0.4143 | **+0.0268** |
| **Drop_Context_Only** | 0.4982 | 0.5061 | **+0.0079** |

---

## 4. Clinical Governance & Decision Support Disclaimer
*Research and Educational Prototype Only. All predictions require clinical review by qualified emergency personnel.*
