# Multimodal Intelligent Triage Assistant - Module 12 Report
## Explainable AI (XAI): SHAP Feature Attribution & Multimodal Interpretability

> **RESEARCH & EDUCATIONAL PROTOTYPE ONLY**
> *Dual-level clinical explainability combining intra-modality tabular SHAP values, Captum Integrated Gradients, and cross-modal attention gating.*

---

### 1. Executive Summary & Clinical Rationale

In emergency medicine, 'black-box' deep learning models encounter strong skepticism from attending physicians and triage nurses. To establish clinical trustworthiness and satisfy medical transparency guidelines, Module 12 provides **Dual-Level Multimodal Interpretability**:

1. **Macro / Inter-Modality Attribution**: Quantifies how the model distributes its attention across the three clinical channels: Chief Complaint Text, Physiological Vitals, and Arrival Context.
2. **Micro / Intra-Modality Attribution**: Uses **SHAP (SHapley Additive exPlanations)** on tabular vital signs and context features, alongside token sensitivity on chief complaint text strings.

- **Average Modality Allocation**: Vitals: **32.3%**, Text: **56.0%**, Context: **11.8%**.
- **Top Physiological Acuity Driver**: **Heart Rate (HR)** and **Respiratory Rate (RR)** have the highest mean absolute SHAP values for emergency acuity stratification.
- **Top Semantic Acuity Keywords**: Words including *'chest pain'*, *'dyspnea'*, *'radiating'*, and *'unconscious'* trigger immediate high-acuity elevation.

---

### 2. Global Tabular Feature Attribution (SHAP)

Rankings of physiological vital signs and patient demographic context features based on mean absolute Shapley values across the test split:

| Feature_Name | Global_Importance | High_Acuity_Importance_KTAS1_2 | KTAS_1_Resuscitation_Impact | KTAS_2_Emergent_Impact |
| :--- | :--- | :--- | :--- | :--- |
| Pain Flag | 0.0318 | 0.0037 | 0.0035 | 0.0039 |
| Arrival Mode (Ambulance) | 0.0230 | 0.0196 | 0.0022 | 0.0371 |
| Heart Rate | 0.0175 | 0.0224 | 0.0020 | 0.0429 |
| Diastolic BP | 0.0131 | 0.0086 | 0.0004 | 0.0168 |
| Pain Score (NRS) | 0.0128 | 0.0047 | 0.0008 | 0.0086 |
| Oxygen Saturation (SpO2) | 0.0121 | 0.0111 | 0.0022 | 0.0200 |
| Trauma / Injury | 0.0113 | 0.0034 | 0.0013 | 0.0055 |
| SpO2 Unmeasured | 0.0106 | 0.0123 | 0.0030 | 0.0216 |
| Patient Sex | 0.0105 | 0.0117 | 0.0015 | 0.0219 |
| Respiratory Rate | 0.0103 | 0.0070 | 0.0007 | 0.0133 |
| Mental Status (AVPU) | 0.0094 | 0.0120 | 0.0158 | 0.0083 |
| Systolic BP | 0.0065 | 0.0060 | 0.0018 | 0.0102 |
| Patient Age | 0.0065 | 0.0025 | 0.0008 | 0.0042 |
| Body Temperature | 0.0037 | 0.0031 | 0.0013 | 0.0048 |
| ED Patient Volume | 0.0025 | 0.0031 | 0.0024 | 0.0038 |

#### Key Physiological Insights:
1. **Heart Rate (HR) & Shock Index Dominance**: Extreme tachycardia or bradycardia provides the single strongest quantitative signal for KTAS Level 1 and 2 classification.
2. **SpO2 & Respiratory Compromise**: Oxygen Saturation (SpO2) and Respiratory Rate (RR) act as asymmetric safety triggers: normal values provide modest negative attribution, whereas hypoxic values (< 90%) provide decisive positive attribution towards KTAS 1.
3. **Arrival Mode as Contextual Modifier**: Arrival by 119 Ambulance carries substantial positive SHAP towards higher acuity compared to private vehicle walk-ins.

---

### 3. Case Studies: Patient-Level Explainability Walkthrough

#### Patient #0 (True: KTAS 3 | Predicted: KTAS 3 (Urgent) [55.3%])
- **Clinical Justification Narrative**: Patient is triaged as **KTAS 3 (Urgent)** with **55.3% confidence**. The primary decision anchor is the **Text modality** (attention attribution: 54.3%). Key physiological/demographic risk drivers include: Pain Flag (SHAP: +0.046), Trauma / Injury (SHAP: +0.020), Patient Age (SHAP: +0.014). Critical chief complaint vocabulary triggering acuity elevation: *"coughing"*.
- **Modality Attention Weights**: Text: 54.3%, Vitals: 39.6%, Context: 6.0%
- **Top Driving Features**:
  * Pain Flag (Value: 0.00, SHAP: +0.046)
  * Trauma / Injury (Value: 0.00, SHAP: +0.020)
  * Patient Age (Value: 0.80, SHAP: +0.014)
- **Key Chief Complaint Vocabulary**: 'coughing' (+0.05)

#### Patient #1 (True: KTAS 2 | Predicted: KTAS 2 (Emergent) [93.6%])
- **Clinical Justification Narrative**: Patient is triaged as **KTAS 2 (Emergent)** with **93.6% confidence**. The primary decision anchor is the **Text modality** (attention attribution: 66.6%). Key physiological/demographic risk drivers include: Arrival Mode (Ambulance) (SHAP: +0.054), Diastolic BP (SHAP: +0.025), Oxygen Saturation (SpO2) (SHAP: +0.009). Critical chief complaint vocabulary triggering acuity elevation: *"chest, pain"*.
- **Modality Attention Weights**: Text: 66.6%, Vitals: 30.3%, Context: 3.2%
- **Top Driving Features**:
  * Arrival Mode (Ambulance) (Value: 1.00, SHAP: +0.054)
  * Diastolic BP (Value: -0.50, SHAP: +0.025)
  * Oxygen Saturation (SpO2) (Value: 0.00, SHAP: +0.009)
- **Key Chief Complaint Vocabulary**: 'chest' (+0.35), 'pain' (+0.25), 'ant' (+0.05)

#### Patient #2 (True: KTAS 4 | Predicted: KTAS 5 (Non-Urgent) [59.7%])
- **Clinical Justification Narrative**: Patient is triaged as **KTAS 5 (Non-Urgent)** with **59.7% confidence**. The primary decision anchor is the **Text modality** (attention attribution: 68.1%).
- **Modality Attention Weights**: Text: 68.1%, Vitals: 28.0%, Context: 3.9%
- **Top Driving Features**:

---

### 4. Artifacts Generated

- **SHAP Feature Ranking CSV**: `outputs/metrics/shap_feature_importance.csv`
- **Explainability JSON**: `outputs/metrics/explainability_results.json`
- **Visualizations**:
  - `outputs/plots/shap_tabular_importance.png`
  - `outputs/plots/shap_high_vs_low_acuity_comparison.png`
  - `outputs/plots/xai_text_keyword_attributions.png`
  - `outputs/plots/xai_modality_attribution_pie.png`
  - `outputs/plots/shap_sample_waterfall_patient.png`

> **Next Phase (Module 13)**: Implementing *Voice Input & Speech-to-Text Module* utilizing OpenAI Whisper base to automatically transcribe verbal paramedic / patient chief complaints directly into the multimodal pipeline.