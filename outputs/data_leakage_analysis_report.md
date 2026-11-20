# Clinical Data Leakage & Feature Selection Analysis Report

> **Project:** Multimodal Intelligent Triage Assistant (Educational & Research Prototype)
> **Objective:** Enforce strict temporal separation between intake presentation vs post-workup clinical data.

---

## 1. Executive Summary & Clinical Framework

In emergency medicine, **triage** is performed within minutes of patient arrival at the hospital. Its sole clinical purpose is to assign an acuity score (e.g., Canadian Triage and Acuity Scale - KTAS Levels 1 to 5) to determine the safe waiting time before a patient is seen by an emergency physician.

A common pitfall in machine learning for emergency medicine is **data leakage**: including features in the model that are only recorded during physician diagnostic evaluation, treatment, hospital admission, or post-hoc auditing. A model trained with such features will exhibit deceptively high accuracy in offline validation but will catastrophically fail in real-time prospective deployment because those features **do not exist** when the patient enters the triage booth.

---

## 2. Complete Column Taxonomy Classification

Total Columns Audited: `24`

| Category | Feature Name | Clinical Subsystem | Temporal Capture Point | Status |
| :--- | :--- | :--- | :--- | :---: |
| **A. SAFE INPUT** | `Chief_complain` | NLP / Text | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `SBP` | Physiological Vitals | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `DBP` | Physiological Vitals | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `HR` | Physiological Vitals | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `RR` | Physiological Vitals | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `BT` | Physiological Vitals | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `Saturation` | Physiological Vitals | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `Mental` | Physiological Vitals | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `Pain` | Physiological Vitals | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `NRS_pain` | Physiological Vitals | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `Sex` | Patient Context / Demographics | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `Age` | Patient Context / Demographics | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `Arrival mode` | Patient Context / Demographics | Triage Booth (Intake) | **INCLUDED** |
| **A. SAFE INPUT** | `Injury` | Patient Context / Demographics | Triage Booth (Intake) | **INCLUDED** |
| **B. POTENTIALLY UNSAFE** | `Patients number per hour` | Operational / Protocol | Hospital System / Study Setup | **REQUIRES CAUTION** |
| **B. POTENTIALLY UNSAFE** | `Group` | Operational / Protocol | Hospital System / Study Setup | **REQUIRES CAUTION** |
| **C. DATA LEAKAGE** | `Diagnosis in ED` | Post-Workup / Audit | Post-Triage / Discharge / Audit | **STRICTLY EXCLUDED** |
| **C. DATA LEAKAGE** | `Disposition` | Post-Workup / Audit | Post-Triage / Discharge / Audit | **STRICTLY EXCLUDED** |
| **C. DATA LEAKAGE** | `KTAS_expert` | Post-Workup / Audit | Post-Triage / Discharge / Audit | **STRICTLY EXCLUDED** |
| **C. DATA LEAKAGE** | `Error_group` | Post-Workup / Audit | Post-Triage / Discharge / Audit | **STRICTLY EXCLUDED** |
| **C. DATA LEAKAGE** | `Length of stay_min` | Post-Workup / Audit | Post-Triage / Discharge / Audit | **STRICTLY EXCLUDED** |
| **C. DATA LEAKAGE** | `KTAS duration_min` | Post-Workup / Audit | Post-Triage / Discharge / Audit | **STRICTLY EXCLUDED** |
| **C. DATA LEAKAGE** | `mistriage` | Post-Workup / Audit | Post-Triage / Discharge / Audit | **STRICTLY EXCLUDED** |
| **TARGET** | `KTAS_RN` | Acuity Ground Truth | Triage Assessment | **PREDICTION TARGET** |

---

## 3. Quantitative Leakage Demonstrations

The following empirical correlations demonstrate the degree of direct target contamination introduced by the excluded features:

| Excluded Feature | Metric Type | Empirical Value | Sample Size | Clinical Contamination Mechanism |
| :--- | :--- | :--- | :---: | :--- |
| `Diagnosis in ED` | Categorical Cardinality | `583 unique diagnoses` | 1,265 | Definitive Medical Diagnosis |
| `Disposition` | Spearman Rank Correlation | `rho = -0.389 (p < 0.001)` | 1,267 | Outcome Metric |
| `KTAS_expert` | Spearman Rank Correlation | `rho = 0.876 (p < 0.001)` | 1,267 | Secondary Ground Truth / Expert Consensus |
| `Error_group` | Spearman Rank Correlation | `rho = 0.081 (p < 0.001)` | 1,267 | Audit Discrepancy Code |
| `Length of stay_min` | Spearman Rank Correlation | `rho = -0.289 (p < 0.001)` | 1,267 | Operational Duration |
| `KTAS duration_min` | Spearman Rank Correlation | `rho = 0.071 (p < 0.001)` | 1,267 | Clinical Process Artifact |
| `mistriage` | Spearman Rank Correlation | `rho = 0.093 (p < 0.001)` | 1,267 | Ground-Truth Mistriage Indicator |

---

## 4. In-Depth Scientific Rationale for Excluded Features

### 4.1. `Diagnosis in ED`
- **Classification:** `DATA LEAKAGE`
- **Capture Timeline:** `Post-Workup / Discharge`
- **Clinical Category:** `Definitive Medical Diagnosis`
- **Scientific Justification:**
  > The emergency physician's diagnosis is established only after comprehensive history-taking, physical examination, point-of-care laboratory tests (troponin, arterial blood gas, electrolytes), and medical imaging (CT, X-ray, ultrasound). Using this feature in an intake triage model represents severe circular reasoning: the goal of triage is to prioritize who receives a diagnostic workup first.

### 4.2. `Disposition`
- **Classification:** `DATA LEAKAGE`
- **Capture Timeline:** `Post-Treatment / Departure`
- **Clinical Category:** `Outcome Metric`
- **Scientific Justification:**
  > Patient disposition (discharge home, admission to general ward, emergency surgery, or ICU transfer) is determined hours after triage when response to acute treatment has been observed. Conditioning triage decisions on future hospitalization outcomes is an impossible temporal leak.

### 4.3. `KTAS_expert`
- **Classification:** `DATA LEAKAGE`
- **Capture Timeline:** `Retrospective Audit`
- **Clinical Category:** `Secondary Ground Truth / Expert Consensus`
- **Scientific Justification:**
  > Assigned retrospectively by a panel of expert emergency physicians who reviewed the entire complete patient medical record. Serving this to the model as an input would bypass the entire triage task by feeding an alternative answer key.

### 4.4. `Error_group`
- **Classification:** `DATA LEAKAGE`
- **Capture Timeline:** `Retrospective Audit`
- **Clinical Category:** `Audit Discrepancy Code`
- **Scientific Justification:**
  > Categorizes whether and why the registered nurse's triage score differed from the expert review panel. It is mathematically and logically dependent on both KTAS_RN and KTAS_expert, creating direct target leakage.

### 4.5. `Length of stay_min`
- **Classification:** `DATA LEAKAGE`
- **Capture Timeline:** `Post-Discharge`
- **Clinical Category:** `Operational Duration`
- **Scientific Justification:**
  > The total number of minutes the patient occupied an ED bed before departure. This metric can only be computed at discharge time. Furthermore, length of stay varies widely based on hospital bed availability, not merely acute triage acuity.

### 4.6. `KTAS duration_min`
- **Classification:** `DATA LEAKAGE`
- **Capture Timeline:** `End of Triage Consultation`
- **Clinical Category:** `Clinical Process Artifact`
- **Scientific Justification:**
  > The time taken by the nurse to evaluate the patient and enter the triage rating. At the onset of automated or AI-assisted triage, this duration does not exist. While longer triage times correlate with ambiguous or difficult presentations, it is a meta-process artifact rather than an intrinsic patient attribute.

### 4.7. `mistriage`
- **Classification:** `DATA LEAKAGE`
- **Capture Timeline:** `Retrospective Audit`
- **Clinical Category:** `Ground-Truth Mistriage Indicator`
- **Scientific Justification:**
  > A categorical flag (0: Correct, 1: Under-triage, 2: Over-triage) generated post-hoc by comparing KTAS_RN with KTAS_expert. Incorporating this into the feature matrix would directly expose the target's validity.

---

## 5. In-Depth Justification for Potentially Unsafe Features

### 5.1. `Patients number per hour`
- **Context:** Represents the instantaneous census / patient arrival volume per hour in the emergency department.
- **Assessment:** Although technically available from electronic ED tracking boards at arrival, this metric captures hospital crowding and staffing stress rather than the patient's biological distress. If triage nurses systematically down-triaged or up-triaged patients during peak volume shifts, a model relying heavily on this feature would learn institutional congestion heuristics rather than pathophysiological signs. It is retained as a candidate context feature for ablation testing, but flagged as potentially unsafe.

### 5.2. `Group`
- **Context:** Binary indicator (Group 1 vs Group 2) representing distinct study arms or administrative cohorts in the underlying hospital research protocol.
- **Assessment:** A patient's study cohort assignment has zero physiological relevance and is an artificial artifact of the dataset collection design. Including it would allow the model to exploit cohort-specific target distribution shifts. It must be excluded from feature vectors.

---

## 6. Preprocessing & Modeling Guardrails for Module 3

1. **Strict Exclusion Filter:** All `EXCLUDED_FEATURES` and `Group` will be dropped before any train/validation/test splitting.
2. **Safe Feature Matrix:** Only the 14 verified `SAFE_FEATURES` (and optionally `Patients number per hour` during context ablation) will enter the preprocessing and encoder pipelines.
3. **Zero Contamination Guarantee:** No target statistics, retrospective diagnoses, or discharge indicators will be accessible to the baseline or multimodal neural models.