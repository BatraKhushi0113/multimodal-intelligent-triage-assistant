# Dataset Quality & Exploratory Analysis Report

> **Project:** Multimodal Intelligent Triage Assistant (Educational & Research Prototype)
> **Notice:** This assessment does not perform clinical diagnosis or replace professional judgment.

---

## 1. Executive Summary

- **Total Records:** `1,267`
- **Total Features:** `24`
- **Exact Duplicate Rows:** `0`
- **Columns with Missing/Placeholder Data:** `8` of `24`
- **Target Variable (`KTAS_RN`):** Present, 5 discrete severity classes (1 to 5)

---

## 2. Column Specifications & Inferred Types

| Column | Raw Type | Non-Null | Missing (%) | Sample Values |
| :--- | :--- | :--- | :--- | :--- |
| `Group` | `int64` | 1,267 | 0.0% | 2, 1 |
| `Sex` | `int64` | 1,267 | 0.0% | 2, 1 |
| `Age` | `int64` | 1,267 | 0.0% | 71, 56, 68 |
| `Patients number per hour` | `int64` | 1,267 | 0.0% | 3, 12, 8 |
| `Arrival mode` | `int64` | 1,267 | 0.0% | 3, 2, 1 |
| `Injury` | `int64` | 1,267 | 0.0% | 2, 1 |
| `Chief_complain` | `object` | 1,267 | 0.0% | right ocular pain, right forearm burn, arm pain, Lt |
| `Mental` | `int64` | 1,267 | 0.0% | 1, 2, 3 |
| `Pain` | `int64` | 1,267 | 0.0% | 1, 0 |
| `NRS_pain` | `object` | 1,267 | 0.0% | 2, 3, 4 |
| `SBP` | `object` | 1,267 | 0.0% | 160, 137, 130 |
| `DBP` | `object` | 1,267 | 0.0% | 100, 75, 80 |
| `HR` | `object` | 1,267 | 0.0% | 84, 60, 102 |
| `RR` | `object` | 1,267 | 0.0% | 18, 20, 16 |
| `BT` | `object` | 1,267 | 0.0% | 36.6, 36.5, 38.1 |
| `Saturation` | `object` | 579 | 54.3% | 100, 98, 97 |
| `KTAS_RN` | `int64` | 1,267 | 0.0% | 2, 4, 3 |
| `Diagnosis in ED` | `object` | 1,265 | 0.16% | Corneal abrasion, Burn of hand, firts degree dorsum, Fracture of surgical neck of humerus, closed |
| `Disposition` | `int64` | 1,267 | 0.0% | 1, 2, 5 |
| `KTAS_expert` | `int64` | 1,267 | 0.0% | 4, 5, 3 |
| `Error_group` | `int64` | 1,267 | 0.0% | 2, 4, 6 |
| `Length of stay_min` | `int64` | 1,267 | 0.0% | 86, 64, 862 |
| `KTAS duration_min` | `object` | 1,267 | 0.0% | 5,00, 3,95, 1,00 |
| `mistriage` | `int64` | 1,267 | 0.0% | 1, 2, 0 |

---

## 3. Missingness & Non-Standard Placeholders

Emergency clinical datasets frequently contain non-standard missingness tokens (e.g., `??`, `#BOŞ!`, unrecorded observations) rather than standard empty values:

| Column | Standard NaN | Masked Placeholders | Effective Missing | Missing (%) |
| :--- | :--- | :--- | :--- | :--- |
| `Saturation` | 688 | 9 | 697 | **55.01%** |
| `DBP` | 0 | 29 | 29 | **2.29%** |
| `SBP` | 0 | 25 | 25 | **1.97%** |
| `RR` | 0 | 22 | 22 | **1.74%** |
| `HR` | 0 | 20 | 20 | **1.58%** |
| `BT` | 0 | 18 | 18 | **1.42%** |
| `Chief_complain` | 0 | 5 | 5 | **0.39%** |
| `Diagnosis in ED` | 2 | 0 | 2 | **0.16%** |

### Key Missingness Findings:
- **Saturation (`Saturation`):** 688 standard NaNs and 9 placeholder (`??`) tokens, resulting in **55.01% missingness**. In emergency triage, oxygen saturation is often only measured when a patient presents with respiratory, cardiovascular, or altered mental complaints.
- **NRS Pain Score (`NRS_pain`):** 556 entries contain the Turkish spreadsheet error token `#BOŞ!` (meaning `#EMPTY!`). Cross-referencing reveals that when `Pain == 0` (no pain), `NRS_pain` is systematically unrecorded.
- **Vitals (`SBP`, `DBP`, `HR`, `RR`, `BT`):** Between 18 and 29 records contain string placeholders `??` indicating unobtained vital measurements.

---

## 4. KTAS Acuity Target Distribution (`KTAS_RN`)

| KTAS Level | Clinical Meaning | Patient Count | Percentage |
| :---: | :--- | :---: | :---: |
| **1** | Resuscitation (Immediately Life-Threatening) | 18 | 1.42% |
| **2** | Emergent (Potential Threat to Life/Limb) | 214 | 16.89% |
| **3** | Urgent (Condition with High Risk of Deterioration) | 447 | 35.28% |
| **4** | Less Urgent (Low Risk, Stable) | 501 | 39.54% |
| **5** | Non-Urgent (Acute Minor or Chronic Condition) | 87 | 6.87% |

> **Imbalance Observation:** Severe acute presentations (KTAS 1) account for only **1.42%** of admissions. Stratified sampling and cost-sensitive loss functions will be essential during modeling.

---

## 5. Suspicious Values & Physiological Audits

| Audit Category | Detected Anomaly / Outlier | Impact / Clinical Interpretation |
| :--- | :--- | :--- |
| `SBP_non_numeric_tokens` | Tokens: `['??']` (Count: 25) | Requires imputation or type casting in Preprocessing. |
| `DBP_non_numeric_tokens` | Tokens: `['??']` (Count: 29) | Requires imputation or type casting in Preprocessing. |
| `HR_non_numeric_tokens` | Tokens: `['??']` (Count: 20) | Requires imputation or type casting in Preprocessing. |
| `RR_non_numeric_tokens` | Tokens: `['??']` (Count: 22) | Requires imputation or type casting in Preprocessing. |
| `BT_non_numeric_tokens` | Tokens: `['??']` (Count: 18) | Requires imputation or type casting in Preprocessing. |
| `Saturation_non_numeric_tokens` | Tokens: `['nan', '??']` (Count: 697) | Requires imputation or type casting in Preprocessing. |
| `NRS_pain_non_numeric_tokens` | Tokens: `['#BOÞ!']` (Count: 556) | Requires imputation or type casting in Preprocessing. |
| `SBP_physiological_outliers` | 3 records outside [60, 240] (Min: 50.0, Max: 275.0) | Extreme emergency values (e.g. severe hypoxemia SpO2=20% or SBP=275 mmHg). |
| `DBP_physiological_outliers` | 6 records outside [40, 140] (Min: 31.0, Max: 160.0) | Extreme emergency values (e.g. severe hypoxemia SpO2=20% or SBP=275 mmHg). |
| `HR_physiological_outliers` | 5 records outside [40, 180] (Min: 32.0, Max: 148.0) | Extreme emergency values (e.g. severe hypoxemia SpO2=20% or SBP=275 mmHg). |
| `Saturation_physiological_outliers` | 2 records outside [70, 100] (Min: 20.0, Max: 100.0) | Extreme emergency values (e.g. severe hypoxemia SpO2=20% or SBP=275 mmHg). |
| `pain_inconsistency_count` | Count: `0` | Data consistency check. |
| `inverted_bp_count` | Count: `0` | Data consistency check. |

---

## 6. Scientific Constraints & Recommendations for Module 2 and 3

1. **No Preprocessing Yet:** The data remains unmutated; raw types and token variations are mapped.
2. **Leakage Analysis Required (Module 2):** Variables such as `Diagnosis in ED`, `Disposition`, `Length of stay_min`, `KTAS duration_min`, `KTAS_expert`, and `mistriage` are post-admission artifacts and must be formally excluded.
3. **Missingness Indicator Strategy (Module 3):** Since SpO2 missingness is clinically informative (not random missingness), binary indicator columns (`Saturation_is_missing`) should be created alongside robust median/MICE imputation.
4. **Class Weighting:** KTAS Level 1 represents a rare positive class requiring macro-F1 and cost-weighted evaluation.