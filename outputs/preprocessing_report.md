# Emergency Triage Preprocessing & Stratified Splitting Report

> **Project:** Multimodal Intelligent Triage Assistant (Educational & Research Prototype)
> **Notice:** All scalers, imputers, and encoders were fit strictly on the Training set to prevent data leakage.

---

## 1. Split Dimensions & Stratification

- **Total Patients:** `1,267`
- **Train Set:** `886` records (69.9%)
- **Validation Set:** `190` records (15.0%)
- **Test Set:** `191` records (15.1%)

### Stratified Acuity Target Distribution Across Splits (`KTAS_RN`)

| KTAS Level | Acuity Description | Train Count (%) | Val Count (%) | Test Count (%) | Total |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Level 1 | 12 (1.4%) | 3 (1.6%) | 3 (1.6%) | 18 |
| **2** | Level 2 | 150 (16.9%) | 32 (16.8%) | 32 (16.8%) | 214 |
| **3** | Level 3 | 313 (35.3%) | 67 (35.3%) | 67 (35.1%) | 447 |
| **4** | Level 4 | 350 (39.5%) | 75 (39.5%) | 76 (39.8%) | 501 |
| **5** | Level 5 | 61 (6.9%) | 13 (6.8%) | 13 (6.8%) | 87 |

---

## 2. Imputation Parameters Learned Strictly from Train Set

| Feature | Learned Imputation Value | Clinical Justification |
| :--- | :--- | :--- |
| `SBP` | `130.00` | Train median replacement for string placeholders ('??'). |
| `DBP` | `80.00` | Train median replacement for string placeholders ('??'). |
| `HR` | `82.00` | Train median replacement for string placeholders ('??'). |
| `RR` | `20.00` | Train median replacement for string placeholders ('??'). |
| `BT` | `36.50` | Train median replacement for string placeholders ('??'). |
| `Saturation` | `98.00` | Median SpO2 for present vitals. Complemented by `Saturation_is_missing` indicator. |
| `NRS_pain_positive_median` | `4.00` | Applied only to patients with Pain=1 who lacked NRS score. Patients with Pain=0 receive 0. |

---

## 3. Processed Feature Specification Matrix

Total Processed Features: `18`

| Processed Column | Source Feature | Modality | Transformation Applied |
| :--- | :--- | :--- | :--- |
| `Saturation_is_missing` | `Saturation_is_missing` | Vitals | Binary Clinical Missingness Indicator (0/1) |
| `SBP_scaled` | `SBP` | Vitals | Robust / Standard Scaling |
| `DBP_scaled` | `DBP` | Vitals | Robust / Standard Scaling |
| `HR_scaled` | `HR` | Vitals | Robust / Standard Scaling |
| `RR_scaled` | `RR` | Vitals | Robust / Standard Scaling |
| `BT_scaled` | `BT` | Vitals | Robust / Standard Scaling |
| `Saturation_scaled` | `Saturation` | Vitals | Robust / Standard Scaling |
| `NRS_pain_scaled` | `NRS_pain` | Vitals | Robust / Standard Scaling |
| `Mental_raw` | `Mental_raw` | Vitals | Discrete Acuity Indicator |
| `Pain_flag` | `Pain_flag` | Vitals | Discrete Acuity Indicator |
| `Age_scaled` | `Age` | Context | Robust / Standard Scaling |
| `Patients_per_hour_scaled` | `Patients_per_hour` | Context | Robust / Standard Scaling |
| `Sex_cat` | `Sex` | Context | Discrete Integer Category Index |
| `Arrival_mode_cat` | `Arrival_mode` | Context | Discrete Integer Category Index |
| `Injury_cat` | `Injury` | Context | Discrete Integer Category Index |
| `Chief_complain_clean` | `Chief_complain` | Text | Lowercased & Missing-Handled Text |
| `KTAS_RN_raw` | `KTAS_RN_raw` | Target | Discrete Acuity Indicator |
| `KTAS_RN_encoded` | `KTAS_RN_encoded` | Target | Zero-Indexed ML Target (0 to 4) |

---

## 4. Leakage Prevention Verification

1. **Post-Admission Artifacts:** Verified zero presence of `Diagnosis in ED`, `Disposition`, `KTAS_expert`, `Error_group`, `Length of stay_min`, `KTAS duration_min`, `mistriage`, or `Group` in processed files.
2. **Distribution Stability:** Train, Val, and Test maintain identical KTAS acuity proportions, preventing evaluation bias on rare acuity classes (KTAS 1).
3. **Saved Pipeline:** Serialized to `models/checkpoints/preprocessor.joblib` for prospective inference.