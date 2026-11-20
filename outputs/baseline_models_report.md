# Classical Machine Learning Baseline Models Report

> **Project:** Multimodal Intelligent Triage Assistant (Educational & Research Prototype)
> **Notice:** All models evaluated strictly on the untouched Test set (N = 191).

---

## 1. Executive Summary & Purpose

Prior to developing multimodal Deep Learning architectures, establishing rigorous baseline benchmarks using classical algorithms (Logistic Regression, Random Forest, and HistGradientBoosting) is vital. These baselines delineate the predictive contributions of each clinical modality (Context, Vitals, Text, and Combinations) and establish the performance threshold that deep learning must exceed.

---

## 2. Overall Model Comparison Table (Test Set N = 191)

| Rank | Experiment | Model | Test Accuracy | Test Macro F1 | Test Weighted F1 | KTAS 1 Recall | KTAS 2 Recall | KTAS 3 Recall | KTAS 4 Recall | KTAS 5 Recall |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `Exp5_Multimodal_All` | random_forest | **0.6283** | **0.6153** | 0.6265 | 1.00 | 0.50 | 0.58 | 0.75 | 0.38 |
| 2 | `Exp4_Vitals_and_Context` | random_forest | **0.5969** | **0.5673** | 0.5800 | 1.00 | 0.38 | 0.66 | 0.71 | 0.08 |
| 3 | `Exp5_Multimodal_All` | logistic_regression | **0.5759** | **0.5495** | 0.5908 | 1.00 | 0.72 | 0.48 | 0.59 | 0.54 |
| 4 | `Exp3_Text_Only` | logistic_regression | 0.5288 | 0.4787 | 0.5524 | 0.67 | 0.56 | 0.51 | 0.50 | 0.69 |
| 5 | `Exp3_Text_Only` | random_forest | 0.4712 | 0.4660 | 0.5111 | 0.67 | 0.47 | 0.42 | 0.49 | 0.62 |
| 6 | `Exp2_Vitals_Only` | random_forest | 0.5131 | 0.4520 | 0.4962 | 1.00 | 0.31 | 0.52 | 0.66 | 0.00 |
| 7 | `Exp4_Vitals_and_Context` | logistic_regression | 0.4031 | 0.3707 | 0.4252 | 1.00 | 0.53 | 0.25 | 0.49 | 0.23 |
| 8 | `Exp5_Multimodal_All` | hist_gradient_boosting | 0.5445 | 0.3645 | 0.5564 | 0.00 | 0.50 | 0.58 | 0.63 | 0.08 |
| 9 | `Exp2_Vitals_Only` | logistic_regression | 0.3874 | 0.3547 | 0.3756 | 1.00 | 0.47 | 0.13 | 0.55 | 0.38 |
| 10 | `Exp4_Vitals_and_Context` | hist_gradient_boosting | 0.5026 | 0.3250 | 0.5204 | 0.00 | 0.44 | 0.55 | 0.59 | 0.00 |
| 11 | `Exp2_Vitals_Only` | hist_gradient_boosting | 0.3770 | 0.3131 | 0.3791 | 0.67 | 0.38 | 0.31 | 0.47 | 0.08 |
| 12 | `Exp1_Context_Only` | random_forest | 0.4293 | 0.2862 | 0.4306 | 0.00 | 0.16 | 0.37 | 0.63 | 0.31 |
| 13 | `Exp1_Context_Only` | hist_gradient_boosting | 0.3455 | 0.2368 | 0.3628 | 0.00 | 0.19 | 0.34 | 0.45 | 0.23 |
| 14 | `Exp1_Context_Only` | logistic_regression | 0.2356 | 0.1694 | 0.2510 | 0.33 | 0.12 | 0.04 | 0.45 | 0.23 |

---

## 3. Key Clinical & Modality Findings

### 3.1. Modality Performance Hierarchy
- **Multimodal Superiority (Exp 5 vs Exp 1-4):** Combining Text + Vitals + Context yields the highest overall Macro F1 and balanced classification, proving that no single modality alone captures the full clinical spectrum.
- **Vitals vs Context (Exp 2 vs Exp 1):** Physiological vital signs provide substantially higher discriminatory power for acute deterioration than demographic context alone.
- **Text Modality (Exp 3):** TF-IDF on short chief complaints alone achieves modest performance, but provides crucial semantic indicators (e.g. 'chest pain', 'dyspnea') that boost sensitivity when fused with vitals.

### 3.2. Critical Acuity Sensitivity (KTAS Level 1 & 2)
- In emergency triage, **failing to detect high-acuity patients (under-triage of KTAS 1 and 2)** has severe clinical consequences.
- Because KTAS 1 accounts for only ~1.5% of arrivals, standard accuracy fails to penalize false negatives on resuscitation cases.
- Using `class_weight='balanced'` enabled classical models to achieve non-zero recall on rare KTAS Level 1 cases, setting a clear benchmark for deep learning.

---

## 4. Benchmark Thresholds Established for Deep Learning

- **Best Baseline Architecture:** `Exp5_Multimodal_All` with `random_forest`
- **Baseline Test Accuracy Target:** `0.6283`
- **Baseline Test Macro F1 Target:** `0.6153`
- **Core Target for Deep Learning:** Outperform simple TF-IDF with semantic sentence representations (MiniLM), regularized MLP encoders, and adaptive reliability-weighted fusion.