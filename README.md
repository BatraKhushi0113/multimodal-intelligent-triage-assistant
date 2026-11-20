# Multimodal Intelligent Triage Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> [!WARNING]
> ### ⚠️ RESEARCH & EDUCATIONAL PROTOTYPE DISCLAIMER
> **THIS SOFTWARE IS AN EDUCATIONAL AND SCIENTIFIC RESEARCH PROTOTYPE ONLY.**
> - It is **NOT** a medical diagnostic device and does **NOT** diagnose diseases, prescribe medication, or formulate treatment plans.
> - It is **NOT** intended to replace or supersede the judgment, clinical assessment, or decision-making of licensed physicians, nurses, or emergency medical technicians.
> - All outputs, risk scores, and triage classifications are research model estimations and must be reviewed by qualified human healthcare providers.

---

## 📖 Overview

The **Multimodal Intelligent Triage Assistant** is a deep learning framework designed to explore how diverse clinical modalities—vital signs, demographic and operational context, unstructured chief complaint text, spoken audio, and chest X-rays—can be processed, fused, and calibrated under clinical uncertainty.

### Key Conceptual Architecture
```
                         USER / CLINICIAN
                               │
                               ▼
                 ┌───────────────────────────┐
                 │     MULTIMODAL INPUT      │
                 └─────────────┬─────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
       TEXT                  VITALS                CONTEXT
  (Chief Complaint)      (Physiological)        (Demographics)
         │                     │                     │
         ▼                     ▼                     ▼
    Text Encoder         Vitals Encoder        Context Encoder
  (MiniLM / TF-IDF)          (MLP)            (Embedding + MLP)
         │                     │                     │
         ▼                     ▼                     ▼
    Text Vector           Vital Vector         Context Vector
  (d_text = 384)        (d_vitals = 128)      (d_context = 64)
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                               ▼
                    RELIABILITY ESTIMATION
               (Missingness & Quality Weighting)
                               │
                               ▼
                     ADAPTIVE FUSION MODEL
                               │
                               ▼
                       TRIAGE CLASSIFIER
                               │
                               ▼
                    KTAS LEVEL PREDICTION
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       UNCERTAINTY ESTIMATION           EXPLAINABLE AI
        (Monte Carlo Dropout)         (SHAP & Attributions)
                │                             │
                └──────────────┬──────────────┘
                               │
                               ▼
                       FINAL TRIAGE REPORT
```

### Auxiliary Pipelines
- **Voice Modality**: Spoken audio $\to$ Pretrained Whisper (Speech-to-Text) $\to$ Cleaned Chief Complaint Text $\to$ Core Text Encoder.
- **Chest X-ray Modality**: Independent Vision Model (DenseNet-121 / EfficientNet-B0) with Grad-CAM visual attention. *(Scientifically decoupled from the tabular triage dataset to eliminate false patient-level pairings).*

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Deep Learning & ML** | PyTorch, torchvision, torchaudio, scikit-learn |
| **NLP & Transformers** | Hugging Face Transformers, Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Vision & Medical Imaging** | timm, OpenCV, Pillow, PyTorch Grad-CAM |
| **Audio & Speech** | OpenAI Whisper, Librosa, SoundFile |
| **Explainability (XAI)** | SHAP (Kernel/Tree Explainer), Captum (Integrated Gradients) |
| **Web UI & API** | Streamlit, FastAPI, Uvicorn, Pydantic |
| **Data & Configuration** | Pandas, NumPy, SciPy, PyYAML, Joblib |

---

## 📁 Repository Structure

```
multimodal-triage-assistant/
│
├── data/
│   ├── raw/
│   │   ├── triage/            # Emergency Service Triage dataset (CSV, semicolon delimiter)
│   │   ├── xray/              # Chest X-ray images
│   │   └── voice/             # Medical voice recordings (.wav, .mp3)
│   └── processed/             # Preprocessed train/val/test splits & cached embeddings
│
├── notebooks/                 # Exploratory data analysis & Colab notebooks
│
├── src/
│   ├── data/
│   │   ├── load_data.py       # Robust CSV ingestion & validation
│   │   ├── preprocessing.py  # Outlier detection, imputation, normalization
│   │   └── validation.py      # Schema & physiological constraint checks
│   │
│   ├── models/
│   │   ├── baseline.py        # Tabular ML baselines (Logistic Regression, RF, XGB)
│   │   ├── text_encoder.py    # MiniLM sentence embeddings & TF-IDF
│   │   ├── vitals_encoder.py  # PyTorch MLP with BatchNorm, GELU, & Dropout
│   │   ├── context_encoder.py # Categorical Entity Embeddings + Numerical MLP
│   │   ├── fusion.py          # Early concatenation & Adaptive Weighted Fusion
│   │   ├── reliability.py     # Data-driven modality reliability scoring
│   │   ├── uncertainty.py     # Monte Carlo Dropout & predictive entropy
│   │   └── multimodal_model.py# End-to-end multi-input PyTorch module
│   │
│   ├── training/
│   │   ├── train.py           # Stratified training loops with early stopping
│   │   └── evaluate.py        # Multiclass metrics, Macro F1, Confusion Matrix
│   │
│   ├── explainability/
│   │   ├── shap_analysis.py   # Tabular feature importances
│   │   └── explanations.py    # Modality attributions & report generation
│   │
│   ├── voice/
│   │   └── speech_to_text.py  # Whisper speech transcription wrapper
│   │
│   └── vision/
│       ├── xray_model.py      # Pretrained DenseNet/EfficientNet for X-rays
│       └── gradcam.py         # Visual model attention maps
│
├── models/
│   └── checkpoints/           # Saved PyTorch .pt weights & preprocessor joblibs
│
├── outputs/
│   ├── metrics/               # Evaluation tables, JSON summaries
│   ├── plots/                 # Confusion matrices, distributions, SHAP plots
│   └── predictions/           # Test set prediction outputs
│
├── app/
│   └── streamlit_app.py       # Clinical decision support research interface
│
├── tests/                     # Automated unit and shape tests
│
├── requirements.txt           # Production dependencies
├── config.yaml                # Master configuration
├── .gitignore                 # Git ignore patterns for models and large data
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 1. Local Environment Setup

#### Prerequisites
- Python 3.10 or higher
- Git
- CUDA-compatible GPU (recommended, but CPU supported)

#### Installation
```bash
# Clone the repository
git clone <repository_url>
cd "multimodal-triage-assistant"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Running on Google Colab

This repository is pre-configured to run seamlessly in Google Colab:
1. Open a new Google Colab notebook.
2. Select **Runtime** > **Change runtime type** > **GPU (T4 or higher)**.
3. Clone or upload the repository:
   ```python
   !git clone <repository_url>
   %cd multimodal-triage-assistant
   !pip install -r requirements.txt
   ```
4. Access `config.yaml` to adjust device settings to `cuda`.

---

## 🗺️ Project Modules & Roadmap

| Module | Title | Status | Description |
| :--- | :--- | :---: | :--- |
| **Module 0** | **Project Setup & Scaffolding** | ✅ Done | Directory tree, dependencies, master config, and documentation. |
| **Module 1** | **Data Loading & Exploratory Analysis** | ✅ Done | Semicolon CSV ingestion, distribution plots, data quality audits. |
| **Module 2** | **Data Leakage Analysis** | ✅ Done | Formal classification: Safe, Potentially Unsafe, Excluded (post-ED). |
| **Module 3** | **Data Preprocessing & Splitting** | ✅ Done | Physiological bounds, imputation, scalers, stratified splits. |
| **Module 4** | **Baseline Models** | ✅ Done | Logistic Regression, Random Forest, HistGradientBoosting baselines. |
| **Module 5** | **Text Modality** | ✅ Done | MiniLM sentence embeddings vs TF-IDF on short chief complaints. |
| **Module 6** | **Vitals Encoder** | ✅ Done | Regularized MLP with BatchNorm, GELU, and Dropout. |
| **Module 7** | **Context Encoder** | ✅ Done | Entity embeddings for categorical variables + continuous MLP. |
| **Module 8** | **Multimodal Baseline Fusion** | ✅ Done | Simple feature concatenation baseline with MLP classification head. |
| **Module 9** | **Reliability-Aware Adaptive Fusion**| ✅ Done | Learned reliability scores, missingness weighting, cross-modal gating. |
| **Module 10** | **Ablation Study** | ✅ Done | Systematic ablation across unimodal, bimodal, and trimodal inputs. |
| **Module 11** | **Prediction Uncertainty** | ✅ Done | Monte Carlo Dropout, predictive entropy, human review thresholds. |
| **Module 12** | **Explainable AI** | ✅ Done | SHAP tabular importance, textual contribution, modality weights. |
| **Module 13** | **Voice Input Module** | ✅ Done | Whisper speech-to-text integration feeding chief complaint text. |
| **Module 14** | **Chest X-ray Analysis Module** | ✅ Done | Decoupled vision model (DenseNet/EfficientNet) for image findings. |
| **Module 15** | **Grad-CAM for X-ray Model** | ✅ Done | Visual model attention overlays for chest radiographs. |
| **Module 16** | **System Integration Layer** | ✅ Done | Multi-pipeline orchestrator uniting triage, speech, and vision. |
| **Module 17** | **Streamlit Application** | ✅ Done | Clinical research interface with tabs, cards, and safety disclaimers. |
| **Module 18** | **Final System Testing** | ✅ Done | End-to-end edge case testing, invalid inputs, and validation suite. |
| **Pillar A** | **Advanced Clinical NLP** | ✅ Done | NegEx syntactic negation scoping, 12-organ taxonomy, temporal duration & severity parsing. |
| **Pillar B** | **Advanced Computer Vision** | ✅ Done | Radiographic IQA (blur, contrast, clipping, photo rejection), Grad-CAM++, MCDO variance, OOD scoring. |
| **Pillar C** | **Cross-Modal Evidence & Conflict Engine (CECE)** | ✅ Done | Hemodynamic & metabolic validation, semantic contradictions (silent hypoxia, occult shock, pyrexia denial). |
| **Pillar D** | **RCCF-Net v2 Architecture** | ✅ Done | Residual Cross-Attention Clinical Fusion Network with missing-modality tokens & modality dropout. |
| **Pillar E** | **Advanced Uncertainty & Safety** | ✅ Done | Temperature scaling calibration ($T^*=1.05$), ECE, Brier score, selective prediction risk-coverage frontier. |
| **API** | **Production FastAPI Backend** | ✅ Done | Pydantic v2 schemas, API Key/Bearer auth, magic-bytes MIME security, lifespan model service, OpenAPI docs. |

---

### 3. Launching the Interactive Streamlit Web Interface

To launch the multi-tab Clinical Decision Support web application:
```bash
streamlit run app/streamlit_app.py
```
Key features:
- **Tab 1: Live Patient Triage**: Real-time multimodal input intake (Text, Voice, Vitals, Demographics, Decoupled X-ray), calibrated Monte Carlo uncertainty breakdown, and Grad-CAM saliency maps.
- **Tab 2: Case Studies & Benchmarks**: Pre-configured emergency presentation cohort (Hemodynamic Shock, Dispatch Audio, Sensor Drop, Ambiguous Malaise, Pneumothorax Escalation).
- **Tab 3: Architecture & Transparency**: Full architectural disclosure, KTAS level definitions, and zero-leakage scientific governance documentation.

---

### 4. Launching the Production FastAPI Backend

To start the enterprise REST API with OpenAPI documentation:
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Interactive OpenAPI / Swagger Documentation**: `http://localhost:8000/docs`
- **ReDoc Technical Interface**: `http://localhost:8000/redoc`
- **System Health & Checkpoint Status**: `GET http://localhost:8000/health`
- **Triage Acuity Inference**: `POST http://localhost:8000/api/v1/triage/predict`
- **CECE Cross-Modal Evidence Audit**: `POST http://localhost:8000/api/v1/cece/audit`
- **Thoracic Radiography & Grad-CAM++**: `POST http://localhost:8000/api/v1/vision/analyze`
- **Whisper Speech-to-Text**: `POST http://localhost:8000/api/v1/voice/transcribe`

Authentication is configured via `X-API-Key: triage-secret-key-2026` or Bearer Token.

---

### 5. Research & Comparative Experiments ($E_0 \to E_3$)

Evaluated on the held-out patient-level test split ($N=191$):

| Benchmark | Architecture & Configuration | Accuracy | Macro F1 | ECE | Brier Score | KTAS 1 Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **E0** | Baseline Adaptive Reliability Fusion | 60.21% | 0.5387 | 0.0757 | 0.5308 | 33.3% |
| **E1** | Baseline + CECE Audit | 60.21% | 0.5285 | 0.0757 | 0.5308 | **66.7%** (+33.4%) |
| **E2** | RCCF-Net v2 (Cross-Attention + Residual) | 59.16% | 0.4870 | 0.1952 | 0.5951 | 33.3% |
| **E3** | RCCF-Net v2 + CECE + Calibrated ($T=1.05$) | **60.21%** | 0.4974 | **0.0673** | 0.5800 | 33.3% |

To re-run the comparative experiment and generate metric artifacts:
```bash
python src/evaluation/run_comparative_experiments.py
```

---

### 6. Running the Complete Test & Validation Suite

To execute the complete 131-test verification suite across foundation modules, advanced pillars, and backend APIs:
```bash
pytest -v
```

To run individual pillar test suites:
```bash
pytest tests/test_clinical_nlp.py -v         # Pillar A: Clinical NLP (7 tests)
pytest tests/test_advanced_vision.py -v      # Pillar B: Advanced Vision & IQA (6 tests)
pytest tests/test_cece.py -v                 # Pillar C: CECE Engine (9 tests)
pytest tests/test_rccf_net_v2.py -v          # Pillar D: RCCF-Net v2 (4 tests)
pytest tests/test_advanced_uncertainty.py -v # Pillar E: Uncertainty & Calibration (6 tests)
pytest tests/test_api.py -v                  # Production FastAPI Endpoints (7 tests)
pytest tests/test_system_e2e.py -v           # Full E2E System Clinical Scenarios (9 tests)
```

For detailed mathematical formulations, architectural diagrams, and defense Q&A, refer to [`docs/MASTER_TECHNICAL_REPORT.md`](docs/MASTER_TECHNICAL_REPORT.md).

---

## 🔒 Scientific Integrity & Design Principles

1. **Strict Post-Admissions Leakage Prevention**:
   Variables generated during or after emergency care (such as *Diagnosis in ED*, *Disposition*, *Length of stay_min*, *KTAS duration_min*, *KTAS_expert*, *mistriage*) are strictly isolated from triage-intake prediction.
2. **Decoupled Medical Imaging**:
   Chest X-rays are analyzed as an independent vision diagnostic finding tool, avoiding synthetic or fictitious patient-record linkages between distinct cohorts.
3. **Calibrated Confidence over Softmax Overconfidence**:
   Softmax scores are complemented by Monte Carlo Dropout predictive variance and entropy to reliably flag ambiguous cases for human clinical intervention.
