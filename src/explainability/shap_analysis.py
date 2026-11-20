"""
Multimodal Intelligent Triage Assistant - SHAP Feature Attribution
===================================================================
Module 12 (Part 1): Tabular SHAP (SHapley Additive exPlanations) analysis
for physiological vital signs and patient demographic context.

Features:
  1. Multimodal neural wrapper for tabular perturbation (Vitals + Context).
  2. SHAP KernelExplainer and TreeExplainer integration.
  3. High-acuity (KTAS 1-2) vs Low-acuity (KTAS 4-5) feature attribution rankings.
  4. Publication-ready global beeswarm/bar plots and patient-level waterfall charts.

Research & Educational Prototype Only.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
import yaml

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure utf-8 encoding for standard output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

from src.models.context_encoder import ContextEncoder
from src.models.fusion import AdaptiveReliabilityFusion
from src.models.vitals_encoder import VitalsEncoder

# Tabular feature names in exact column order as produced by preprocessing
VITALS_FEATURE_NAMES = [
    "Saturation_is_missing",
    "SBP_scaled",
    "DBP_scaled",
    "HR_scaled",
    "RR_scaled",
    "BT_scaled",
    "Saturation_scaled",
    "NRS_pain_scaled",
    "Mental_raw",
    "Pain_flag",
]

CONTEXT_FEATURE_NAMES = [
    "Age_scaled",
    "Patients_per_hour_scaled",
    "Sex_cat",
    "Arrival_mode_cat",
    "Injury_cat",
]

TABULAR_FEATURE_NAMES = VITALS_FEATURE_NAMES + CONTEXT_FEATURE_NAMES

FEATURE_DISPLAY_NAMES = {
    "Saturation_is_missing": "SpO2 Unmeasured",
    "SBP_scaled": "Systolic BP",
    "DBP_scaled": "Diastolic BP",
    "HR_scaled": "Heart Rate",
    "RR_scaled": "Respiratory Rate",
    "BT_scaled": "Body Temperature",
    "Saturation_scaled": "Oxygen Saturation (SpO2)",
    "NRS_pain_scaled": "Pain Score (NRS)",
    "Mental_raw": "Mental Status (AVPU)",
    "Pain_flag": "Pain Flag",
    "Age_scaled": "Patient Age",
    "Patients_per_hour_scaled": "ED Patient Volume",
    "Sex_cat": "Patient Sex",
    "Arrival_mode_cat": "Arrival Mode (Ambulance)",
    "Injury_cat": "Trauma / Injury",
}


# ==============================================================================
# 1. TABULAR PREDICTION WRAPPER FOR MULTIMODAL MODEL
# ==============================================================================
class TabularMultimodalPredictor:
    """
    Wrapper mapping tabular features (15-d) through VitalsEncoder and ContextEncoder,
    then combining with a reference text embedding into AdaptiveReliabilityFusion.
    Outputs softmax class probabilities (N, 5) for SHAP KernelExplainer.
    """

    def __init__(
        self,
        adaptive_model: AdaptiveReliabilityFusion,
        vitals_encoder: VitalsEncoder,
        context_encoder: ContextEncoder,
        reference_text_embedding: torch.Tensor,
        device: str = "cpu",
    ):
        self.adaptive_model = adaptive_model.to(device).eval()
        self.vitals_encoder = vitals_encoder.to(device).eval()
        self.context_encoder = context_encoder.to(device).eval()
        self.reference_text = reference_text_embedding.to(device)
        self.device = device

    def predict_proba(self, X_tabular: np.ndarray) -> np.ndarray:
        """
        Takes (N, 15) numpy array of tabular features and returns (N, 5) probabilities.
        """
        n_samples = len(X_tabular)
        if n_samples == 0:
            return np.zeros((0, 5))

        # Split into vitals (10) and context (5)
        # Slices: vitals = cols 0..9, context_num = cols 10..11, context_cat = cols 12..14
        x_vit = torch.tensor(X_tabular[:, :10], dtype=torch.float32, device=self.device)
        x_num = torch.tensor(X_tabular[:, 10:12], dtype=torch.float32, device=self.device)
        x_cat = torch.tensor(X_tabular[:, 12:15], dtype=torch.long, device=self.device)

        # Broadcast text embedding if single reference vector
        if self.reference_text.dim() == 1:
            x_text = self.reference_text.unsqueeze(0).expand(n_samples, -1)
        elif len(self.reference_text) == n_samples:
            x_text = self.reference_text.to(self.device)
        else:
            x_text = self.reference_text[0:1].expand(n_samples, -1)

        with torch.no_grad():
            z_vitals = self.vitals_encoder(x_vit)
            z_context = self.context_encoder(x_cat, x_num)
            logits = self.adaptive_model(text=x_text, vitals=z_vitals, context=z_context)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return probs


# ==============================================================================
# 2. SHAP EXPLAINER COMPUTATION
# ==============================================================================
def compute_tabular_shap_explanations(
    predictor: TabularMultimodalPredictor,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Optional[List[str]] = None,
    num_samples_nsamples: int = 100,
) -> Dict[str, Any]:
    """
    Compute SHAP values using KernelExplainer across KTAS triage acuity classes.
    """
    if feature_names is None:
        feature_names = TABULAR_FEATURE_NAMES

    # Summarize background with k-means or median if large
    if len(X_background) > 50:
        bg_summary = shap.sample(X_background, 30, random_state=42)
    else:
        bg_summary = X_background

    print(f"Initializing SHAP KernelExplainer (background={len(bg_summary)}, explain={len(X_explain)})...")
    explainer = shap.KernelExplainer(predictor.predict_proba, bg_summary)

    # Compute Shapley values
    shap_values = explainer.shap_values(X_explain, nsamples=num_samples_nsamples)

    # Convert shap_values list of arrays [class 0..4] to (N, num_features, num_classes)
    # SHAP returns a list of length 5 (one per class), each (N, num_features)
    # Or in newer shap versions: (N, num_features, num_classes)
    if isinstance(shap_values, list):
        shap_array = np.stack(shap_values, axis=-1)  # (N, features, classes)
    else:
        shap_array = np.array(shap_values)

    expected_value = explainer.expected_value

    # Compute mean absolute SHAP per feature across all classes
    mean_abs_shap = np.mean(np.abs(shap_array), axis=0)  # (features, classes)
    global_importance = np.mean(mean_abs_shap, axis=-1)   # (features,)

    # High-Acuity specific importance: KTAS 1 & 2
    high_acuity_importance = (mean_abs_shap[:, 0] + mean_abs_shap[:, 1]) / 2.0

    importance_df = pd.DataFrame({
        "Feature_Key": feature_names,
        "Feature_Name": [FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names],
        "Global_Importance": global_importance,
        "High_Acuity_Importance_KTAS1_2": high_acuity_importance,
        "KTAS_1_Resuscitation_Impact": mean_abs_shap[:, 0],
        "KTAS_2_Emergent_Impact": mean_abs_shap[:, 1],
        "KTAS_3_Urgent_Impact": mean_abs_shap[:, 2],
        "KTAS_4_Less_Urgent_Impact": mean_abs_shap[:, 3],
        "KTAS_5_Non_Urgent_Impact": mean_abs_shap[:, 4],
    }).sort_values(by="Global_Importance", ascending=False).reset_index(drop=True)

    return {
        "shap_array": shap_array,
        "expected_value": expected_value,
        "importance_df": importance_df,
        "feature_names": feature_names,
    }


# ==============================================================================
# 3. HIGH-RESOLUTION SHAP VISUALIZATIONS
# ==============================================================================
def plot_tabular_shap_global_summary(
    importance_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Generate horizontal bar plot of global physiological and context feature importances.
    """
    plt.figure(figsize=(10, 6.5))
    sns.set_theme(style="whitegrid")

    df_sorted = importance_df.sort_values(by="Global_Importance", ascending=True)

    # Color code by domain (Vitals in Teal/Blue, Context in Purple/Orange)
    colors = [
        "#1f77b4" if f in VITALS_FEATURE_NAMES else "#ff7f0e"
        for f in df_sorted["Feature_Key"]
    ]

    bars = plt.barh(df_sorted["Feature_Name"], df_sorted["Global_Importance"], color=colors, alpha=0.85, edgecolor="black", height=0.65)

    # Add data labels
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#333333",
            weight="bold",
        )

    plt.title("Multimodal Triage Model: Global Tabular Feature Attribution (SHAP)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Mean Absolute SHAP Value (Impact on KTAS Acuity Probabilities)", fontsize=11)
    plt.xlim(0, max(df_sorted["Global_Importance"]) * 1.18)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", edgecolor="black", label="Physiological Vital Signs (VitalsEncoder)"),
        Patch(facecolor="#ff7f0e", edgecolor="black", label="Demographic & Arrival Context (ContextEncoder)"),
    ]
    plt.legend(handles=legend_elements, loc="lower right", frameon=True, framealpha=0.95)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved global SHAP summary plot to: {save_path}")


def plot_high_vs_low_acuity_shap_comparison(
    importance_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Side-by-side comparative bar chart contrasting feature impacts on
    High-Acuity (KTAS 1-2) versus Low-Acuity (KTAS 4-5) emergency patients.
    """
    plt.figure(figsize=(11, 7))
    sns.set_theme(style="whitegrid")

    df_plot = importance_df.sort_values(by="High_Acuity_Importance_KTAS1_2", ascending=True)

    y = np.arange(len(df_plot))
    height = 0.38

    low_acuity = (df_plot["KTAS_4_Less_Urgent_Impact"] + df_plot["KTAS_5_Non_Urgent_Impact"]) / 2.0

    plt.barh(y + height / 2, df_plot["High_Acuity_Importance_KTAS1_2"], height=height, color="#d62728", alpha=0.85, label="High Acuity (KTAS 1 & 2: Resuscitation/Emergent)", edgecolor="black")
    plt.barh(y - height / 2, low_acuity, height=height, color="#2ca02c", alpha=0.85, label="Low Acuity (KTAS 4 & 5: Less/Non-Urgent)", edgecolor="black")

    plt.yticks(y, df_plot["Feature_Name"], fontsize=10.5)
    plt.xlabel("Mean Absolute SHAP Value (Class Probability Impact)", fontsize=11, weight="bold")
    plt.title("Feature Specificity: High-Acuity vs Low-Acuity Triage Drivers", fontsize=13, weight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved High vs Low acuity SHAP comparison plot to: {save_path}")


def plot_patient_shap_waterfall(
    patient_shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: List[str],
    base_value: float,
    pred_class_name: str,
    patient_id: int,
    save_path: Path,
    top_k: int = 8,
) -> None:
    """
    Generate waterfall attribution plot for a specific patient's prediction.
    """
    plt.figure(figsize=(9, 5.5))
    sns.set_theme(style="whitegrid")

    # Sort by absolute SHAP value
    indices = np.argsort(np.abs(patient_shap_values))[::-1][:top_k]
    sel_shap = patient_shap_values[indices]
    sel_names = [f"{FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i])} ({feature_values[i]:.2f})" for i in indices]

    # Reverse for plotting top features at the top
    sel_shap = sel_shap[::-1]
    sel_names = sel_names[::-1]

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in sel_shap]

    plt.barh(sel_names, sel_shap, color=colors, alpha=0.85, edgecolor="black", height=0.6)
    plt.axvline(x=0, color="black", linestyle="-", linewidth=0.8)

    plt.title(f"Patient #{patient_id} Decision Waterfall (Target: {pred_class_name})", fontsize=12, weight="bold", pad=12)
    plt.xlabel("SHAP Value (Contribution to Predicted Acuity Log-Odds)", fontsize=10.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", edgecolor="black", label="Increases Acuity Severity (+)"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Decreases Acuity Severity (-)"),
    ]
    plt.legend(handles=legend_elements, loc="lower right", frameon=True)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved patient #{patient_id} SHAP waterfall plot to: {save_path}")
