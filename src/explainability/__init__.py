"""
Multimodal Intelligent Triage Assistant - Explainability (XAI)
=============================================================
Module 12: SHAP tabular attribution, Captum Integrated Gradients,
chief complaint token attribution, and unified clinical justifications.
"""

from src.explainability.explanations import (
    MultimodalIntegratedGradientsAttributor,
    TextKeywordAttributor,
    TriageExplainer,
    run_explainability_module,
)
from src.explainability.shap_analysis import (
    FEATURE_DISPLAY_NAMES,
    TABULAR_FEATURE_NAMES,
    VITALS_FEATURE_NAMES,
    CONTEXT_FEATURE_NAMES,
    TabularMultimodalPredictor,
    compute_tabular_shap_explanations,
    plot_high_vs_low_acuity_shap_comparison,
    plot_patient_shap_waterfall,
    plot_tabular_shap_global_summary,
)

__all__ = [
    "TABULAR_FEATURE_NAMES",
    "VITALS_FEATURE_NAMES",
    "CONTEXT_FEATURE_NAMES",
    "FEATURE_DISPLAY_NAMES",
    "TabularMultimodalPredictor",
    "compute_tabular_shap_explanations",
    "plot_tabular_shap_global_summary",
    "plot_high_vs_low_acuity_shap_comparison",
    "plot_patient_shap_waterfall",
    "MultimodalIntegratedGradientsAttributor",
    "TextKeywordAttributor",
    "TriageExplainer",
    "run_explainability_module",
]
