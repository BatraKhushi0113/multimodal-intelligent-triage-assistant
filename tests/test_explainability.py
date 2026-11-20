"""
Unit Tests for Module 12: Explainable AI (XAI)
==============================================
Tests tabular predictor wrapper, SHAP attribution dimensions, Captum Integrated
Gradients, text keyword attribution, and unified clinical triage narratives.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import (
    MultimodalIntegratedGradientsAttributor,
    TABULAR_FEATURE_NAMES,
    TabularMultimodalPredictor,
    TextKeywordAttributor,
    TriageExplainer,
)
from src.models.context_encoder import ContextEncoder
from src.models.fusion import AdaptiveReliabilityFusion
from src.models.vitals_encoder import VitalsEncoder


@pytest.fixture
def dummy_multimodal_setup():
    """Create lightweight test models and embeddings."""
    vitals_enc = VitalsEncoder(input_dim=10, hidden_dim=32, embedding_dim=64, dropout_prob=0.1)
    context_enc = ContextEncoder(hidden_dim=32, embedding_dim=32, dropout_prob=0.1)
    adaptive_model = AdaptiveReliabilityFusion(
        text_dim=64,
        vitals_dim=64,
        context_dim=32,
        shared_dim=64,
        num_classes=5,
    )
    ref_text = torch.randn(64)
    return {
        "vitals_enc": vitals_enc,
        "context_enc": context_enc,
        "adaptive_model": adaptive_model,
        "ref_text": ref_text,
    }


def test_tabular_multimodal_predictor_shapes(dummy_multimodal_setup):
    """Verify TabularMultimodalPredictor accepts (N, 15) and returns (N, 5) valid probabilities."""
    setup = dummy_multimodal_setup
    predictor = TabularMultimodalPredictor(
        adaptive_model=setup["adaptive_model"],
        vitals_encoder=setup["vitals_enc"],
        context_encoder=setup["context_enc"],
        reference_text_embedding=setup["ref_text"],
        device="cpu",
    )

    X_dummy = np.random.randn(8, 15)
    # Categorical features in cols 12, 13, 14 should be valid non-negative ints
    X_dummy[:, 12] = np.random.randint(0, 2, size=8)
    X_dummy[:, 13] = np.random.randint(0, 5, size=8)
    X_dummy[:, 14] = np.random.randint(0, 2, size=8)

    probs = predictor.predict_proba(X_dummy)

    assert probs.shape == (8, 5)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    # Check probabilities sum to 1.0
    row_sums = probs.sum(axis=-1)
    assert np.allclose(row_sums, np.ones(8), atol=1e-5)


def test_integrated_gradients_attributions(dummy_multimodal_setup):
    """Verify Integrated Gradients computes valid norms and relative attributions summing to 1.0."""
    setup = dummy_multimodal_setup
    ig_attributor = MultimodalIntegratedGradientsAttributor(setup["adaptive_model"], device="cpu")

    t = torch.randn(64)
    v = torch.randn(64)
    c = torch.randn(32)

    res = ig_attributor.attribute(t, v, c, target_class=1, n_steps=10)

    assert "text_attribution_vector" in res
    assert "vitals_attribution_vector" in res
    assert "context_attribution_vector" in res
    assert "relative_attributions" in res

    rel = res["relative_attributions"]
    assert "Text" in rel and "Vitals" in rel and "Context" in rel
    total_rel = rel["Text"] + rel["Vitals"] + rel["Context"]
    assert np.isclose(total_rel, 1.0, atol=1e-5)


def test_text_keyword_attributor(dummy_multimodal_setup):
    """Verify keyword attributor extracts clinical trigger tokens."""
    setup = dummy_multimodal_setup
    attributor = TextKeywordAttributor(setup["adaptive_model"], device="cpu")

    t = torch.randn(64)
    v = torch.randn(64)
    c = torch.randn(32)

    text_complaint = "severe chest pain radiating to left shoulder and dyspnea"
    kw_attrs = attributor.attribute_text_keywords(
        text_string=text_complaint,
        base_text_embedding=t,
        vitals_embedding=v,
        context_embedding=c,
        target_class=1,  # Emergent
    )

    assert len(kw_attrs) > 0
    words = [k["clean_token"] for k in kw_attrs]
    assert "chest" in words or "pain" in words or "dyspnea" in words

    for kw in kw_attrs:
        assert "word" in kw
        assert "attribution" in kw
        assert "direction" in kw


def test_triage_explainer_schema(dummy_multimodal_setup):
    """Verify TriageExplainer produces valid clinical justification narratives and complete schema."""
    setup = dummy_multimodal_setup

    dummy_shap_data = {
        "shap_array": np.random.randn(5, 15, 5),
    }

    explainer = TriageExplainer(
        adaptive_model=setup["adaptive_model"],
        vitals_encoder=setup["vitals_enc"],
        context_encoder=setup["context_enc"],
        tabular_shap_data=dummy_shap_data,
        device="cpu",
    )

    t = torch.randn(64)
    v = torch.randn(64)
    c = torch.randn(32)
    tab_feat = np.random.randn(15)

    exp = explainer.explain_patient(
        patient_idx=0,
        raw_text="cardiac arrest and unconscious",
        text_embedding=t,
        vitals_embedding=v,
        context_embedding=c,
        tabular_features=tab_feat,
        true_ktas=0,
    )

    assert exp["patient_id"] == 0
    assert 0 <= exp["predicted_ktas"] < 5
    assert "KTAS" in exp["predicted_ktas_name"]
    assert 0.0 <= exp["confidence"] <= 1.0
    assert "modality_attention_weights" in exp
    assert "clinical_narrative" in exp
    assert len(exp["clinical_narrative"]) > 20
    assert "triaged as" in exp["clinical_narrative"]
