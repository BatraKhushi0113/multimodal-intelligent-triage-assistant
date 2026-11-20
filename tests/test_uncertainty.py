"""
Unit Tests for Module 11: Prediction Uncertainty Estimation & Clinical Safety Gating
=====================================================================================
Tests Monte Carlo Dropout control, predictive entropy bounds, epistemic/aleatoric
decomposition, calibration error, and triage safety alert rules.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.uncertainty import (
    MCDropoutPredictor,
    TriageSafetyAuditor,
    compute_expected_calibration_error,
    disable_mc_dropout,
    enable_mc_dropout,
    evaluate_misclassification_detection,
    evaluate_selective_prediction,
)


class DummyModelWithDropout(nn.Module):
    """Simple toy neural net with LayerNorm and Dropout for testing."""

    def __init__(self, in_features=16, num_classes=5, p=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, 32)
        self.drop = nn.Dropout(p)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, text, vitals, context, mask_modalities=None):
        # Concatenate dummy inputs
        x = torch.cat([text, vitals, context], dim=-1)
        x = self.norm(x)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        logits = self.fc2(x)
        return logits


def test_enable_disable_mc_dropout():
    """Verify that enable_mc_dropout sets dropout to train mode and norm to eval mode."""
    model = DummyModelWithDropout()
    model.train()  # start in train mode

    enable_mc_dropout(model)

    assert model.drop.training is True, "Dropout should be active in MCDO mode."
    assert model.norm.training is False, "LayerNorm must remain in eval mode during MCDO."

    disable_mc_dropout(model)
    assert model.drop.training is False, "Dropout should be disabled in deterministic eval mode."
    assert model.norm.training is False, "LayerNorm should be in eval mode."


def test_mc_dropout_stochastic_variance():
    """Verify that MCDropoutPredictor produces non-zero probability variance across passes."""
    model = DummyModelWithDropout(in_features=16, num_classes=5, p=0.5)
    predictor = MCDropoutPredictor(model=model, num_passes=15, num_classes=5)

    # 4 dummy samples
    t = torch.randn(4, 6)
    v = torch.randn(4, 6)
    c = torch.randn(4, 4)

    res = predictor.predict_stochastic_batch(t, v, c)

    p_samples = res["p_samples"]  # (15, 4, 5)
    mean_probs = res["mean_probs"]  # (4, 5)
    var_probs = res["prob_variance"]  # (4, 5)

    assert p_samples.shape == (15, 4, 5)
    assert mean_probs.shape == (4, 5)
    assert var_probs.shape == (4, 5)

    # Probabilities must sum to 1.0 for each sample and pass
    sample_sums = p_samples.sum(dim=-1)
    assert torch.allclose(sample_sums, torch.ones_like(sample_sums), atol=1e-5)

    # Variance must be strictly positive with p=0.5 dropout
    assert (var_probs > 0).any(), "MCDO should produce stochastic probability variations."


def test_predictive_entropy_bounds():
    """Verify mathematical bounds of normalized predictive entropy H in [0, 1]."""
    num_classes = 5
    log2_c = np.log2(num_classes)

    # 1. Perfectly confident distribution (one-hot)
    one_hot = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    ent_one_hot = -np.sum(one_hot * np.log2(np.clip(one_hot, 1e-12, 1.0)), axis=-1) / log2_c
    assert np.isclose(ent_one_hot[0], 0.0, atol=1e-5)

    # 2. Maximum uncertainty distribution (uniform)
    uniform = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    ent_uniform = -np.sum(uniform * np.log2(np.clip(uniform, 1e-12, 1.0)), axis=-1) / log2_c
    assert np.isclose(ent_uniform[0], 1.0, atol=1e-5)

    # 3. Intermediate distribution
    intermediate = np.array([[0.5, 0.3, 0.1, 0.05, 0.05]])
    ent_inter = -np.sum(intermediate * np.log2(np.clip(intermediate, 1e-12, 1.0)), axis=-1) / log2_c
    assert 0.0 < ent_inter[0] < 1.0


def test_epistemic_aleatoric_decomposition():
    """Verify mutual information is non-negative and sums logically."""
    # Simulated N=10 samples with T=20 passes
    rng = np.random.RandomState(42)
    raw_logits = rng.randn(10, 20, 5)
    # Softmax over classes
    exp_l = np.exp(raw_logits - np.max(raw_logits, axis=-1, keepdims=True))
    p_samples = exp_l / np.sum(exp_l, axis=-1, keepdims=True)  # (10, 20, 5)

    mean_p = np.mean(p_samples, axis=1)  # (10, 5)
    log2_c = np.log2(5)

    # Total entropy
    h_total = -np.sum(mean_p * np.log2(np.clip(mean_p, 1e-12, 1.0)), axis=-1) / log2_c
    # Aleatoric
    h_sample = -np.sum(p_samples * np.log2(np.clip(p_samples, 1e-12, 1.0)), axis=-1) / log2_c
    h_aleatoric = np.mean(h_sample, axis=-1)
    # Epistemic
    h_epistemic = np.maximum(h_total - h_aleatoric, 0.0)

    # Mathematical property: H(E[p]) >= E[H(p)] (Jensen's inequality)
    assert np.all(h_total >= h_aleatoric - 1e-6), "Predictive entropy must be >= expected entropy."
    assert np.all(h_epistemic >= 0.0), "Epistemic uncertainty (Mutual Information) must be non-negative."


def test_safety_auditor_flags():
    """Verify clinical decision support safety auditor rules."""
    auditor = TriageSafetyAuditor(
        entropy_threshold=0.85,
        min_confidence_threshold=0.50,
        ktas1_under_triage_hazard_threshold=0.12,
        ktas2_under_triage_hazard_threshold=0.25,
    )

    # Case A: Low risk, high confidence patient
    res_a = auditor.audit_patient(
        pred_class=2,  # KTAS 3
        probabilities=np.array([0.01, 0.04, 0.90, 0.03, 0.02]),
        normalized_entropy=0.25,
        epistemic_uncertainty=0.03,
    )
    assert res_a["is_flagged"] is False
    assert res_a["risk_level"] == "LOW_RISK"
    assert res_a["clinical_action"] == "ROUTINE_TRIAGE_WORKFLOW"

    # Case B: Critical Under-Triage Hazard (Predicted KTAS 3, but KTAS 1 prob = 15%)
    res_b = auditor.audit_patient(
        pred_class=2,  # KTAS 3
        probabilities=np.array([0.15, 0.10, 0.60, 0.10, 0.05]),
        normalized_entropy=0.70,
        epistemic_uncertainty=0.08,
    )
    assert res_b["is_flagged"] is True
    assert res_b["risk_level"] == "CRITICAL_UNDER_TRIAGE_HAZARD"
    assert res_b["clinical_action"] == "IMMEDIATE_PHYSICIAN_OVERRIDE"

    # Case C: High Ambiguity (Entropy = 0.92 > 0.85)
    res_c = auditor.audit_patient(
        pred_class=3,  # KTAS 4
        probabilities=np.array([0.02, 0.15, 0.35, 0.40, 0.08]),
        normalized_entropy=0.92,
        epistemic_uncertainty=0.18,
    )
    assert res_c["is_flagged"] is True
    assert res_c["risk_level"] == "HIGH_UNCERTAINTY"
    assert res_c["clinical_action"] == "MANDATORY_NURSE_REASSESSMENT"


def test_expected_calibration_error():
    """Verify ECE calculation on synthetic perfectly calibrated predictions."""
    y_true = np.array([0, 1, 2, 3, 4] * 20)
    # Perfectly confident correct predictions
    y_probs_perfect = np.zeros((100, 5))
    for i, t in enumerate(y_true):
        y_probs_perfect[i, t] = 1.0

    res = compute_expected_calibration_error(y_true, y_probs_perfect, num_bins=10)
    assert np.isclose(res["ece"], 0.0, atol=1e-5), "Perfect predictions should have 0 ECE."

    # Miscalibrated: 100% confidence on all incorrect predictions
    y_probs_wrong = np.zeros((100, 5))
    for i, t in enumerate(y_true):
        wrong_cls = (t + 1) % 5
        y_probs_wrong[i, wrong_cls] = 1.0

    res_wrong = compute_expected_calibration_error(y_true, y_probs_wrong, num_bins=10)
    assert np.isclose(res_wrong["ece"], 1.0, atol=1e-5), "100% confident errors should have ECE = 1.0."


def test_selective_prediction_monotonicity():
    """Verify that filtering out high uncertainty cases generates a valid monotonic dataframe."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 10)
    # Correct on first 80, wrong on last 20
    y_pred = y_true.copy()
    y_pred[80:] = (y_true[80:] + 1) % 5

    # Errors have high uncertainty
    uncertainty = np.zeros(100)
    uncertainty[:80] = np.random.uniform(0.1, 0.4, size=80)
    uncertainty[80:] = np.random.uniform(0.7, 0.95, size=20)

    df_rej = evaluate_selective_prediction(
        y_true=y_true,
        y_pred=y_pred,
        uncertainty_scores=uncertainty,
        rejection_fractions=[0.0, 0.10, 0.20, 0.30],
    )

    assert len(df_rej) == 4
    assert "Accuracy" in df_rej.columns
    assert "Macro_F1" in df_rej.columns

    # As we reject top 20% most uncertain samples, accuracy should increase from 80% to 100%
    assert df_rej.iloc[-1]["Accuracy"] >= df_rej.iloc[0]["Accuracy"]
