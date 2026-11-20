"""
Unit Tests for Advanced Uncertainty, Calibration & Safety Module.
================================================================
Tests Temperature Scaling, multi-class Brier score, ECE computation,
selective prediction abstention curves, and multi-tier clinical safety audits.
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

from src.models.advanced_uncertainty import (
    ClinicalSafetyAuditor,
    ClinicalSafetyVerdict,
    SelectivePredictionEngine,
    TemperatureCalibrator,
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    compute_predictive_entropy,
)


def test_temperature_scaling_optimization():
    """Verify TemperatureCalibrator fits temperature parameter on validation logits."""
    torch.manual_seed(42)
    calibrator = TemperatureCalibrator(init_temperature=1.0)

    # Overconfident synthetic logits
    logits = torch.randn(100, 5) * 4.0
    targets = torch.randint(0, 5, (100,))

    res = calibrator.fit(logits, targets, lr=0.05, max_iters=50)
    T = res["fitted_temperature"]

    assert T > 0.05
    scaled = calibrator(logits)
    assert scaled.shape == (100, 5)
    # Scaled logits should have lower variance than overconfident initial logits
    assert torch.std(scaled) < torch.std(logits)


def test_multiclass_brier_score():
    """Verify proper scoring behavior of Brier score."""
    targets = np.array([0, 1, 2])
    # Perfect predictions
    perfect_probs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    bs_perfect = compute_multiclass_brier_score(perfect_probs, targets, num_classes=3)
    assert bs_perfect == 0.0

    # Completely wrong predictions
    worst_probs = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    bs_worst = compute_multiclass_brier_score(worst_probs, targets, num_classes=3)
    assert bs_worst == 2.0


def test_expected_calibration_error():
    """Verify ECE and MCE computation across probability bins."""
    np.random.seed(42)
    probs = np.array([
        [0.90, 0.05, 0.05],
        [0.85, 0.10, 0.05],
        [0.40, 0.40, 0.20],
        [0.10, 0.80, 0.10],
    ])
    targets = np.array([0, 0, 1, 1])

    ece, mce, bins = compute_expected_calibration_error(probs, targets, num_bins=5)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert len(bins) > 0


def test_predictive_entropy_bounds():
    """Verify normalized entropy bounds: 1.0 for uniform, 0.0 for one-hot."""
    uniform = np.full((1, 5), 0.20)
    ent_uniform = compute_predictive_entropy(uniform, num_classes=5)
    assert np.isclose(ent_uniform[0], 1.0, atol=1e-4)

    one_hot = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])
    ent_one_hot = compute_predictive_entropy(one_hot, num_classes=5)
    assert np.isclose(ent_one_hot[0], 0.0, atol=1e-4)


def test_selective_prediction_curve():
    """Verify abstention reduces error on retained samples."""
    np.random.seed(42)
    N = 100
    # Create mix of high confidence correct and low confidence wrong
    probs = np.zeros((N, 5))
    targets = np.random.randint(0, 5, size=N)

    for i in range(N):
        if i < 70:  # Confident correct
            probs[i, targets[i]] = 0.90
            probs[i] += 0.02
            probs[i] /= np.sum(probs[i])
        else:  # Ambiguous incorrect
            wrong = (targets[i] + 1) % 5
            probs[i, wrong] = 0.35
            probs[i, targets[i]] = 0.30
            probs[i] += 0.07
            probs[i] /= np.sum(probs[i])

    points, auroc = SelectivePredictionEngine.evaluate_abstention_curve(
        probs, targets, coverage_steps=[1.0, 0.85, 0.70]
    )
    assert len(points) == 3
    assert auroc > 0.70
    # As coverage decreases, accuracy on retained samples should increase
    assert points[-1].accuracy >= points[0].accuracy


def test_clinical_safety_auditor_rules():
    """Verify safety auditor triggers correct clinical governance verdicts."""
    auditor = ClinicalSafetyAuditor(entropy_threshold=0.80)

    # 1. Safe clear case
    p_safe = np.array([0.05, 0.05, 0.85, 0.03, 0.02])
    v_safe = auditor.audit_prediction(p_safe, cece_conflict_severity="NONE")
    assert v_safe.verdict == "AUTOMATED_DECISION_CLEARED"
    assert v_safe.is_safe_to_automate is True

    # 2. High entropy ambiguity case (subacute KTAS 3/4/5) -> Deferred to Nurse
    p_ambig = np.array([0.02, 0.05, 0.35, 0.33, 0.25])
    v_ambig = auditor.audit_prediction(p_ambig, cece_conflict_severity="NONE")
    assert v_ambig.verdict == "DEFERRED_FOR_NURSE_TRIAGE"
    assert v_ambig.requires_human_review is True

    # 3. Occult KTAS 1 risk (predicted KTAS 3, but 18% tail for Resuscitation) -> Physician Override
    p_occult = np.array([0.18, 0.10, 0.60, 0.08, 0.04])
    v_occult = auditor.audit_prediction(p_occult, cece_conflict_severity="NONE")
    assert v_occult.verdict == "IMMEDIATE_PHYSICIAN_OVERRIDE"
    assert "OCCULT_KTAS_1" in str(v_occult.primary_hazard_flag)

    # 4. Critical CECE Conflict -> Physician Override
    v_conflict = auditor.audit_prediction(p_safe, cece_conflict_severity="CRITICAL")
    assert v_conflict.verdict == "IMMEDIATE_PHYSICIAN_OVERRIDE"
