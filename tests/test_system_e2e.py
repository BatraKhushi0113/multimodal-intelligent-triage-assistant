"""
Unit & Integration Tests for Module 18: System End-to-End Stress Validation.
=============================================================================
Tests clinical edge cases, sensor dropouts, text absence, discordant presentations,
and decoupled radiographic escalation protocols across the unified orchestrator.
"""

from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure unpickling compatibility for preprocessor.joblib
import src.data.preprocessing as prep
for attr in ["ClinicalVitalsCleaner", "ClinicalContextCleaner", "PreprocessingPipeline"]:
    if hasattr(prep, attr):
        setattr(sys.modules.setdefault("__main__", sys.modules[__name__]), attr, getattr(prep, attr))

from src.evaluation.evaluate_system_e2e import get_e2e_stress_test_matrix
from src.integration.triage_orchestrator import (
    PatientIntakeRecord,
    TriageOrchestrator,
    TriageResult,
)
from src.vision.xray_model import generate_synthetic_chest_xray


@pytest.fixture(scope="module")
def e2e_orchestrator():
    """Module-level TriageOrchestrator fixture with lightweight loading."""
    orch = TriageOrchestrator(
        config_path="config.yaml",
        device="cpu",
        load_whisper=False,  # Lazy load on demand
        load_vision=True,
    )
    return orch


def test_e2e_stress_matrix_completeness():
    """Verify that the 15-scenario stress matrix is properly defined and complete."""
    matrix = get_e2e_stress_test_matrix()
    assert len(matrix) == 15, f"Expected 15 stress test scenarios, found {len(matrix)}"

    required_keys = {"id", "name", "category", "expected_ktas_range", "description", "record"}
    for item in matrix:
        assert required_keys.issubset(item.keys()), f"Missing keys in scenario {item.get('id')}"
        assert isinstance(item["record"], PatientIntakeRecord)
        assert len(item["expected_ktas_range"]) == 2
        assert 1 <= item["expected_ktas_range"][0] <= item["expected_ktas_range"][1] <= 5


def test_extreme_shock_patient_ktas1(e2e_orchestrator):
    """Verify that profound cardiogenic shock (E2E-01) is triaged as emergent/resuscitation (KTAS 1 or 2)."""
    matrix = get_e2e_stress_test_matrix()
    shock_case = next(c for c in matrix if c["id"] == "E2E-01")

    res: TriageResult = e2e_orchestrator.triage_patient(shock_case["record"], generate_plots=False)
    assert res.final_ktas_level in [1, 2], f"Shock patient should be KTAS 1 or 2, got {res.final_ktas_level}"
    assert res.modality_weights["Vitals"] > 0.30, "Vitals should receive high reliability weight in shock"
    assert not np.isnan(res.confidence)
    assert not np.isnan(res.predictive_entropy)


def test_empty_chief_complaint_handling(e2e_orchestrator):
    """Verify that empty / whitespace chief complaint (E2E-05) executes without error."""
    matrix = get_e2e_stress_test_matrix()
    empty_case = next(c for c in matrix if c["id"] == "E2E-05")

    res: TriageResult = e2e_orchestrator.triage_patient(empty_case["record"], generate_plots=False)
    assert 1 <= res.final_ktas_level <= 5
    assert len(res.intake_probabilities) == 5
    assert np.isclose(sum(res.intake_probabilities), 1.0, atol=1e-3)
    assert not any(np.isnan(p) for p in res.intake_probabilities)


def test_missing_spo2_sensor_dropout(e2e_orchestrator):
    """Verify that unmeasured SpO2 (E2E-06) is handled via adaptive missingness without NaNs."""
    matrix = get_e2e_stress_test_matrix()
    sensor_case = next(c for c in matrix if c["id"] == "E2E-06")

    res: TriageResult = e2e_orchestrator.triage_patient(sensor_case["record"], generate_plots=False)
    assert 1 <= res.final_ktas_level <= 5
    assert not any(np.isnan(p) for p in res.intake_probabilities)
    assert res.modality_weights["Vitals"] <= 0.60, "Vitals weight should be penalized when sensor missing"


def test_contradictory_pain_reporting(e2e_orchestrator):
    """Verify that contradictory pain flags (E2E-08, pain=0 but NRS=10) execute gracefully."""
    matrix = get_e2e_stress_test_matrix()
    pain_case = next(c for c in matrix if c["id"] == "E2E-08")

    res: TriageResult = e2e_orchestrator.triage_patient(pain_case["record"], generate_plots=False)
    assert 1 <= res.final_ktas_level <= 5
    assert not np.isnan(res.confidence)


def test_vitals_override_benign_text_discordance(e2e_orchestrator):
    """Verify that severe vitals discordance (E2E-10) allocates dominant weight to vitals and flags red flags."""
    matrix = get_e2e_stress_test_matrix()
    disc_case = next(c for c in matrix if c["id"] == "E2E-10")

    res: TriageResult = e2e_orchestrator.triage_patient(disc_case["record"], generate_plots=False)
    # Adaptive fusion must allocate higher reliability weight to vitals than to benign text
    assert res.modality_weights["Vitals"] > res.modality_weights["Text"], "Vitals must receive higher weight than text"
    assert any("Hypoxia" in d["factor"] or "Respiratory" in d["factor"] for d in res.top_clinical_drivers), "Must identify hypoxia or tachypnea driver"


def test_decoupled_radiographic_pneumothorax_escalation(e2e_orchestrator):
    """Verify that tension pneumothorax on radiograph (E2E-12) triggers critical alert or escalation."""
    matrix = get_e2e_stress_test_matrix()
    xray_case = next(c for c in matrix if c["id"] == "E2E-12")

    res: TriageResult = e2e_orchestrator.triage_patient(xray_case["record"], generate_plots=False)
    assert res.xray_evaluated, "Chest X-ray should be evaluated"
    assert res.primary_xray_finding == "Pneumothorax"
    assert res.final_ktas_level <= 2, f"Pneumothorax must result in KTAS 1 or 2, got {res.final_ktas_level}"
    assert (
        "CRITICAL" in res.triage_status_badge
        or "ESCALATED" in res.triage_status_badge
        or res.escalation_reason is not None
    ), "Pneumothorax must trigger critical badge or escalation reason"


def test_decoupled_radiographic_clear_non_escalation(e2e_orchestrator):
    """Verify that normal clear radiograph (E2E-14) preserves subacute baseline without escalation."""
    matrix = get_e2e_stress_test_matrix()
    clear_case = next(c for c in matrix if c["id"] == "E2E-14")

    res: TriageResult = e2e_orchestrator.triage_patient(clear_case["record"], generate_plots=False)
    assert res.xray_evaluated, "Chest X-ray should be evaluated"
    assert res.escalation_reason is None, "Clear radiograph must not trigger escalation"
    assert res.triage_status_badge in ["IMAGING_CLEAR", "CONCORDANT_INTAKE", "PATHOLOGY_NOTED"]
    assert res.final_ktas_level >= 3, f"Non-acute patient should remain subacute, got {res.final_ktas_level}"


def test_deterministic_reproducibility(e2e_orchestrator):
    """Verify that multiple passes of the same intake record yield stable classifications."""
    record = PatientIntakeRecord(
        patient_id="REPRO-TEST-01",
        age=52.0,
        sex="Male",
        arrival_mode="119",
        injury="Non-trauma",
        mental_status="Alert",
        sbp=138.0,
        dbp=86.0,
        hr=90.0,
        rr=19.0,
        bt=36.7,
        saturation=96.0,
        pain=1,
        nrs_pain=7.0,
        chief_complaint_text="substernal chest pain radiating to left jaw",
    )

    res1 = e2e_orchestrator.triage_patient(record, generate_plots=False)
    res2 = e2e_orchestrator.triage_patient(record, generate_plots=False)

    assert res1.final_ktas_level == res2.final_ktas_level
    assert np.isclose(res1.confidence, res2.confidence, atol=0.08)
    assert np.isclose(res1.predictive_entropy, res2.predictive_entropy, atol=0.08)
