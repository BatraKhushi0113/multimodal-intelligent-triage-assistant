"""
Unit Tests for Module 16: System Integration Layer.
===================================================
Tests PatientIntakeRecord validation, TriageOrchestrator cross-modal pipeline,
text-based triage, voice-first audio transcription, missing modality robustness,
uncertainty estimation, decoupled chest X-ray acuity modifiers, and batch processing.
"""

from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.triage_orchestrator import (
    KTAS_DESCRIPTIONS,
    KTAS_LEVEL_NAMES,
    PatientIntakeRecord,
    TriageOrchestrator,
    TriageResult,
)
from src.vision.xray_model import generate_synthetic_chest_xray
from src.voice.speech_to_text import generate_synthetic_emergency_call


@pytest.fixture(scope="module")
def orchestrator():
    """Module-level TriageOrchestrator fixture."""
    orch = TriageOrchestrator(
        config_path="config.yaml",
        device="cpu",
        load_whisper=False,  # Lazy loaded on demand
        load_vision=True,
    )
    return orch


def test_patient_intake_record_initialization():
    """Verify PatientIntakeRecord fields, defaults, and dictionary conversion."""
    rec = PatientIntakeRecord(
        patient_id="TEST-001",
        age=45.0,
        sex="Female",
        arrival_mode="Walking",
        injury="Non-trauma",
        mental_status="Alert",
        sbp=125.0,
        dbp=82.0,
        hr=76.0,
        rr=16.0,
        bt=36.6,
        saturation=98.0,
        pain=1,
        nrs_pain=5.0,
        chief_complaint_text="intermittent epigastric pain",
    )
    assert rec.patient_id == "TEST-001"
    assert rec.age == 45.0
    assert rec.sbp == 125.0
    assert rec.saturation == 98.0


def test_triage_orchestrator_components(orchestrator):
    """Verify all deep learning encoders and safety modules are attached."""
    assert orchestrator.vitals_encoder is not None
    assert orchestrator.context_encoder is not None
    assert orchestrator.fusion_model is not None
    assert orchestrator.mc_predictor is not None
    assert orchestrator.safety_auditor is not None
    assert orchestrator._gradcam_pipeline is not None


def test_text_chief_complaint_triage(orchestrator):
    """Verify end-to-end triage prediction on a standard text-based patient intake."""
    record = PatientIntakeRecord(
        patient_id="TEST-PT-TEXT",
        age=62.0,
        sex="Male",
        arrival_mode="119",
        injury="Non-trauma",
        mental_status="Alert",
        sbp=165.0,
        dbp=98.0,
        hr=110.0,
        rr=22.0,
        bt=36.7,
        saturation=95.0,
        pain=1,
        nrs_pain=8.0,
        chief_complaint_text="severe crushing substernal chest pressure with cold sweat",
    )

    result = orchestrator.triage_patient(record, generate_plots=False)

    assert isinstance(result, TriageResult)
    assert result.patient_id == "TEST-PT-TEXT"
    assert 1 <= result.intake_ktas_level <= 5
    assert 1 <= result.final_ktas_level <= 5
    assert len(result.intake_probabilities) == 5
    assert np.isclose(sum(result.intake_probabilities), 1.0, atol=1e-3)
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.predictive_entropy <= 1.0
    assert len(result.top_clinical_drivers) > 0
    assert result.modality_weights["Text"] > 0.0
    assert result.modality_weights["Vitals"] > 0.0
    assert result.modality_weights["Context"] > 0.0
    assert not result.xray_evaluated


def test_missing_modality_robustness(orchestrator):
    """Verify missing physiological measurements (e.g. unmeasured SpO2/Pain) run safely."""
    record = PatientIntakeRecord(
        patient_id="TEST-PT-MISSING",
        age=30.0,
        sex="Female",
        arrival_mode="Walking",
        injury="Non-trauma",
        mental_status="Alert",
        sbp=118.0,
        dbp=76.0,
        hr=72.0,
        rr=16.0,
        bt=36.6,
        saturation=None,          # Missing SpO2 pulse oximeter
        pain=0,
        nrs_pain=None,            # Missing pain score
        chief_complaint_text="mild rhinitis and seasonal congestion",
    )

    result = orchestrator.triage_patient(record, generate_plots=False)

    assert isinstance(result, TriageResult)
    assert not np.isnan(result.confidence)
    assert not np.isnan(result.predictive_entropy)
    assert 1 <= result.intake_ktas_level <= 5
    assert len(result.intake_probabilities) == 5
    assert np.isclose(sum(result.intake_probabilities), 1.0, atol=1e-3)


def test_voice_audio_triage(orchestrator):
    """Verify voice audio intake via numpy array transcribes and triggers triage prediction."""
    # Synthesize realistic audio call
    audio_arr = generate_synthetic_emergency_call(duration_sec=2.0, sample_rate=16000)

    record = PatientIntakeRecord(
        patient_id="TEST-PT-VOICE",
        age=50.0,
        sex="Male",
        arrival_mode="119",
        injury="Non-trauma",
        mental_status="Alert",
        sbp=140.0,
        dbp=88.0,
        hr=90.0,
        rr=18.0,
        bt=36.8,
        saturation=97.0,
        pain=1,
        nrs_pain=6.0,
        chief_complaint_text="dyspnea and chest pain",
        voice_audio_array=audio_arr,
    )

    result = orchestrator.triage_patient(record, generate_plots=False)

    assert isinstance(result, TriageResult)
    assert result.voice_transcription_used is True
    assert len(result.chief_complaint_analyzed) > 0
    assert 1 <= result.final_ktas_level <= 5


def test_decoupled_radiograph_critical_escalation(orchestrator):
    """Verify acute tension pneumothorax detected on X-ray modifies/escalates triage acuity."""
    xray_img = generate_synthetic_chest_xray("pneumothorax", image_size=224, random_seed=902)

    record = PatientIntakeRecord(
        patient_id="TEST-PT-XRAY-CRITICAL",
        age=22.0,
        sex="Male",
        arrival_mode="Walking",
        injury="Non-trauma",
        mental_status="Alert",
        sbp=115.0,
        dbp=75.0,
        hr=88.0,
        rr=18.0,
        bt=36.6,
        saturation=96.0,
        pain=1,
        nrs_pain=4.0,
        chief_complaint_text="sharp localized right pleuritic pain",
        chest_xray_input=xray_img,
    )

    result = orchestrator.triage_patient(record, generate_plots=False)

    assert result.xray_evaluated is True
    assert result.xray_acuity_tier in ("CRITICAL_ESCALATION", "URGENT_EVALUATION")
    assert result.primary_xray_finding == "Pneumothorax"
    assert result.xray_confidence is not None and result.xray_confidence > 0.85
    assert result.radiographic_severity_index is not None and result.radiographic_severity_index >= 0.80
    assert "CRITICAL" in result.triage_status_badge or "ESCALATED" in result.triage_status_badge


def test_decoupled_radiograph_concordant_clear(orchestrator):
    """Verify normal unremarkable chest radiograph preserves baseline triage intake level."""
    xray_img = generate_synthetic_chest_xray("normal", image_size=224, random_seed=901)

    record = PatientIntakeRecord(
        patient_id="TEST-PT-XRAY-CLEAR",
        age=26.0,
        sex="Female",
        arrival_mode="Walking",
        injury="Trauma",
        mental_status="Alert",
        sbp=120.0,
        dbp=80.0,
        hr=72.0,
        rr=16.0,
        bt=36.5,
        saturation=99.0,
        pain=1,
        nrs_pain=3.0,
        chief_complaint_text="mild ankle contusion after stumbling",
        chest_xray_input=xray_img,
    )

    result = orchestrator.triage_patient(record, generate_plots=False)

    assert result.xray_evaluated is True
    assert result.final_ktas_level == result.intake_ktas_level
    assert result.triage_status_badge in ("IMAGING_CLEAR", "CONCORDANT_INTAKE", "PATHOLOGY_NOTED")


def test_triage_batch_processing(orchestrator):
    """Verify batch processing over multiple patient records."""
    records = [
        PatientIntakeRecord(
            patient_id="BATCH-01",
            age=70.0,
            sex="Male",
            arrival_mode="119",
            mental_status="Unresponsive",
            sbp=80.0,
            dbp=50.0,
            hr=125.0,
            rr=28.0,
            saturation=88.0,
            chief_complaint_text="unresponsive shock state",
        ),
        PatientIntakeRecord(
            patient_id="BATCH-02",
            age=25.0,
            sex="Female",
            arrival_mode="Walking",
            mental_status="Alert",
            sbp=120.0,
            dbp=80.0,
            hr=70.0,
            rr=16.0,
            saturation=99.0,
            chief_complaint_text="prescription refill request",
        ),
    ]

    results = orchestrator.triage_batch(records, generate_plots=False)

    assert len(results) == 2
    assert results[0].patient_id == "BATCH-01"
    assert results[1].patient_id == "BATCH-02"
    # Acuity of shock state must exceed prescription refill
    assert results[0].final_ktas_level < results[1].final_ktas_level


def test_triage_result_to_dict():
    """Verify serialization of TriageResult to dictionary."""
    result = TriageResult(
        patient_id="TEST-DICT",
        intake_ktas_level=2,
        intake_ktas_name="KTAS 2 (Emergent)",
        intake_probabilities=[0.05, 0.85, 0.05, 0.03, 0.02],
        confidence=0.85,
        predictive_entropy=0.35,
        epistemic_uncertainty=0.04,
        aleatoric_uncertainty=0.31,
        requires_human_review=False,
        safety_alert_message="Routine triage cleared.",
        modality_weights={"Text": 0.4, "Vitals": 0.4, "Context": 0.2},
        top_clinical_drivers=[{"factor": "Tachycardia", "detail": "HR 110", "impact": "High"}],
        chief_complaint_analyzed="chest pain",
        voice_transcription_used=False,
        final_ktas_level=2,
        final_ktas_name="KTAS 2 (Emergent)",
        triage_status_badge="CONCORDANT_INTAKE",
        clinical_recommendation="Emergent care.",
        clinical_narrative="Patient triaged to KTAS 2.",
    )

    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["patient_id"] == "TEST-DICT"
    assert d["intake_ktas_level"] == 2
    assert d["confidence"] == 0.85
