"""
Unit Tests for Cross-Modal Evidence & Conflict Engine (CECE).
=============================================================
Tests physiological anomaly auditing, cross-modal semantic contradiction detection,
linked clinical corroboration, missingness tracking, and human review triggers.
"""

from pathlib import Path
import sys
import pytest

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integration.cece import (
    ClinicalConflict,
    CrossModalEvidenceConflictEngine,
    EvidenceReport,
    LinkedCorroboration,
    PhysiologicalValidator,
)


@pytest.fixture
def cece_engine() -> CrossModalEvidenceConflictEngine:
    return CrossModalEvidenceConflictEngine()


def test_physiological_inverted_blood_pressure():
    """Verify inverted BP (SBP < DBP) triggers critical physiological conflict."""
    alerts, conflicts, indices = PhysiologicalValidator.audit_vitals(
        sbp=70.0, dbp=95.0, hr=80.0, rr=16.0, bt=36.8, spo2=98.0
    )
    assert any(c.conflict_id == "BP-INVERTED" for c in conflicts)
    assert any(c.severity == "CRITICAL" for c in conflicts)


def test_physiological_severe_shock_index():
    """Verify severe Shock Index (HR 140 / SBP 80 = 1.75) triggers critical conflict."""
    alerts, conflicts, indices = PhysiologicalValidator.audit_vitals(
        sbp=80.0, dbp=50.0, hr=140.0, rr=24.0, bt=37.0, spo2=95.0
    )
    assert any(c.conflict_id == "SHOCK-INDEX-SEVERE" for c in conflicts)
    assert indices["shock_index"] >= 1.30


def test_cece_pain_discordance(cece_engine: CrossModalEvidenceConflictEngine):
    """Verify patient denying pain in text with recorded NRS 9/10 triggers discordance."""
    report = cece_engine.audit_intake(
        chief_complaint="Patient denies chest pain and denies any pain.",
        sbp=120.0, dbp=80.0, hr=75.0, rr=16.0, bt=36.7, saturation=99.0,
        nrs_pain=9.0, pain_flag=1,
    )
    assert any(c.conflict_id == "PAIN-NEG-VS-HIGH-NRS" for c in report.conflicts)
    assert report.conflict_severity in ["MODERATE", "CRITICAL"]
    assert report.requires_human_review is True


def test_cece_silent_hypoxia(cece_engine: CrossModalEvidenceConflictEngine):
    """Verify silent hypoxia (SpO2 81% without dyspnea text) is detected as critical."""
    report = cece_engine.audit_intake(
        chief_complaint="Mild dizziness and nausea after breakfast.",
        sbp=118.0, dbp=76.0, hr=82.0, rr=16.0, bt=36.6, saturation=81.0,
    )
    assert any(c.conflict_id == "SILENT-HYPOXIA" for c in report.conflicts)
    assert report.conflict_severity == "CRITICAL"
    assert report.human_review_priority == "IMMEDIATE_RED_FLAG"


def test_cece_respiratory_corroboration(cece_engine: CrossModalEvidenceConflictEngine):
    """Verify acute shortness of breath corroborated by hypoxia creates linked corroboration."""
    report = cece_engine.audit_intake(
        chief_complaint="Acute shortness of breath and wheezing for 1 hour.",
        sbp=125.0, dbp=80.0, hr=110.0, rr=28.0, bt=37.1, saturation=88.0,
    )
    assert any(cor.corroboration_id == "RESP-CORROBORATION" for cor in report.linked_corroborations)


def test_cece_occult_shock_with_benign_text(cece_engine: CrossModalEvidenceConflictEngine):
    """Verify benign text with severe shock triggers critical conflict and text override recommendation."""
    report = cece_engine.audit_intake(
        chief_complaint="Prescription refill and mild scratch on finger.",
        sbp=65.0, dbp=40.0, hr=135.0, rr=26.0, bt=36.0, saturation=94.0,
    )
    assert any(c.conflict_id == "OCCULT-SHOCK-BENIGN-TEXT" for c in report.conflicts)
    assert report.conflict_severity == "CRITICAL"


def test_cece_fever_denial_pyrexia(cece_engine: CrossModalEvidenceConflictEngine):
    """Verify fever denial with core BT 39.4°C triggers thermal conflict."""
    report = cece_engine.audit_intake(
        chief_complaint="Patient denies fever, mild cough.",
        sbp=120.0, dbp=80.0, hr=95.0, rr=18.0, bt=39.4, saturation=97.0,
    )
    assert any(c.conflict_id == "FEVER-DENIAL-PYREXIA" for c in report.conflicts)


def test_cece_trauma_context_mismatch(cece_engine: CrossModalEvidenceConflictEngine):
    """Verify trauma complaint with non-trauma context triggers context mismatch conflict."""
    report = cece_engine.audit_intake(
        chief_complaint="High speed motor vehicle accident with rollover.",
        sbp=130.0, dbp=85.0, hr=90.0, rr=18.0, bt=36.8, saturation=98.0,
        injury="Non-trauma",
    )
    assert any(c.conflict_id == "TRAUMA-CONTEXT-MISMATCH" for c in report.conflicts)


def test_cece_conditioning_vector_structure(cece_engine: CrossModalEvidenceConflictEngine):
    """Verify CECE conditioning vector produces 8-dimensional valid numeric tensor."""
    report = cece_engine.audit_intake(
        chief_complaint="Chest pain radiating to left arm.",
        sbp=120.0, dbp=80.0, hr=90.0, rr=18.0, bt=37.0, saturation=96.0,
    )
    assert isinstance(report.cece_conditioning_vector, list)
    assert len(report.cece_conditioning_vector) == 8
    assert all(isinstance(v, (int, float)) for v in report.cece_conditioning_vector)
    assert 0.0 <= report.overall_concordance_score <= 1.0

