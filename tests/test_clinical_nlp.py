"""
Unit Tests for Advanced Clinical NLP Module.
===========================================
Tests symptom extraction, NegEx-style negation detection, severity parsing,
duration parsing, uncertainty detection, and structured evidence schema.
"""

from pathlib import Path
import sys
import pytest

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.clinical_nlp import AdvancedClinicalNLP, ClinicalTextEvidence, ExtractedSymptom


@pytest.fixture
def clinical_nlp() -> AdvancedClinicalNLP:
    return AdvancedClinicalNLP()


def test_empty_text_handling(clinical_nlp: AdvancedClinicalNLP):
    """Verify empty or None text gracefully returns empty structured evidence."""
    ev1 = clinical_nlp.extract_evidence("")
    assert isinstance(ev1, ClinicalTextEvidence)
    assert len(ev1.affirmed_symptoms) == 0
    assert ev1.primary_severity == 0.0

    ev2 = clinical_nlp.extract_evidence(None)
    assert isinstance(ev2, ClinicalTextEvidence)
    assert len(ev2.affirmed_symptoms) == 0


def test_affirmed_symptom_extraction(clinical_nlp: AdvancedClinicalNLP):
    """Verify explicit high-acuity symptoms are recognized with correct category."""
    text = "Patient presents with acute chest pain and shortness of breath radiating to left arm."
    ev = clinical_nlp.extract_evidence(text)

    affirmed_terms = [s.term for s in ev.affirmed_symptoms]
    assert "chest pain" in affirmed_terms
    assert "shortness of breath" in affirmed_terms
    assert ev.max_urgency_score >= 0.90
    assert ev.has_high_acuity_keywords is True
    assert ev.is_acute_onset is True
    assert any("Left Arm" in site for site in ev.anatomical_sites)


def test_negation_detection(clinical_nlp: AdvancedClinicalNLP):
    """Verify negation cues correctly partition symptoms into negated_symptoms."""
    text = "Patient with severe abdominal pain, but denies fever and denies chest pain."
    ev = clinical_nlp.extract_evidence(text)

    affirmed_terms = [s.term for s in ev.affirmed_symptoms]
    negated_terms = [s.term for s in ev.negated_symptoms]

    assert "abdominal pain" in affirmed_terms
    assert "fever" in negated_terms
    assert "chest pain" in negated_terms
    assert "chest pain" not in affirmed_terms


def test_uncertainty_hedging_detection(clinical_nlp: AdvancedClinicalNLP):
    """Verify hedging cues classify findings into uncertain_symptoms."""
    text = "Bystander reports possible seizure; eval for rule out syncope."
    ev = clinical_nlp.extract_evidence(text)

    uncertain_terms = [s.term for s in ev.uncertain_symptoms]
    assert "seizure" in uncertain_terms or "syncope" in uncertain_terms


def test_severity_extraction(clinical_nlp: AdvancedClinicalNLP):
    """Verify numeric NRS scores and qualitative severity words are parsed."""
    text = "Crushing retrosternal chest pain, rated 9/10, sudden onset."
    ev = clinical_nlp.extract_evidence(text)

    assert ev.primary_severity >= 0.90
    assert ev.is_acute_onset is True
    assert any("Retrosternal" in site for site in ev.anatomical_sites)


def test_duration_parsing(clinical_nlp: AdvancedClinicalNLP):
    """Verify temporal durations are extracted and normalized."""
    text1 = "Severe headache for 3 hours."
    ev1 = clinical_nlp.extract_evidence(text1)
    assert "3 hours" in ev1.duration_summary or ev1.is_acute_onset is True

    text2 = "Mild knee pain for 4 weeks."
    ev2 = clinical_nlp.extract_evidence(text2)
    assert "4 weeks" in ev2.duration_summary
    assert ev2.is_acute_onset is False


def test_evidence_vector_schema(clinical_nlp: AdvancedClinicalNLP):
    """Verify evidence_vector produces valid 8-dimensional numeric representation."""
    text = "Acute dyspnea and wheezing since 30 minutes. Denies chest pain."
    ev = clinical_nlp.extract_evidence(text)

    assert ev.evidence_vector is not None
    assert len(ev.evidence_vector) == 8
    assert all(isinstance(v, (int, float)) for v in ev.evidence_vector)
    # len(affirmed) >= 2, len(negated) >= 1
    assert ev.evidence_vector[0] >= 1.0  # affirmed count
    assert ev.evidence_vector[1] >= 1.0  # negated count

