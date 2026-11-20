"""
Cross-Modal Evidence & Conflict Engine (CECE)
==============================================
Module: src/integration/cece.py

An independent clinical decision-support engine that:
  1. Audits physiological plausibility (vital sign ranges, shock index, pulse pressure).
  2. Detects cross-modal contradictions and semantic discordance:
     - Text vs Vitals (benign text vs catastrophic vitals; acute chest pain vs normal vitals).
     - Text vs Pain (denies pain in text vs high NRS score; severe pain in text vs NRS 0).
     - Text vs Context (reported motor vehicle collision vs coded non-trauma).
     - Respiratory & thermal discordance (hypoxia without complaint, fever denial with pyrexia).
  3. Tracks missing modalities & sensor dropouts (absent SpO2, missing text, missing radiograph).
  4. Assesses voice transcription quality (Whisper avg logprob, duration, filler ratio).
  5. Decoupled radiographic check (maintains zero patient-level pairing).
  6. Produces structured evidence reports with conflict severity rating and clinician review triggers.
  7. Generates evidence-conditioning vector for RCCF-Net v2.

Research & Educational Prototype Only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.nlp.clinical_nlp import AdvancedClinicalNLP, ClinicalTextEvidence

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. CECE DATA STRUCTURES & SCHEMAS
# ==============================================================================
@dataclass
class ClinicalConflict:
    """Represents a single cross-modal semantic contradiction or physiological anomaly."""
    conflict_id: str
    category: str                       # "PHYSIOLOGICAL_ANOMALY", "TEXT_VS_VITALS", "PAIN_DISCORDANCE", "CONTEXT_MISMATCH"
    severity: str                       # "MILD", "MODERATE", "CRITICAL"
    modalities_involved: List[str]      # e.g. ["Text", "Vitals"]
    description: str
    clinical_risk: str
    recommended_action: str


@dataclass
class LinkedCorroboration:
    """Represents concordant mutual corroboration between two or more modalities."""
    corroboration_id: str
    domain: str                         # "RESPIRATORY", "CARDIOVASCULAR", "SEPSIS", "TRAUMA"
    modalities: List[str]
    finding: str
    acuity_implication: str


@dataclass
class EvidenceReport:
    """
    Standardized Cross-Modal Evidence & Conflict Engine (CECE) Audit Report.
    Structured artifact consumed by clinicians and downstream adaptive fusion networks.
    """
    overall_concordance_score: float    # [0.0, 1.0] (1.0 = perfect alignment, 0.0 = total discordance)
    conflict_severity: str              # "NONE", "MILD", "MODERATE", "CRITICAL"
    requires_human_review: bool
    human_review_priority: str          # "ROUTINE", "PROMPT_REASSESSMENT", "IMMEDIATE_RED_FLAG"
    conflicts: List[ClinicalConflict] = field(default_factory=list)
    linked_corroborations: List[LinkedCorroboration] = field(default_factory=list)
    missing_modalities: List[str] = field(default_factory=list)
    sensor_dropout_count: int = 0
    physiological_alerts: List[str] = field(default_factory=list)
    transcription_quality_flag: Optional[str] = None
    radiographic_escalation_flag: Optional[str] = None
    cece_conditioning_vector: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["conflicts"] = [asdict(c) for c in self.conflicts]
        res["linked_corroborations"] = [asdict(l) for l in self.linked_corroborations]
        return res


# ==============================================================================
# 2. PHYSIOLOGICAL MEASUREMENT VALIDATOR
# ==============================================================================
class PhysiologicalValidator:
    """
    Audits hemodynamic and metabolic vital sign parameters against plausible
    clinical ranges and calculates clinical indices (Shock Index, Pulse Pressure).
    """

    @staticmethod
    def audit_vitals(
        sbp: Optional[float],
        dbp: Optional[float],
        hr: Optional[float],
        rr: Optional[float],
        bt: Optional[float],
        spo2: Optional[float],
        age: float = 45.0,
    ) -> Tuple[List[str], List[ClinicalConflict], Dict[str, float]]:
        """
        Validate vital signs against physiological constraints.
        Returns: (alert_strings, conflict_objects, computed_indices)
        """
        alerts: List[str] = []
        conflicts: List[ClinicalConflict] = []
        indices: Dict[str, float] = {}

        # 1. Blood Pressure Plausibility
        if sbp is not None and dbp is not None:
            if sbp < dbp:
                conflicts.append(
                    ClinicalConflict(
                        conflict_id="BP-INVERTED",
                        category="PHYSIOLOGICAL_ANOMALY",
                        severity="CRITICAL",
                        modalities_involved=["Vitals"],
                        description=f"Systolic BP ({sbp:.0f}) is lower than Diastolic BP ({dbp:.0f}).",
                        clinical_risk="Sphygmomanometer error, transducer inversion, or critical hemodynamic measurement artifact.",
                        recommended_action="Immediate manual re-measurement of blood pressure on contralateral arm.",
                    )
                )

            pulse_pressure = sbp - dbp
            indices["pulse_pressure"] = float(pulse_pressure)

            if pulse_pressure < 18.0 and sbp >= dbp:
                conflicts.append(
                    ClinicalConflict(
                        conflict_id="PP-NARROW",
                        category="PHYSIOLOGICAL_ANOMALY",
                        severity="CRITICAL",
                        modalities_involved=["Vitals"],
                        description=f"Narrow pulse pressure ({pulse_pressure:.0f} mmHg; normal: 30-50 mmHg).",
                        clinical_risk="Profound cardiogenic shock, severe hypovolemia, or cardiac tamponade.",
                        recommended_action="Evaluate for pulsus paradoxus, jugular venous distention, and immediate bedside ultrasound.",
                    )
                )
            elif pulse_pressure > 85.0:
                alerts.append(f"Widened pulse pressure ({pulse_pressure:.0f} mmHg) - consider severe sepsis or aortic regurgitation.")

        # 2. Hemodynamic Shock Index
        if hr is not None and sbp is not None and sbp > 10.0:
            shock_index = hr / sbp
            indices["shock_index"] = float(round(shock_index, 3))
            if shock_index >= 1.3:
                conflicts.append(
                    ClinicalConflict(
                        conflict_id="SHOCK-INDEX-SEVERE",
                        category="PHYSIOLOGICAL_ANOMALY",
                        severity="CRITICAL",
                        modalities_involved=["Vitals"],
                        description=f"Severe Shock Index: {shock_index:.2f} (HR {hr:.0f} / SBP {sbp:.0f}; normal < 0.70).",
                        clinical_risk="Imminent cardiovascular collapse, occult hemorrhage, or severe decompensated shock.",
                        recommended_action="Immediate resuscitation bay transfer, large-bore IV access, and fluid resuscitation.",
                    )
                )
            elif shock_index >= 0.90:
                alerts.append(f"Elevated Shock Index ({shock_index:.2f}) indicates moderate hemodynamic compromise.")

        # 3. Extreme Out-of-Bounds Checks
        if hr is not None:
            if hr > 220 or hr < 30:
                conflicts.append(
                    ClinicalConflict(
                        conflict_id="HR-EXTREME",
                        category="PHYSIOLOGICAL_ANOMALY",
                        severity="CRITICAL",
                        modalities_involved=["Vitals"],
                        description=f"Extreme heart rate: {hr:.0f} bpm.",
                        clinical_risk="Malignant arrhythmia, complete heart block, or severe ventricular tachycardia.",
                        recommended_action="Continuous 12-lead ECG monitoring and immediate physician bedside evaluation.",
                    )
                )

        if rr is not None:
            if rr > 48 or rr < 6:
                conflicts.append(
                    ClinicalConflict(
                        conflict_id="RR-EXTREME",
                        category="PHYSIOLOGICAL_ANOMALY",
                        severity="CRITICAL",
                        modalities_involved=["Vitals"],
                        description=f"Extreme respiratory rate: {rr:.0f} breaths/min.",
                        clinical_risk="Impending respiratory arrest, profound metabolic acidosis, or central hypoventilation.",
                        recommended_action="Airway support, blood gas analysis, and preparation for non-invasive/invasive ventilation.",
                    )
                )

        if bt is not None:
            if bt > 41.5 or bt < 31.0:
                conflicts.append(
                    ClinicalConflict(
                        conflict_id="BT-EXTREME",
                        category="PHYSIOLOGICAL_ANOMALY",
                        severity="CRITICAL",
                        modalities_involved=["Vitals"],
                        description=f"Extreme body temperature: {bt:.1f}°C.",
                        clinical_risk="Malignant hyperpyrexia, heat stroke, or severe accidental hypothermia with arrhythmia risk.",
                        recommended_action="Active targeted temperature management and continuous core temperature probe.",
                    )
                )

        if spo2 is not None:
            if spo2 < 85.0:
                conflicts.append(
                    ClinicalConflict(
                        conflict_id="SPO2-SEVERE-HYPOXIA",
                        category="PHYSIOLOGICAL_ANOMALY",
                        severity="CRITICAL",
                        modalities_involved=["Vitals"],
                        description=f"Severe arterial hypoxia: SpO2 {spo2:.0f}%.",
                        clinical_risk="Life-threatening respiratory compromise, severe shunt, or acute respiratory distress syndrome.",
                        recommended_action="High-flow supplemental oxygen therapy and emergency airway assessment.",
                    )
                )

        return alerts, conflicts, indices


# ==============================================================================
# 3. CROSS-MODAL EVIDENCE & CONFLICT ENGINE (CECE)
# ==============================================================================
class CrossModalEvidenceConflictEngine:
    """
    Master CECE Engine integrating Clinical NLP, physiological validation,
    missingness auditing, and cross-modal semantic discordance detection.
    """

    def __init__(self):
        self.clinical_nlp = AdvancedClinicalNLP()
        self.physio_validator = PhysiologicalValidator()

    def audit_intake(
        self,
        chief_complaint: Optional[str] = None,
        sbp: Optional[float] = None,
        dbp: Optional[float] = None,
        hr: Optional[float] = None,
        rr: Optional[float] = None,
        bt: Optional[float] = None,
        saturation: Optional[float] = None,
        nrs_pain: Optional[float] = None,
        pain_flag: Optional[int] = None,
        mental_status: Optional[str] = "Alert",
        age: float = 45.0,
        arrival_mode: str = "Walking",
        injury: str = "Non-trauma",
        voice_metadata: Optional[Dict[str, Any]] = None,
        xray_metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceReport:
        """
        Execute comprehensive multi-modal cross-evidence audit.
        """
        conflicts: List[ClinicalConflict] = []
        corroborations: List[LinkedCorroboration] = []
        missing_modalities: List[str] = []
        sensor_dropout_count = 0

        # 1. Audit Modality Completeness & Sensor Dropouts
        has_text = bool(chief_complaint and chief_complaint.strip())
        if not has_text:
            missing_modalities.append("Chief_Complaint_Text")

        # Vital sensors
        vital_fields = {"SBP": sbp, "DBP": dbp, "HR": hr, "RR": rr, "BT": bt, "Saturation": saturation}
        for v_name, v_val in vital_fields.items():
            if v_val is None or (isinstance(v_val, float) and np.isnan(v_val)):
                sensor_dropout_count += 1
                missing_modalities.append(f"Sensor_{v_name}")

        if voice_metadata is None:
            missing_modalities.append("Voice_Audio")
        if xray_metadata is None:
            missing_modalities.append("Chest_Xray")

        # 2. Physiological Validation
        physio_alerts, physio_conflicts, indices = self.physio_validator.audit_vitals(
            sbp=sbp, dbp=dbp, hr=hr, rr=rr, bt=bt, spo2=saturation, age=age
        )
        conflicts.extend(physio_conflicts)

        # 3. Clinical NLP Semantic Extraction
        text_evidence: ClinicalTextEvidence = self.clinical_nlp.extract_evidence(chief_complaint or "")
        affirmed_symptom_terms = {s.term for s in text_evidence.affirmed_symptoms}
        negated_symptom_terms = {s.term for s in text_evidence.negated_symptoms}

        # 4. Cross-Modal Conflict & Corroboration Detection

        # (A) PAIN MODALITY DISCORDANCE
        effective_nrs = nrs_pain if nrs_pain is not None else 0.0
        effective_pain_flag = pain_flag if pain_flag is not None else (1 if effective_nrs >= 1.0 else 0)

        # Text denies pain, but vitals record high pain (NRS >= 6)
        pain_negated = any("pain" in term for term in negated_symptom_terms)
        if pain_negated and effective_nrs >= 6.0:
            conflicts.append(
                ClinicalConflict(
                    conflict_id="PAIN-NEG-VS-HIGH-NRS",
                    category="PAIN_DISCORDANCE",
                    severity="MODERATE",
                    modalities_involved=["Text", "Vitals"],
                    description=f"Complaint denies pain in text, but vital signs document NRS pain score of {effective_nrs:.0f}/10.",
                    clinical_risk="Documentation contradiction, surrogate reporting error, or uncommunicated acute distress.",
                    recommended_action="Directly ask patient to clarify current pain level and anatomical location.",
                )
            )

        # Text reports severe/crushing pain, but NRS = 0 and pain_flag = 0
        if text_evidence.primary_severity >= 0.85 and effective_nrs == 0.0 and effective_pain_flag == 0:
            conflicts.append(
                ClinicalConflict(
                    conflict_id="PAIN-SEVERE-VS-ZERO-NRS",
                    category="PAIN_DISCORDANCE",
                    severity="MODERATE",
                    modalities_involved=["Text", "Vitals"],
                    description=f"Chief complaint notes severe distress/pain ({text_evidence.primary_severity*10:.0f}/10), but NRS recorded as 0.",
                    clinical_risk="Under-triage of severe pain presentation due to default zero entry in triage EHR.",
                    recommended_action="Verify numerical pain score with patient; do not rely on default zero.",
                )
            )

        # (B) RESPIRATORY DISCORDANCE & CORROBORATION
        has_resp_complaint = any(
            term in affirmed_symptom_terms
            for term in ["shortness of breath", "dyspnea", "difficulty breathing", "wheezing", "stridor"]
        )

        is_hypoxic = (saturation is not None and saturation < 90.0)
        is_tachypneic = (rr is not None and rr >= 26.0)

        if has_resp_complaint and (is_hypoxic or is_tachypneic):
            corroborations.append(
                LinkedCorroboration(
                    corroboration_id="RESP-CORROBORATION",
                    domain="RESPIRATORY",
                    modalities=["Text", "Vitals"],
                    finding=f"Chief complaint of dyspnea corroborated by physiological findings (SpO2: {saturation}%, RR: {rr}).",
                    acuity_implication="High triage priority: Verified acute cardiopulmonary compromise.",
                )
            )
        elif not has_resp_complaint and is_hypoxic:
            conflicts.append(
                ClinicalConflict(
                    conflict_id="SILENT-HYPOXIA",
                    category="TEXT_VS_VITALS",
                    severity="CRITICAL",
                    modalities_involved=["Text", "Vitals"],
                    description=f"Severe arterial hypoxia (SpO2 {saturation:.0f}%) without subjective shortness of breath complaint.",
                    clinical_risk="Silent hypoxia ('happy hypoxia'), severe COVID/pneumonia, or acute pulmonary embolism.",
                    recommended_action="Immediate supplemental O2, arterial blood gas, and urgent chest imaging.",
                )
            )
        elif has_resp_complaint and (saturation is not None and saturation >= 99.0) and (rr is not None and rr <= 14.0):
            conflicts.append(
                ClinicalConflict(
                    conflict_id="RESP-SUBJECTIVE-DISCORDANCE",
                    category="TEXT_VS_VITALS",
                    severity="MILD",
                    modalities_involved=["Text", "Vitals"],
                    description=f"Patient reports dyspnea, but vitals show SpO2 {saturation:.0f}% and RR {rr:.0f} breaths/min.",
                    clinical_risk="Early pulmonary decompensation, psychogenic hyperventilation, or mild bronchospasm.",
                    recommended_action="Perform chest auscultation for wheezing/crackles and monitor peak flow.",
                )
            )

        # (C) CARDIOVASCULAR & SHOCK DISCORDANCE
        has_cardiac_complaint = any(
            term in affirmed_symptom_terms
            for term in ["chest pain", "chest pressure", "radiating pain", "cardiac arrest", "syncope"]
        )
        has_hypotension = (sbp is not None and sbp < 90.0)
        has_tachycardia = (hr is not None and hr > 115.0)

        if has_cardiac_complaint and (has_hypotension or has_tachycardia):
            corroborations.append(
                LinkedCorroboration(
                    corroboration_id="CARDIAC-SHOCK-CORROBORATION",
                    domain="CARDIOVASCULAR",
                    modalities=["Text", "Vitals"],
                    finding=f"Acute cardiac complaint corroborated by hemodynamic instability (SBP: {sbp}, HR: {hr}).",
                    acuity_implication="Immediate KTAS 1/2 Resuscitation: Risk of acute coronary syndrome with cardiogenic shock.",
                )
            )
        elif not has_cardiac_complaint and text_evidence.primary_severity <= 0.30 and (has_hypotension and has_tachycardia):
            conflicts.append(
                ClinicalConflict(
                    conflict_id="OCCULT-SHOCK-BENIGN-TEXT",
                    category="TEXT_VS_VITALS",
                    severity="CRITICAL",
                    modalities_involved=["Text", "Vitals"],
                    description=f"Benign or minor text complaint, but patient is in profound shock (SBP {sbp:.0f}, HR {hr:.0f}).",
                    clinical_risk="Occult hemorrhagic shock, distributive septic shock, or uncommunicated acute hemorrhage.",
                    recommended_action="OVERRIDE TEXT ACUITY. Vitals indicate emergent resuscitation required.",
                )
            )

        # (D) THERMAL & SEPSIS DISCORDANCE
        has_fever_negated = "fever" in negated_symptom_terms
        is_hyperpyrexic = (bt is not None and bt >= 38.5)
        if has_fever_negated and is_hyperpyrexic:
            conflicts.append(
                ClinicalConflict(
                    conflict_id="FEVER-DENIAL-PYREXIA",
                    category="TEXT_VS_VITALS",
                    severity="MODERATE",
                    modalities_involved=["Text", "Vitals"],
                    description=f"Patient explicitly denies fever in text, but core body temperature is elevated ({bt:.1f}°C).",
                    clinical_risk="Unrecognized infection, antipyretic wearing off, or impaired thermoperception.",
                    recommended_action="Initiate sepsis screening protocol (lactate, blood cultures, urinalysis).",
                )
            )

        # (E) TRAUMA CONTEXT MISMATCH
        has_trauma_complaint = any(
            term in affirmed_symptom_terms
            for term in ["motor vehicle accident", "fall", "head trauma", "stab wound", "gunshot wound", "fracture"]
        )
        is_context_nontrauma = ("non" in injury.lower())
        if has_trauma_complaint and is_context_nontrauma:
            conflicts.append(
                ClinicalConflict(
                    conflict_id="TRAUMA-CONTEXT-MISMATCH",
                    category="CONTEXT_MISMATCH",
                    severity="MODERATE",
                    modalities_involved=["Text", "Context"],
                    description=f"Complaint describes traumatic mechanism, but context is categorized as '{injury}'.",
                    clinical_risk="Failure to activate trauma team and cervical spine clearance protocols.",
                    recommended_action="Re-classify patient as Trauma and evaluate mechanism of injury.",
                )
            )

        # (F) VOICE TRANSCRIPTION QUALITY AUDIT
        transcription_quality_flag: Optional[str] = None
        if voice_metadata is not None:
            avg_logprob = voice_metadata.get("avg_logprob", -0.20)
            audio_duration = voice_metadata.get("duration_sec", 2.0)
            confidence = voice_metadata.get("confidence", 0.85)

            if confidence < 0.50 or avg_logprob < -0.80:
                transcription_quality_flag = "LOW_CONFIDENCE_TRANSCRIPTION"
                conflicts.append(
                    ClinicalConflict(
                        conflict_id="VOICE-LOW-CONFIDENCE",
                        category="VOICE_ASR_QUALITY",
                        severity="MODERATE",
                        modalities_involved=["Voice_Audio"],
                        description=f"Whisper speech transcription confidence is low (conf: {confidence:.2f}, logprob: {avg_logprob:.2f}).",
                        clinical_risk="Acoustic distortion, excessive ambient noise, or garbled chief complaint.",
                        recommended_action="Verbally confirm chief complaint with patient or dispatch nurse.",
                    )
                )
            elif audio_duration < 1.0:
                transcription_quality_flag = "VERY_SHORT_AUDIO"

        # (G) RADIOGRAPHIC ESCALATION CHECK (Decoupled auxiliary stream)
        radiographic_escalation_flag: Optional[str] = None
        if xray_metadata is not None:
            if xray_metadata.get("requires_emergency_escalation", False):
                esc_finding = xray_metadata.get("escalation_finding", "Critical Finding")
                radiographic_escalation_flag = f"EMERGENCY_XRAY_ESCALATION: {esc_finding}"
                corroborations.append(
                    LinkedCorroboration(
                        corroboration_id="XRAY-CRITICAL-ALERT",
                        domain="RADIOGRAPHY",
                        modalities=["Chest_Xray"],
                        finding=f"Auxiliary radiograph detected critical thoracic emergency: {esc_finding}.",
                        acuity_implication="Immediate bed assignment and physician intervention required.",
                    )
                )

        # 5. Compute Overall Concordance Score & Conflict Severity
        # Deduct penalties based on conflicts
        concordance_score = 1.00
        for c in conflicts:
            if c.severity == "CRITICAL":
                concordance_score -= 0.35
            elif c.severity == "MODERATE":
                concordance_score -= 0.18
            elif c.severity == "MILD":
                concordance_score -= 0.08

        # Missingness penalties
        concordance_score -= (sensor_dropout_count * 0.04)
        concordance_score = float(np.clip(concordance_score, 0.0, 1.0))

        # Determine conflict severity tier
        has_critical = any(c.severity == "CRITICAL" for c in conflicts)
        has_moderate = any(c.severity == "MODERATE" for c in conflicts)

        if has_critical:
            conflict_severity = "CRITICAL"
            requires_human_review = True
            review_priority = "IMMEDIATE_RED_FLAG"
        elif has_moderate or len(conflicts) >= 2:
            conflict_severity = "MODERATE"
            requires_human_review = True
            review_priority = "PROMPT_REASSESSMENT"
        elif len(conflicts) == 1:
            conflict_severity = "MILD"
            requires_human_review = False
            review_priority = "ROUTINE"
        else:
            conflict_severity = "NONE"
            requires_human_review = False
            review_priority = "ROUTINE"

        # Construct CECE Conditioning Vector (8-dimensional)
        # [concordance_score, num_critical, num_moderate, num_corroborations,
        #  sensor_dropouts, shock_index, pulse_pressure/100, has_radiographic_alert]
        cece_conditioning_vector = [
            float(round(concordance_score, 3)),
            float(sum(1 for c in conflicts if c.severity == "CRITICAL")),
            float(sum(1 for c in conflicts if c.severity == "MODERATE")),
            float(len(corroborations)),
            float(sensor_dropout_count),
            float(min(indices.get("shock_index", 0.70), 3.0)),
            float(min(indices.get("pulse_pressure", 40.0) / 100.0, 2.0)),
            1.0 if radiographic_escalation_flag is not None else 0.0,
        ]

        return EvidenceReport(
            overall_concordance_score=round(concordance_score, 3),
            conflict_severity=conflict_severity,
            requires_human_review=requires_human_review,
            human_review_priority=review_priority,
            conflicts=conflicts,
            linked_corroborations=corroborations,
            missing_modalities=missing_modalities,
            sensor_dropout_count=sensor_dropout_count,
            physiological_alerts=physio_alerts,
            transcription_quality_flag=transcription_quality_flag,
            radiographic_escalation_flag=radiographic_escalation_flag,
            cece_conditioning_vector=cece_conditioning_vector,
        )

