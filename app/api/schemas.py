"""
Pydantic v2 Request & Response Schemas for Triage Assistant API.
================================================================
Module: app/api/schemas.py

Strict validation schemas enforcing physiological plausibility, clinical
typings, and comprehensive structured outputs for frontend / hospital EHR integrations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# 1. INTAKE REQUEST SCHEMAS
# ==============================================================================
class PatientIntakeRequest(BaseModel):
    """Clinical patient intake payload for automated triage prediction."""
    patient_id: str = Field(default="PT-LIVE-001", description="Unique patient identifier")
    age: float = Field(default=45.0, ge=0.0, le=125.0, description="Age in years")
    sex: str = Field(default="Male", description="Biological sex ('Male' or 'Female')")
    arrival_mode: str = Field(default="Walking", description="Arrival mode ('Walking', '119', 'Private car', 'Other')")
    injury: str = Field(default="Non-trauma", description="Injury classification ('Trauma' or 'Non-trauma')")
    mental_status: str = Field(default="Alert", description="AVPU mental status ('Alert', 'Verbal', 'Pain', 'Unresponsive')")
    patients_per_hour: float = Field(default=3.5, ge=0.0, le=50.0, description="ED census crowding index")

    # Chief complaint (free-text or transcribed voice)
    chief_complaint: Optional[str] = Field(default=None, description="Primary clinical complaint text")

    # Physiological Vitals (nullable for sensor dropout evaluation)
    sbp: Optional[float] = Field(default=120.0, ge=30.0, le=300.0, description="Systolic Blood Pressure (mmHg)")
    dbp: Optional[float] = Field(default=80.0, ge=20.0, le=200.0, description="Diastolic Blood Pressure (mmHg)")
    hr: Optional[float] = Field(default=75.0, ge=20.0, le=280.0, description="Heart Rate (beats/min)")
    rr: Optional[float] = Field(default=16.0, ge=4.0, le=70.0, description="Respiratory Rate (breaths/min)")
    bt: Optional[float] = Field(default=36.8, ge=25.0, le=44.0, description="Body Temperature (°C)")
    saturation: Optional[float] = Field(default=98.0, ge=40.0, le=100.0, description="Pulse Oximetry SpO2 (%)")
    pain: Optional[int] = Field(default=0, ge=0, le=1, description="Binary pain flag (0=No, 1=Yes)")
    nrs_pain: Optional[float] = Field(default=0.0, ge=0.0, le=10.0, description="Numeric Pain Rating Scale (0 to 10)")

    # Model architecture selector
    model_architecture: str = Field(
        default="adaptive",
        description="Inference model: 'adaptive' (E0 baseline), 'rccf_v2' (RCCF-Net v2), or 'calibrated_v2' (E3)",
    )

    @field_validator("sex")
    def validate_sex(cls, v: str) -> str:
        clean = v.strip().capitalize()
        if clean not in ["Male", "Female"]:
            return "Male"
        return clean

    @field_validator("arrival_mode")
    def validate_arrival(cls, v: str) -> str:
        valid_modes = ["Walking", "119", "Private car", "Other", "Ambulance"]
        for m in valid_modes:
            if m.lower() in v.lower():
                return m
        return "Walking"


class BatchIntakeRequest(BaseModel):
    """Batch patient intake request for ED queue triage."""
    patients: List[PatientIntakeRequest] = Field(..., max_length=50)


# ==============================================================================
# 2. EVIDENCE & CONFLICT SCHEMAS
# ==============================================================================
class ConflictItem(BaseModel):
    conflict_id: str
    category: str
    severity: str
    modalities_involved: List[str]
    description: str
    clinical_risk: str
    recommended_action: str


class CorroborationItem(BaseModel):
    corroboration_id: str
    domain: str
    modalities: List[str]
    finding: str
    acuity_implication: str


class EvidenceReportResponse(BaseModel):
    overall_concordance_score: float
    conflict_severity: str
    requires_human_review: bool
    human_review_priority: str
    conflicts: List[ConflictItem]
    linked_corroborations: List[CorroborationItem]
    missing_modalities: List[str]
    sensor_dropout_count: int
    physiological_alerts: List[str]
    transcription_quality_flag: Optional[str] = None
    radiographic_escalation_flag: Optional[str] = None


# ==============================================================================
# 3. PREDICTION & SAFETY SCHEMAS
# ==============================================================================
class ClinicalSafetyVerdictResponse(BaseModel):
    verdict: str
    is_safe_to_automate: bool
    requires_human_review: bool
    predicted_ktas: int
    calibrated_confidence: float
    predictive_entropy: float
    primary_hazard_flag: Optional[str] = None
    audit_trail: List[str]


class TriagePredictionResponse(BaseModel):
    patient_id: str
    predicted_ktas_level: int
    ktas_level_name: str
    ktas_description: str
    confidence: float
    predictive_entropy: float
    class_probabilities: Dict[str, float]
    modality_weights: Dict[str, float]
    architecture_used: str
    inference_latency_ms: float
    safety_verdict: ClinicalSafetyVerdictResponse
    cece_evidence_summary: Optional[EvidenceReportResponse] = None


# ==============================================================================
# 4. VISION & QUALITY SCHEMAS
# ==============================================================================
class ImageQualityReportResponse(BaseModel):
    is_valid: bool
    status: str
    blur_score: float
    is_blurry: bool
    rms_contrast: float
    michelson_contrast: float
    is_low_contrast: bool
    underexposure_pct: float
    overexposure_pct: float
    exposure_status: str
    is_color_natural_photo: bool
    resolution: List[int]
    aspect_ratio: float
    rejection_reasons: List[str]


class PathologyPredictionItem(BaseModel):
    finding: str
    probability: float
    detected: bool
    threshold: float
    epistemic_uncertainty: float
    ci_lower: float
    ci_upper: float
    clinical_urgency_weight: float


class VisionAnalysisResponse(BaseModel):
    quality_report: ImageQualityReportResponse
    evaluated: bool
    primary_finding: Optional[str]
    max_probability: float
    radiographic_severity_index: float
    requires_emergency_escalation: bool
    escalation_finding: Optional[str] = None
    escalation_recommendation: Optional[str] = None
    predictions: List[PathologyPredictionItem]
    mean_predictive_entropy: float
    is_ood: bool
    ood_score: float
    focal_regions: List[Dict[str, Any]]
    cam_overlay_base64: Optional[str] = None


# ==============================================================================
# 5. VOICE TRANSCRIPTION SCHEMAS
# ==============================================================================
class VoiceTranscriptionResponse(BaseModel):
    raw_transcript: str
    cleaned_transcript: str
    language: str
    duration_sec: float
    confidence: float
    avg_logprob: float
    clinical_evidence: Dict[str, Any]


# ==============================================================================
# 6. SYSTEM HEALTH & VERSION SCHEMAS
# ==============================================================================
class ModelMetadata(BaseModel):
    model_name: str
    checkpoint_file: str
    checkpoint_size_kb: float
    device: str
    loaded: bool
    architecture: str


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    device: str
    models_status: Dict[str, bool]
    timestamp: str


class ModelVersionResponse(BaseModel):
    api_version: str
    project_name: str
    models: List[ModelMetadata]
    baselines: Dict[str, Any]

