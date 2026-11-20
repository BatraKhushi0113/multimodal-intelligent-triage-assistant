"""
Master System Integration Layer: Multimodal Triage Orchestrator.
===============================================================
Module 16: Central orchestrator unifying clinical intake across all modalities:
  1. Primary Clinical Intake (Text Chief Complaint, Physiological Vitals, Demographics).
  2. Voice-First Audio Intake (Emergency call / voice memo -> Whisper transcription).
  3. Multimodal Deep Learning Acuity Inference (MiniLM + Vitals MLP + Context Embeddings).
  4. Adaptive Reliability & Missingness Weighting (Learned cross-modal gating).
  5. Monte Carlo Dropout Uncertainty & Safety Gating (Epistemic/Aleatoric decomposition).
  6. Interpretability Synthesis (Modality weights, tabular feature impacts, key keywords).
  7. Decoupled Radiographic Auxiliary Diagnostic Evaluation (Chest X-ray + Grad-CAM XAI).
  8. Multi-Pipeline Clinical Escalation Protocols (Zero-leakage emergency acuity modifiers).

Research & Educational Prototype Only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure utf-8 encoding for standard output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# Ensure unpickling compatibility for preprocessor.joblib
import src.data.preprocessing as prep
for attr in ["ClinicalVitalsCleaner", "ClinicalContextCleaner", "PreprocessingPipeline"]:
    if hasattr(prep, attr):
        setattr(sys.modules.setdefault("__main__", sys.modules[__name__]), attr, getattr(prep, attr))

from src.models.context_encoder import ContextEncoder
from src.models.fusion import AdaptiveReliabilityFusion
from src.models.uncertainty import MCDropoutPredictor, TriageSafetyAuditor
from src.models.vitals_encoder import VitalsEncoder
from src.vision.gradcam import GradCAMPipeline
from src.vision.xray_model import (
    NIH_CHEST_XRAY_CLASSES,
    RadiographicAcuityModifier,
)
from src.voice.speech_to_text import AudioPreprocessor, WhisperTranscriber

logger = logging.getLogger(__name__)

# Standard KTAS Levels
KTAS_LEVEL_NAMES = [
    "KTAS 1 (Resuscitation)",
    "KTAS 2 (Emergent)",
    "KTAS 3 (Urgent)",
    "KTAS 4 (Less Urgent)",
    "KTAS 5 (Non-Urgent)",
]

KTAS_DESCRIPTIONS = {
    1: "Immediately life-threatening cardiopulmonary or neurologic compromise. Resuscitation required.",
    2: "Emergent condition with potential threat to life, limb, or organ function. Immediate physician exam.",
    3: "Urgent condition causing distress or moderate physiologic disturbance. Prompt assessment required.",
    4: "Less urgent condition with low potential for rapid deterioration. Routine medical evaluation.",
    5: "Non-urgent minor condition or chronic complaint without acute physiologic disturbance.",
}


@dataclass
class PatientIntakeRecord:
    """
    Standardized Clinical Patient Intake Payload.
    Accepts clinical inputs from text, tabular vitals, demographics, spoken audio, and X-ray.
    """
    patient_id: str
    age: float
    sex: str = "Male"                     # "Male" or "Female"
    arrival_mode: str = "Walking"          # "Walking", "119", "Private car", "Other"
    injury: str = "Non-trauma"             # "Trauma" or "Non-trauma"
    mental_status: str = "Alert"           # "Alert", "Verbal", "Pain", "Unresponsive"
    patients_per_hour: float = 3.5

    # Physiological Vitals
    sbp: Optional[float] = 120.0
    dbp: Optional[float] = 80.0
    hr: Optional[float] = 78.0
    rr: Optional[float] = 16.0
    bt: Optional[float] = 36.6
    saturation: Optional[float] = 98.0
    pain: Optional[int] = 0                # 0 = No pain, 1 = Pain present
    nrs_pain: Optional[float] = 0.0        # 0 to 10

    # Modality Inputs
    chief_complaint_text: Optional[str] = None
    voice_audio_path: Optional[Union[str, Path]] = None
    voice_audio_array: Optional[np.ndarray] = None
    chest_xray_input: Optional[Union[str, Path, Image.Image, np.ndarray]] = None

    # Metadata
    admission_timestamp: Optional[str] = None
    nurse_notes: Optional[str] = None


@dataclass
class TriageResult:
    """
    Unified Comprehensive Clinical Triage Result Payload.
    """
    patient_id: str
    intake_ktas_level: int                 # 1 to 5
    intake_ktas_name: str
    intake_probabilities: List[float]      # [p1, p2, p3, p4, p5]
    confidence: float

    # Uncertainty & Safety Gating
    predictive_entropy: float
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    requires_human_review: bool
    safety_alert_message: str

    # Explainability & Modality Weights
    modality_weights: Dict[str, float]     # {"Text": w_t, "Vitals": w_v, "Context": w_c}
    top_clinical_drivers: List[Dict[str, Any]]
    chief_complaint_analyzed: str
    voice_transcription_used: bool

    # Final Clinical Decision Support Synthesis
    final_ktas_level: int                  # Final level after imaging modifier
    final_ktas_name: str
    triage_status_badge: str               # "CONCORDANT_INTAKE", "ESCALATED_BY_IMAGING", etc.
    clinical_recommendation: str
    clinical_narrative: str
    escalation_reason: Optional[str] = None

    # Decoupled Radiographic Auxiliary Findings
    xray_evaluated: bool = False
    radiographic_severity_index: Optional[float] = None
    xray_acuity_tier: Optional[str] = None
    primary_xray_finding: Optional[str] = None
    xray_confidence: Optional[float] = None
    xray_anatomical_zone: Optional[str] = None
    xray_overlay_path: Optional[str] = None

    # Optional Cross-Modal Evidence & Advanced Architecture Extensions
    cece_concordance_score: Optional[float] = None
    cece_conflict_severity: Optional[str] = None
    cece_conflicts: Optional[List[Dict[str, Any]]] = None
    architecture_used: Optional[str] = "adaptive_reliability_fusion"

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to clean JSON-serializable dictionary."""
        d = asdict(self)
        # Ensure floats are rounded for readability
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
            elif isinstance(v, list) and v and isinstance(v[0], float):
                d[k] = [round(x, 4) for x in v]
        return d


class TriageOrchestrator:
    """
    Master Clinical Decision Support Orchestrator.
    Coordinates speech, text, tabular vitals, demographics, uncertainty, explainability,
    and decoupled chest radiograph interpretation into a single clinical pipeline.
    """

    def __init__(
        self,
        config_path: Union[str, Path] = "config.yaml",
        device: Optional[str] = None,
        load_whisper: bool = False,
        load_vision: bool = True,
    ) -> None:
        self.config_path = Path(config_path)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Initializing TriageOrchestrator on compute device: {self.device}")

        # 1. Load Preprocessor
        prep_path = PROJECT_ROOT / "models" / "checkpoints" / "preprocessor.joblib"
        if prep_path.exists():
            self.preprocessor = joblib.load(prep_path)
            logger.info("Loaded tabular PreprocessingPipeline.")
        else:
            self.preprocessor = None
            logger.warning("preprocessor.joblib not found; fallback heuristics will be used.")

        # 2. Load Vitals Encoder
        self.vitals_encoder = VitalsEncoder(input_dim=10, hidden_dim=64, embedding_dim=128).to(self.device)
        vit_ckpt = PROJECT_ROOT / "models" / "checkpoints" / "vitals_encoder.pt"
        if vit_ckpt.exists():
            self.vitals_encoder.load_state_dict(torch.load(vit_ckpt, map_location=self.device, weights_only=True))
            logger.info("Loaded VitalsEncoder checkpoint.")
        self.vitals_encoder.eval()

        # 3. Load Context Encoder
        self.context_encoder = ContextEncoder().to(self.device)
        ctx_ckpt = PROJECT_ROOT / "models" / "checkpoints" / "context_encoder.pt"
        if ctx_ckpt.exists():
            self.context_encoder.load_state_dict(torch.load(ctx_ckpt, map_location=self.device, weights_only=True))
            logger.info("Loaded ContextEncoder checkpoint.")
        self.context_encoder.eval()

        # 4. Load Adaptive Reliability Fusion Model
        self.fusion_model = AdaptiveReliabilityFusion(
            text_dim=384, vitals_dim=128, context_dim=64, num_classes=5
        ).to(self.device)
        fusion_ckpt = PROJECT_ROOT / "models" / "checkpoints" / "adaptive_fusion_model.pt"
        if fusion_ckpt.exists():
            self.fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=self.device, weights_only=True))
            logger.info("Loaded AdaptiveReliabilityFusion checkpoint.")
        self.fusion_model.eval()

        # 5. Initialize Uncertainty Engine & Safety Auditor
        self.mc_predictor = MCDropoutPredictor(
            model=self.fusion_model,
            device=str(self.device),
            num_passes=20,
            num_classes=5,
        )
        self.safety_auditor = TriageSafetyAuditor(
            entropy_threshold=0.82,
            min_confidence_threshold=0.50,
            ktas1_under_triage_hazard_threshold=0.12,
            ktas2_under_triage_hazard_threshold=0.25,
        )

        # 6. Sentence Transformer for Chief Complaints (lazy initialized)
        self._sentence_transformer = None

        # 7. Whisper Voice Transcriber (lazy initialized)
        self._whisper_transcriber = None
        if load_whisper:
            self._init_whisper()

        # 8. Grad-CAM Chest X-ray Pipeline (lazy initialized)
        self._gradcam_pipeline = None
        if load_vision:
            self._init_vision()

    @property
    def sentence_transformer(self):
        """Lazy load SentenceTransformer for text embeddings."""
        if self._sentence_transformer is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
            self._sentence_transformer = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device=str(self.device)
            )
        return self._sentence_transformer

    def _init_whisper(self):
        """Initialize Whisper speech-to-text engine."""
        try:
            self._whisper_transcriber = WhisperTranscriber(model_name="tiny", device=str(self.device))
            logger.info("Initialized Whisper speech-to-text engine.")
        except Exception as e:
            logger.warning(f"Could not load Whisper model: {e}")
            self._whisper_transcriber = None

    def _init_vision(self):
        """Initialize Grad-CAM Chest X-ray pipeline."""
        xray_ckpt = PROJECT_ROOT / "models" / "checkpoints" / "chest_xray_densenet121.pt"
        if xray_ckpt.exists():
            self._gradcam_pipeline = GradCAMPipeline(
                model_architecture="densenet121",
                pretrained=False,
                checkpoint_path=xray_ckpt,
                image_size=224,
                device=str(self.device),
            )
            logger.info("Initialized GradCAMPipeline with calibrated DenseNet-121.")
        else:
            logger.warning("Chest X-ray checkpoint not found. Vision auxiliary stream disabled.")

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode chief complaint string into a 384-dimensional tensor."""
        clean_str = text.strip() if text and text.strip() else "unspecified chief complaint"
        emb = self.sentence_transformer.encode(clean_str, normalize_embeddings=True)
        return torch.tensor(emb, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _encode_tabular(self, record: PatientIntakeRecord) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess clinical vitals and demographic context into latent embeddings.
        Returns:
            z_vitals: (1, 128)
            z_context: (1, 64)
            tabular_raw_vector: (15,) for SHAP/feature attribution
        """
        # Mapping categorical features
        sex_cat = 0 if record.sex.lower().startswith("m") else 1
        arr_map = {"walking": 0, "119": 1, "private car": 2, "ambulance": 1, "other": 3}
        arr_cat = arr_map.get(record.arrival_mode.lower(), 0)
        inj_cat = 1 if "trauma" in record.injury.lower() and "non" not in record.injury.lower() else 0

        # Mapping mental status (AVPU: Alert=1, Verbal=2, Pain=3, Unresponsive=4)
        mental_map = {"alert": 1, "verbal": 2, "pain": 3, "unresponsive": 4}
        mental_val = mental_map.get(record.mental_status.lower(), 1)

        # Scale continuous numericals using reference clinical bounds
        # Reference medians and IQR from ED dataset
        sbp_val = record.sbp if record.sbp is not None else 125.0
        dbp_val = record.dbp if record.dbp is not None else 80.0
        hr_val = record.hr if record.hr is not None else 80.0
        rr_val = record.rr if record.rr is not None else 18.0
        bt_val = record.bt if record.bt is not None else 36.6
        sat_val = record.saturation if record.saturation is not None else 98.0
        pain_flag = 1.0 if (record.pain == 1 or (record.nrs_pain is not None and record.nrs_pain > 0)) else 0.0
        nrs_val = record.nrs_pain if (record.nrs_pain is not None and pain_flag == 1.0) else 0.0
        sat_missing = 1.0 if record.saturation is None else 0.0

        # Standard Robust Scaling transforms
        sbp_s = (sbp_val - 128.0) / 24.0
        dbp_s = (dbp_val - 80.0) / 16.0
        hr_s = (hr_val - 80.0) / 20.0
        rr_s = (rr_val - 18.0) / 4.0
        bt_s = (bt_val - 36.6) / 0.6
        sat_s = (sat_val - 98.0) / 2.0
        nrs_s = (nrs_val - 4.0) / 4.0
        age_s = (record.age - 50.0) / 28.0
        pph_s = (record.patients_per_hour - 3.5) / 2.0

        # Vitals vector: (1, 10)
        vit_arr = np.array([
            sat_missing, sbp_s, dbp_s, hr_s, rr_s, bt_s, sat_s, nrs_s, float(mental_val), pain_flag
        ], dtype=np.float32)

        # Context continuous: (1, 2)
        ctx_num = np.array([age_s, pph_s], dtype=np.float32)
        # Context categorical: (1, 3)
        ctx_cat = np.array([sex_cat, arr_cat, inj_cat], dtype=np.int64)

        # Combined tabular array for attribution: (15,)
        tabular_raw = np.concatenate([vit_arr, ctx_num, ctx_cat.astype(np.float32)])

        # Forward passes through deep encoders
        x_vit = torch.tensor(vit_arr, device=self.device).unsqueeze(0)
        x_num = torch.tensor(ctx_num, device=self.device).unsqueeze(0)
        x_cat = torch.tensor(ctx_cat, device=self.device).unsqueeze(0)

        with torch.no_grad():
            z_vitals = self.vitals_encoder(x_vit)
            z_context = self.context_encoder(x_cat, x_num)

        return z_vitals, z_context, torch.tensor(tabular_raw, device=self.device)

    def _estimate_clinical_drivers(
        self, record: PatientIntakeRecord, pred_ktas: int, tabular_vector: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Identify key physiological and demographic drivers influencing the triage score."""
        drivers: List[Dict[str, Any]] = []

        # 1. Severe Vital Signs Alerts
        if record.sbp is not None and (record.sbp < 90 or record.sbp > 190):
            drivers.append({
                "factor": "Blood Pressure Alert",
                "detail": f"Systolic BP is critical: {record.sbp:.0f} mmHg",
                "impact": "High Acuity Driver",
                "direction": "Escalating",
            })
        if record.hr is not None and (record.hr < 50 or record.hr > 125):
            drivers.append({
                "factor": "Heart Rate Alert",
                "detail": f"Abnormal heart rate: {record.hr:.0f} bpm (tachy/bradycardia)",
                "impact": "High Acuity Driver",
                "direction": "Escalating",
            })
        if record.saturation is not None and record.saturation < 92.0:
            drivers.append({
                "factor": "Hypoxia Alert",
                "detail": f"Severe desaturation: SpO2 {record.saturation:.0f}%",
                "impact": "Critical Acuity Driver",
                "direction": "Escalating",
            })
        if record.rr is not None and (record.rr > 26 or record.rr < 10):
            drivers.append({
                "factor": "Respiratory Rate Alert",
                "detail": f"Tachypnea / Bradypnea: {record.rr:.0f} breaths/min",
                "impact": "Urgent Acuity Driver",
                "direction": "Escalating",
            })
        if record.mental_status.lower() in ("verbal", "pain", "unresponsive"):
            drivers.append({
                "factor": "Altered Mental Status",
                "detail": f"Non-alert neurologic status: {record.mental_status}",
                "impact": "Critical Acuity Driver",
                "direction": "Escalating",
            })
        if record.arrival_mode.lower() in ("119", "ambulance"):
            drivers.append({
                "factor": "Ambulance Arrival",
                "detail": "Patient arrived via emergency medical transport (119)",
                "impact": "Acuity Driver",
                "direction": "Escalating",
            })
        if record.nrs_pain is not None and record.nrs_pain >= 7:
            drivers.append({
                "factor": "Severe Pain",
                "detail": f"Reported severe pain: NRS {record.nrs_pain:.0f}/10",
                "impact": "Symptom Burden",
                "direction": "Moderate",
            })

        # If no acute red flags, acknowledge stable physiological state
        if not drivers:
            drivers.append({
                "factor": "Stable Vital Signs",
                "detail": "Physiological measurements fall within normal emergency thresholds",
                "impact": "Baseline Acuity",
                "direction": "Stabilizing",
            })

        return drivers

    def triage_patient(
        self,
        record: Union[PatientIntakeRecord, Dict[str, Any]],
        generate_plots: bool = True,
        plots_output_dir: Optional[Union[str, Path]] = None,
        model_architecture: str = "adaptive",
        enable_cece: bool = True,
    ) -> TriageResult:
        """
        Execute full end-to-end multimodal clinical triage on a patient intake record.
        """
        start_time = time.time()

        if isinstance(record, dict):
            record = PatientIntakeRecord(**record)

        pid = record.patient_id
        logger.info(f"Initiating Multimodal Triage Orchestration for Patient: {pid}")

        # -------------------------------------------------------------
        # 1. Process Chief Complaint (Voice or Text)
        # -------------------------------------------------------------
        voice_used = False
        chief_complaint = record.chief_complaint_text or ""

        if record.voice_audio_path or record.voice_audio_array is not None:
            if self._whisper_transcriber is None:
                self._init_whisper()

            if self._whisper_transcriber is not None:
                audio_src = record.voice_audio_array if record.voice_audio_array is not None else record.voice_audio_path
                transcription_res = self._whisper_transcriber.transcribe(
                    audio_src, mock_fallback_text=chief_complaint
                )
                chief_complaint = transcription_res.get("clean_text") or chief_complaint
                voice_used = True
                logger.info(f"Transcribed voice audio into: '{chief_complaint}'")

        if not chief_complaint:
            chief_complaint = "unspecified chief complaint"

        # -------------------------------------------------------------
        # 1b. Cross-Modal Evidence & Conflict Engine (CECE) Audit
        # -------------------------------------------------------------
        cece_report = None
        if enable_cece:
            try:
                from src.integration.cece import CrossModalEvidenceConflictEngine
                if not hasattr(self, "_cece_engine") or self._cece_engine is None:
                    self._cece_engine = CrossModalEvidenceConflictEngine()
                cece_report = self._cece_engine.audit_intake(
                    chief_complaint=chief_complaint,
                    sbp=record.sbp,
                    dbp=record.dbp,
                    hr=record.hr,
                    rr=record.rr,
                    bt=record.bt,
                    saturation=record.saturation,
                    nrs_pain=record.nrs_pain,
                    pain_flag=record.pain,
                    mental_status=record.mental_status,
                    age=record.age,
                    arrival_mode=record.arrival_mode,
                    injury=record.injury,
                )
            except Exception as e:
                logger.warning(f"CECE audit skipped: {e}")

        # -------------------------------------------------------------
        # 2. Extract Latent Embeddings (Text, Vitals, Context)
        # -------------------------------------------------------------
        z_text = self._encode_text(chief_complaint)
        z_vitals, z_context, tab_vec = self._encode_tabular(record)

        # -------------------------------------------------------------
        # 3. Model Inference (Adaptive Baseline or RCCF-Net v2)
        # -------------------------------------------------------------
        if "rccf" in model_architecture.lower():
            if not hasattr(self, "_rccf_v2_model") or self._rccf_v2_model is None:
                from src.models.rccf_net_v2 import RCCFNetV2
                ckpt = PROJECT_ROOT / "models" / "checkpoints" / "rccf_net_v2_model.pt"
                self._rccf_v2_model = RCCFNetV2(
                    text_dim=384, vitals_dim=128, context_dim=64, cece_dim=8, shared_dim=128, num_classes=5
                ).to(self.device)
                if ckpt.exists():
                    st = torch.load(ckpt, map_location=self.device, weights_only=True)
                    self._rccf_v2_model.load_state_dict(st["model_state_dict"])
                self._rccf_v2_model.eval()

            e_vec = cece_report.cece_conditioning_vector if cece_report else [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            e_tensor = torch.tensor(e_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits, aux = self._rccf_v2_model(
                    text=z_text,
                    vitals=z_vitals,
                    context=z_context,
                    cece_evidence=e_tensor,
                    return_weights=True,
                )
                mean_probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                m_weights = aux["modality_weights"][0].cpu().numpy()
                modality_weights_dict = {
                    "Text": float(m_weights[0]),
                    "Vitals": float(m_weights[1]),
                    "Context": float(m_weights[2]),
                }
                pred_idx = int(np.argmax(mean_probs))
                intake_ktas = pred_idx + 1
                top1_conf = float(mean_probs[pred_idx])
                eps = 1e-12
                entropy_raw = -np.sum(mean_probs * np.log(mean_probs + eps))
                norm_entropy = float(entropy_raw / np.log(5.0))
                epistemic = 0.05
                aleatoric = 0.10
                arch_used = "rccf_net_v2"
        else:
            # Baseline Adaptive Reliability Fusion (Module 9) with MCDO
            stochastic_res = self.mc_predictor.predict_stochastic_batch(
                text=z_text,
                vitals=z_vitals,
                context=z_context,
            )

            mean_probs = stochastic_res["mean_probs"][0].cpu().numpy()
            pred_idx = int(np.argmax(mean_probs))   # 0 to 4 (corresponding to KTAS 1 to 5)
            intake_ktas = pred_idx + 1              # 1-indexed clinical KTAS level
            top1_conf = float(mean_probs[pred_idx])

            # Decompose uncertainty metrics
            p_samples = stochastic_res["p_samples"].cpu().numpy()  # (T, 1, 5)
            eps = 1e-12
            entropy_raw = -np.sum(mean_probs * np.log(mean_probs + eps))
            norm_entropy = float(entropy_raw / np.log(5.0))

            per_sample_entropy = -np.sum(p_samples * np.log(p_samples + eps), axis=-1)  # (T, 1)
            expected_sample_entropy = np.mean(per_sample_entropy, axis=0)[0]
            epistemic = float(max(0.0, entropy_raw - expected_sample_entropy))
            aleatoric = float(expected_sample_entropy)

            # Modality reliability weights from adaptive cross-modal gate
            if "modality_weights" in stochastic_res:
                m_weights = stochastic_res["modality_weights"].mean(dim=0)[0].cpu().numpy()
            else:
                with torch.no_grad():
                    _, weights_tensor = self.fusion_model(z_text, z_vitals, z_context, return_weights=True)
                    m_weights = weights_tensor[0].cpu().numpy()

            modality_weights_dict = {
                "Text": float(m_weights[0]),
                "Vitals": float(m_weights[1]),
                "Context": float(m_weights[2]),
            }
            arch_used = "adaptive_reliability_fusion"

        # -------------------------------------------------------------
        # 4. Clinical Safety Audit
        # -------------------------------------------------------------
        safety_eval = self.safety_auditor.audit_patient(
            pred_class=pred_idx,
            probabilities=mean_probs,
            normalized_entropy=norm_entropy,
            epistemic_uncertainty=epistemic,
        )
        requires_review = bool(safety_eval.get("is_flagged", False))
        reasons_list = list(safety_eval.get("reasons", []))

        # Check if CECE detected critical/moderate conflict
        if cece_report and cece_report.requires_human_review:
            requires_review = True
            for c in cece_report.conflicts:
                reasons_list.append(f"CECE Conflict [{c.conflict_id}]: {c.description}")

        alert_msg = "; ".join(reasons_list) if reasons_list else "Routine triage evaluation cleared."

        # -------------------------------------------------------------
        # 5. Explainability & Key Clinical Drivers
        # -------------------------------------------------------------
        clinical_drivers = self._estimate_clinical_drivers(record, intake_ktas, tab_vec)

        # -------------------------------------------------------------
        # 6. Decoupled Radiographic Auxiliary Evaluation (Chest X-ray)
        # -------------------------------------------------------------
        xray_evaluated = False
        rsi: Optional[float] = None
        xray_acuity: Optional[str] = None
        primary_finding: Optional[str] = None
        xray_conf: Optional[float] = None
        xray_zone: Optional[str] = None
        xray_overlay_file: Optional[str] = None

        if record.chest_xray_input is not None and self._gradcam_pipeline is not None:
            plots_dir = Path(plots_output_dir or (PROJECT_ROOT / "outputs" / "plots" / "orchestration"))
            plots_dir.mkdir(parents=True, exist_ok=True)
            overlay_save_path = plots_dir / f"{pid}_xray_gradcam.png"

            xray_res = self._gradcam_pipeline.analyze_and_explain(
                image_input=record.chest_xray_input,
                target_finding=None,
                save_plot_path=overlay_save_path if generate_plots else None,
            )

            xray_evaluated = True
            rsi = xray_res["radiographic_severity_index"]
            xray_acuity = xray_res["acuity_tier"]
            primary_finding = xray_res["target_finding"]
            xray_conf = xray_res["target_confidence"]
            xray_zone = xray_res["primary_anatomical_zone"]
            xray_overlay_file = str(overlay_save_path.relative_to(PROJECT_ROOT)) if generate_plots else None

        # -------------------------------------------------------------
        # 7. Clinical Triage Acuity Synthesis & Escalation Protocols
        # -------------------------------------------------------------
        final_ktas = intake_ktas
        escalation_reason = None
        triage_status_badge = "CONCORDANT_INTAKE"

        if xray_evaluated and xray_acuity is not None:
            if xray_acuity == "CRITICAL_ESCALATION":
                # Tension pneumothorax / flash pulmonary edema
                if intake_ktas >= 3:
                    final_ktas = 1 if (record.sbp and record.sbp < 90) or (record.saturation and record.saturation < 90) else 2
                    escalation_reason = (
                        f"Critical radiographic finding detected ({primary_finding}, {xray_conf*100:.1f}%). "
                        f"Initial intake level KTAS {intake_ktas} escalated to KTAS {final_ktas}."
                    )
                    triage_status_badge = f"ESCALATED_BY_IMAGING (KTAS {intake_ktas} -> {final_ktas})"
                else:
                    triage_status_badge = "CRITICAL_RADIOGRAPHIC_ALERT"

            elif xray_acuity == "URGENT_EVALUATION":
                # Lobar pneumonia, marked cardiomegaly
                if intake_ktas in (4, 5):
                    final_ktas = 3
                    escalation_reason = (
                        f"Acute consolidation / marked cardiomegaly ({primary_finding}, {xray_conf*100:.1f}%) "
                        f"detected. Subacute intake triage level bumped to KTAS 3 (Urgent)."
                    )
                    triage_status_badge = f"MODIFIED_BY_IMAGING (KTAS {intake_ktas} -> 3)"
                else:
                    triage_status_badge = "URGENT_RADIOGRAPHIC_FINDING"

            elif xray_acuity == "PATHOLOGY_DETECTED":
                triage_status_badge = "PATHOLOGY_NOTED"
            elif xray_acuity == "NO_ACUTE_PATHOLOGY":
                triage_status_badge = "IMAGING_CLEAR"

        # -------------------------------------------------------------
        # 8. Formulate Clinician Narrative & Recommendations
        # -------------------------------------------------------------
        ktas_name = KTAS_LEVEL_NAMES[final_ktas - 1]
        intake_ktas_name = KTAS_LEVEL_NAMES[intake_ktas - 1]

        narrative_parts = [
            f"Patient {pid} (Age {record.age:.0f}, {record.sex}) triaged to {ktas_name}.",
            f"Chief Complaint: \"{chief_complaint}\"" + (" (via Voice)" if voice_used else "") + ".",
            f"Intake Acuity Assessment: {intake_ktas_name} with {top1_conf*100:.1f}% confidence.",
            f"Adaptive Modality Weights: Text={modality_weights_dict['Text']*100:.1f}%, "
            f"Vitals={modality_weights_dict['Vitals']*100:.1f}%, "
            f"Context={modality_weights_dict['Context']*100:.1f}%.",
        ]

        if xray_evaluated:
            narrative_parts.append(
                f"Auxiliary Chest Radiograph: {xray_acuity} (RSI: {rsi:.2f}). "
                f"Primary Finding: {primary_finding} ({xray_conf*100:.1f}%, {xray_zone})."
            )

        if escalation_reason:
            narrative_parts.append(f"ESCALATION ALERT: {escalation_reason}")

        if requires_review:
            narrative_parts.append(f"SAFETY ADVISORY: {alert_msg}")

        clinical_narrative = " ".join(narrative_parts)

        # Standard clinical recommendation
        if final_ktas == 1:
            clinical_rec = (
                "IMMEDIATE RESUSCITATION REQUIRED: Direct patient to resuscitation bay immediately. "
                "Initiate high-flow oxygen, continuous cardiac monitoring, and alert attending emergency physician."
            )
        elif final_ktas == 2:
            clinical_rec = (
                "EMERGENT PHYSICIAN EVALUATION: Place in monitored bed within 15 minutes. "
                "Assess for impending hemodynamic or respiratory decompensation."
            )
        elif final_ktas == 3:
            clinical_rec = (
                "URGENT EVALUATION: Transfer to acute treatment area. Diagnostic workup and physician assessment "
                "recommended within 30 minutes."
            )
        else:
            clinical_rec = (
                "LESS URGENT / ROUTINE CARE: Patient stable for triage waiting area with vital sign re-evaluation "
                "every 60 minutes."
            )

        result = TriageResult(
            patient_id=pid,
            intake_ktas_level=intake_ktas,
            intake_ktas_name=intake_ktas_name,
            intake_probabilities=[float(p) for p in mean_probs],
            confidence=top1_conf,
            predictive_entropy=norm_entropy,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            requires_human_review=requires_review,
            safety_alert_message=alert_msg,
            modality_weights=modality_weights_dict,
            top_clinical_drivers=clinical_drivers,
            chief_complaint_analyzed=chief_complaint,
            voice_transcription_used=voice_used,
            xray_evaluated=xray_evaluated,
            radiographic_severity_index=rsi,
            xray_acuity_tier=xray_acuity,
            primary_xray_finding=primary_finding,
            xray_confidence=xray_conf,
            xray_anatomical_zone=xray_zone,
            xray_overlay_path=xray_overlay_file,
            final_ktas_level=final_ktas,
            final_ktas_name=ktas_name,
            triage_status_badge=triage_status_badge,
            escalation_reason=escalation_reason,
            clinical_recommendation=clinical_rec,
            clinical_narrative=clinical_narrative,
            cece_concordance_score=cece_report.overall_concordance_score if cece_report else None,
            cece_conflict_severity=cece_report.conflict_severity if cece_report else None,
            cece_conflicts=[asdict(c) for c in cece_report.conflicts] if cece_report else None,
            architecture_used=arch_used,
        )

        logger.info(
            f"Triage Completed for {pid}: Final {ktas_name} ({triage_status_badge}) in {time.time()-start_time:.2f}s"
        )
        return result

    def triage_batch(
        self,
        records: List[Union[PatientIntakeRecord, Dict[str, Any]]],
        generate_plots: bool = False,
    ) -> List[TriageResult]:
        """Process a collection of patient intake records in batch."""
        return [self.triage_patient(r, generate_plots=generate_plots) for r in records]
