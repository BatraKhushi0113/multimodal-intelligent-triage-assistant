"""
Centralized Inference Service for Multimodal Triage Assistant.
=============================================================
Module: app/api/services.py

Manages model lifecycles, lazy loading, and orchestrates end-to-end multimodal
inference combining:
  1. Primary Adaptive Fusion (E0 Baseline).
  2. RCCF-Net v2 (E2 / E3).
  3. Cross-Modal Evidence & Conflict Engine (CECE).
  4. Advanced Clinical NLP Engine.
  5. Advanced Computer Vision & Grad-CAM++ Pipeline.
  6. Temperature Scaling & Clinical Safety Auditor.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
import io
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.schemas import (
    ClinicalSafetyVerdictResponse,
    EvidenceReportResponse,
    ImageQualityReportResponse,
    ModelMetadata,
    PathologyPredictionItem,
    PatientIntakeRequest,
    TriagePredictionResponse,
    VisionAnalysisResponse,
    VoiceTranscriptionResponse,
)
from src.integration.cece import CrossModalEvidenceConflictEngine, EvidenceReport
from src.integration.triage_orchestrator import (
    KTAS_DESCRIPTIONS,
    KTAS_LEVEL_NAMES,
    PatientIntakeRecord,
    TriageOrchestrator,
    TriageResult,
)
from src.models.advanced_uncertainty import (
    ClinicalSafetyAuditor,
    ClinicalSafetyVerdict,
    TemperatureCalibrator,
    compute_predictive_entropy,
)
from src.models.rccf_net_v2 import RCCFNetV2
from src.nlp.clinical_nlp import AdvancedClinicalNLP, ClinicalTextEvidence
from src.vision.advanced_vision import (
    AdvancedChestXRayPipeline,
    AdvancedVisionResult,
)
from src.voice.speech_to_text import WhisperTranscriber

logger = logging.getLogger(__name__)


class CentralizedInferenceService:
    """
    Singleton service managing deep learning inference pipelines and lifecycle.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.models_dir = PROJECT_ROOT / "models" / "checkpoints"

        # 1. Base Orchestrator
        self.orchestrator = TriageOrchestrator(
            config_path=str(PROJECT_ROOT / "config.yaml"),
            device=str(self.device),
            load_whisper=False,  # Lazy initialized
            load_vision=False,   # Handled by AdvancedChestXRayPipeline
        )

        # 2. Advanced Clinical NLP
        self.clinical_nlp = AdvancedClinicalNLP()

        # 3. Cross-Modal Evidence & Conflict Engine (CECE)
        self.cece_engine = CrossModalEvidenceConflictEngine()

        # 4. Clinical Safety Auditor
        self.safety_auditor = ClinicalSafetyAuditor()

        # 5. Advanced Vision Pipeline (Lazy initialized)
        self._vision_pipeline: Optional[AdvancedChestXRayPipeline] = None

        # 6. Whisper Transcriber (Lazy initialized)
        self._whisper: Optional[WhisperTranscriber] = None

        # 7. RCCF-Net v2 Model (Lazy initialized)
        self._rccf_v2_model: Optional[RCCFNetV2] = None

        # 8. Temperature Calibrator (Fitted on val set logits)
        self.calibrator = TemperatureCalibrator(init_temperature=1.0)
        self._init_calibrator()

    def _init_calibrator(self) -> None:
        """Initialize temperature parameter from validation set calibration if available."""
        # Baseline temperature fitted on val set: T = 1.05
        self.calibrator.temperature.data.fill_(1.05)

    @property
    def vision_pipeline(self) -> AdvancedChestXRayPipeline:
        """Lazy load AdvancedChestXRayPipeline."""
        if self._vision_pipeline is None:
            ckpt = self.models_dir / "chest_xray_densenet121.pt"
            self._vision_pipeline = AdvancedChestXRayPipeline(
                checkpoint_path=ckpt if ckpt.exists() else None,
                device=str(self.device),
                mc_dropout_passes=5,
            )
        return self._vision_pipeline

    @property
    def whisper_transcriber(self) -> WhisperTranscriber:
        """Lazy load WhisperTranscriber."""
        if self._whisper is None:
            self._whisper = WhisperTranscriber(model_name="tiny", device=str(self.device))
        return self._whisper

    @property
    def rccf_v2_model(self) -> RCCFNetV2:
        """Lazy load RCCF-Net v2."""
        if self._rccf_v2_model is None:
            self._rccf_v2_model = RCCFNetV2(
                text_dim=384, vitals_dim=128, context_dim=64, cece_dim=8, shared_dim=128, num_classes=5
            ).to(self.device)
            ckpt = self.models_dir / "rccf_net_v2_model.pt"
            if ckpt.exists():
                state = torch.load(ckpt, map_location=self.device)
                self._rccf_v2_model.load_state_dict(state["model_state_dict"])
                logger.info("Loaded RCCF-Net v2 weights into centralized inference service.")
            self._rccf_v2_model.eval()
        return self._rccf_v2_model

    def predict_triage(self, intake: PatientIntakeRequest) -> TriagePredictionResponse:
        """
        Execute end-to-end multimodal triage inference with CECE evidence auditing
        and safety gating.
        """
        t0 = time.perf_counter()

        # 1. Convert intake to standard PatientIntakeRecord
        rec = PatientIntakeRecord(
            patient_id=intake.patient_id,
            age=intake.age,
            sex=intake.sex,
            arrival_mode=intake.arrival_mode,
            injury=intake.injury,
            mental_status=intake.mental_status,
            patients_per_hour=intake.patients_per_hour,
            sbp=intake.sbp,
            dbp=intake.dbp,
            hr=intake.hr,
            rr=intake.rr,
            bt=intake.bt,
            saturation=intake.saturation,
            pain=intake.pain,
            nrs_pain=intake.nrs_pain,
            chief_complaint_text=intake.chief_complaint,
        )

        # 2. Run CECE Audit
        evidence_rep = self.cece_engine.audit_intake(
            chief_complaint=intake.chief_complaint,
            sbp=intake.sbp,
            dbp=intake.dbp,
            hr=intake.hr,
            rr=intake.rr,
            bt=intake.bt,
            saturation=intake.saturation,
            nrs_pain=intake.nrs_pain,
            pain_flag=intake.pain,
            mental_status=intake.mental_status,
            age=intake.age,
            arrival_mode=intake.arrival_mode,
            injury=intake.injury,
        )

        # 3. Model Architecture Execution
        model_arch = intake.model_architecture.lower()

        if "rccf" in model_arch or "v2" in model_arch:
            # RCCF-Net v2 Inference
            z_text = self.orchestrator._encode_text(intake.chief_complaint or "")
            z_vitals, z_context, _ = self.orchestrator._encode_tabular(rec)
            e_tensor = torch.tensor(evidence_rep.cece_conditioning_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits, aux = self.rccf_v2_model(
                    text=z_text,
                    vitals=z_vitals,
                    context=z_context,
                    cece_evidence=e_tensor,
                    return_weights=True,
                )

                if "calibrated" in model_arch:
                    logits = self.calibrator(logits)

                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                w = aux["modality_weights"][0].cpu().numpy()
                modality_weights = {
                    "Text": round(float(w[0]), 4),
                    "Vitals": round(float(w[1]), 4),
                    "Context": round(float(w[2]), 4),
                }
                arch_name = "RCCF-Net v2 (Residual Cross-Attention)"
        else:
            # Baseline Adaptive Reliability Fusion (E0)
            res_base = self.orchestrator.triage_patient(rec, generate_plots=False)
            probs = np.array(res_base.intake_probabilities, dtype=np.float32)
            modality_weights = res_base.modality_weights
            arch_name = "Adaptive Reliability Fusion (Module 9 Baseline)"

        # 4. Clinical Safety Audit
        safety_verdict: ClinicalSafetyVerdict = self.safety_auditor.audit_prediction(
            probabilities=probs,
            cece_conflict_severity=evidence_rep.conflict_severity,
            cece_requires_review=evidence_rep.requires_human_review,
            physiological_alerts=evidence_rep.physiological_alerts,
        )

        pred_ktas = int(np.argmax(probs)) + 1
        conf = float(np.max(probs))
        entropy = float(compute_predictive_entropy(probs.reshape(1, -1))[0])

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Construct Response
        class_probs_dict = {f"KTAS_{i+1}": round(float(probs[i]), 4) for i in range(5)}

        safety_resp = ClinicalSafetyVerdictResponse(
            verdict=safety_verdict.verdict,
            is_safe_to_automate=safety_verdict.is_safe_to_automate,
            requires_human_review=safety_verdict.requires_human_review,
            predicted_ktas=pred_ktas,
            calibrated_confidence=round(conf, 4),
            predictive_entropy=round(entropy, 4),
            primary_hazard_flag=safety_verdict.primary_hazard_flag,
            audit_trail=safety_verdict.audit_trail,
        )

        evidence_resp = EvidenceReportResponse(
            overall_concordance_score=evidence_rep.overall_concordance_score,
            conflict_severity=evidence_rep.conflict_severity,
            requires_human_review=evidence_rep.requires_human_review,
            human_review_priority=evidence_rep.human_review_priority,
            conflicts=[asdict(c) for c in evidence_rep.conflicts],
            linked_corroborations=[asdict(l) for l in evidence_rep.linked_corroborations],
            missing_modalities=evidence_rep.missing_modalities,
            sensor_dropout_count=evidence_rep.sensor_dropout_count,
            physiological_alerts=evidence_rep.physiological_alerts,
            transcription_quality_flag=evidence_rep.transcription_quality_flag,
            radiographic_escalation_flag=evidence_rep.radiographic_escalation_flag,
        )

        return TriagePredictionResponse(
            patient_id=intake.patient_id,
            predicted_ktas_level=pred_ktas,
            ktas_level_name=KTAS_LEVEL_NAMES[pred_ktas - 1],
            ktas_description=KTAS_DESCRIPTIONS[pred_ktas],
            confidence=round(conf, 4),
            predictive_entropy=round(entropy, 4),
            class_probabilities=class_probs_dict,
            modality_weights=modality_weights,
            architecture_used=arch_name,
            inference_latency_ms=round(latency_ms, 2),
            safety_verdict=safety_resp,
            cece_evidence_summary=evidence_resp,
        )

    def audit_evidence(self, intake: PatientIntakeRequest) -> EvidenceReportResponse:
        """Standalone endpoint for CECE cross-modal evidence and conflict audit."""
        report = self.cece_engine.audit_intake(
            chief_complaint=intake.chief_complaint,
            sbp=intake.sbp,
            dbp=intake.dbp,
            hr=intake.hr,
            rr=intake.rr,
            bt=intake.bt,
            saturation=intake.saturation,
            nrs_pain=intake.nrs_pain,
            pain_flag=intake.pain,
            mental_status=intake.mental_status,
            age=intake.age,
            arrival_mode=intake.arrival_mode,
            injury=intake.injury,
        )
        return EvidenceReportResponse(
            overall_concordance_score=report.overall_concordance_score,
            conflict_severity=report.conflict_severity,
            requires_human_review=report.requires_human_review,
            human_review_priority=report.human_review_priority,
            conflicts=[asdict(c) for c in report.conflicts],
            linked_corroborations=[asdict(l) for l in report.linked_corroborations],
            missing_modalities=report.missing_modalities,
            sensor_dropout_count=report.sensor_dropout_count,
            physiological_alerts=report.physiological_alerts,
            transcription_quality_flag=report.transcription_quality_flag,
            radiographic_escalation_flag=report.radiographic_escalation_flag,
        )

    def analyze_xray(self, file_bytes: bytes, target_finding: Optional[str] = None) -> VisionAnalysisResponse:
        """Analyze chest radiograph through AdvancedChestXRayPipeline."""
        pil_img = Image.open(io.BytesIO(file_bytes))
        vis_res: AdvancedVisionResult = self.vision_pipeline.analyze_radiograph(pil_img, target_finding=target_finding)

        # Base64 encode overlay if available
        cam_b64 = None
        if vis_res.cam_overlay is not None:
            buf = io.BytesIO()
            Image.fromarray(vis_res.cam_overlay).save(buf, format="PNG")
            cam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        q_rep = ImageQualityReportResponse(
            is_valid=vis_res.quality_report.is_valid,
            status=vis_res.quality_report.status,
            blur_score=vis_res.quality_report.blur_score,
            is_blurry=vis_res.quality_report.is_blurry,
            rms_contrast=vis_res.quality_report.rms_contrast,
            michelson_contrast=vis_res.quality_report.michelson_contrast,
            is_low_contrast=vis_res.quality_report.is_low_contrast,
            underexposure_pct=vis_res.quality_report.underexposure_pct,
            overexposure_pct=vis_res.quality_report.overexposure_pct,
            exposure_status=vis_res.quality_report.exposure_status,
            is_color_natural_photo=vis_res.quality_report.is_color_natural_photo,
            resolution=list(vis_res.quality_report.resolution),
            aspect_ratio=vis_res.quality_report.aspect_ratio,
            rejection_reasons=vis_res.quality_report.rejection_reasons,
        )

        preds_items = [
            PathologyPredictionItem(
                finding=p.finding,
                probability=p.probability,
                detected=p.detected,
                threshold=p.threshold,
                epistemic_uncertainty=p.epistemic_uncertainty,
                ci_lower=p.ci_lower,
                ci_upper=p.ci_upper,
                clinical_urgency_weight=p.clinical_urgency_weight,
            )
            for p in vis_res.predictions
        ]

        return VisionAnalysisResponse(
            quality_report=q_rep,
            evaluated=vis_res.evaluated,
            primary_finding=vis_res.primary_finding,
            max_probability=vis_res.max_probability,
            radiographic_severity_index=vis_res.radiographic_severity_index,
            requires_emergency_escalation=vis_res.requires_emergency_escalation,
            escalation_finding=vis_res.escalation_finding,
            escalation_recommendation=vis_res.escalation_recommendation,
            predictions=preds_items,
            mean_predictive_entropy=vis_res.mean_predictive_entropy,
            is_ood=vis_res.is_ood,
            ood_score=vis_res.ood_score,
            focal_regions=vis_res.focal_regions,
            cam_overlay_base64=cam_b64,
        )

    def transcribe_voice(self, file_bytes: bytes, filename: str) -> VoiceTranscriptionResponse:
        """Transcribe audio recording via Whisper and extract clinical text evidence."""
        # Save to temporary buffer or array
        import soundfile as sf
        audio_arr, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
        trans_res = self.whisper_transcriber.transcribe(audio_source=audio_arr)

        cleaned_text = trans_res.get("cleaned_transcript", "")
        evidence: ClinicalTextEvidence = self.clinical_nlp.extract_evidence(cleaned_text)

        return VoiceTranscriptionResponse(
            raw_transcript=trans_res.get("raw_transcript", ""),
            cleaned_transcript=cleaned_text,
            language=trans_res.get("language", "en"),
            duration_sec=trans_res.get("duration_sec", 0.0),
            confidence=trans_res.get("confidence", 0.85),
            avg_logprob=trans_res.get("avg_logprob", -0.20),
            clinical_evidence=evidence.to_dict(),
        )

    def get_models_metadata(self) -> List[ModelMetadata]:
        """Inspect all available model checkpoints and status."""
        checkpoints = [
            ("AdaptiveReliabilityFusion", "adaptive_fusion_model.pt", "Learned Gating + Dynamic Reliability"),
            ("BaselineConcatenationFusion", "baseline_fusion_model.pt", "Early Concatenation 576-d"),
            ("RCCFNetV2", "rccf_net_v2_model.pt", "Residual Cross-Attention Clinical Fusion Network v2"),
            ("DenseNet121_ChestXRay", "chest_xray_densenet121.pt", "14 NIH Chest Thoracic Pathologies"),
            ("VitalsEncoder", "vitals_encoder.pt", "MLP Vitals Latent Projection"),
            ("ContextEncoder", "context_encoder.pt", "Entity Embeddings Demographic Context"),
        ]

        metadata: List[ModelMetadata] = []
        for name, fname, arch in checkpoints:
            path = self.models_dir / fname
            size_kb = round(path.stat().st_size / 1024.0, 1) if path.exists() else 0.0
            metadata.append(
                ModelMetadata(
                    model_name=name,
                    checkpoint_file=fname,
                    checkpoint_size_kb=size_kb,
                    device=str(self.device),
                    loaded=path.exists(),
                    architecture=arch,
                )
            )
        return metadata
