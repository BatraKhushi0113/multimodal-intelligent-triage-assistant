"""
Cross-Modal Evidence & Conflict Engine (CECE) Routes.
=====================================================
Module: app/api/routes/cece.py
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.auth import AuthenticatedUser, get_current_user
from app.api.schemas import EvidenceReportResponse, PatientIntakeRequest
from app.api.services import CentralizedInferenceService

router = APIRouter(prefix="/cece", tags=["Cross-Modal Evidence & Conflict Engine"])


def get_inference_service() -> CentralizedInferenceService:
    from app.api.main import app_state
    return app_state["inference_service"]


@router.post(
    "/audit",
    response_model=EvidenceReportResponse,
    summary="Audit Cross-Modal Evidence & Detect Clinical Conflicts",
    description="Audits physiological plausibility, detects semantic contradictions across modalities, and produces structured clinical evidence reports.",
)
async def audit_evidence(
    intake: PatientIntakeRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    service: CentralizedInferenceService = Depends(get_inference_service),
) -> EvidenceReportResponse:
    try:
        return service.audit_evidence(intake)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing CECE audit: {str(e)}",
        )

