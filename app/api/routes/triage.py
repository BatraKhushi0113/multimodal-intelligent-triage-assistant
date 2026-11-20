"""
Triage Acuity Prediction Routes.
=================================
Module: app/api/routes/triage.py
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.auth import AuthenticatedUser, get_current_user
from app.api.schemas import (
    BatchIntakeRequest,
    PatientIntakeRequest,
    TriagePredictionResponse,
)
from app.api.services import CentralizedInferenceService

router = APIRouter(prefix="/triage", tags=["Clinical Triage Acuity"])


def get_inference_service() -> CentralizedInferenceService:
    from app.api.main import app_state
    return app_state["inference_service"]


@router.post(
    "/predict",
    response_model=TriagePredictionResponse,
    summary="Predict Clinical Triage Acuity (KTAS 1-5)",
    description="Processes multimodal intake (text, vitals, context), runs CECE audit, and returns calibrated KTAS prediction with safety governance.",
)
async def predict_triage(
    intake: PatientIntakeRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    service: CentralizedInferenceService = Depends(get_inference_service),
) -> TriagePredictionResponse:
    try:
        return service.predict_triage(intake)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during triage prediction: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=List[TriagePredictionResponse],
    summary="Batch Triage for Emergency Queue",
)
async def batch_triage(
    batch: BatchIntakeRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    service: CentralizedInferenceService = Depends(get_inference_service),
) -> List[TriagePredictionResponse]:
    results: List[TriagePredictionResponse] = []
    for intake in batch.patients:
        try:
            results.append(service.predict_triage(intake))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing patient {intake.patient_id}: {str(e)}",
            )
    return results

