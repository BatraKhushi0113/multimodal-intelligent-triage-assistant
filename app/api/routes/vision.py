"""
Computer Vision & Thoracic Radiography Routes.
==============================================
Module: app/api/routes/vision.py
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api.auth import AuthenticatedUser, get_current_user
from app.api.schemas import VisionAnalysisResponse
from app.api.security import validate_image_upload
from app.api.services import CentralizedInferenceService

router = APIRouter(prefix="/vision", tags=["Thoracic Vision & Radiography"])


def get_inference_service() -> CentralizedInferenceService:
    from app.api.main import app_state
    return app_state["inference_service"]


@router.post(
    "/analyze",
    response_model=VisionAnalysisResponse,
    summary="Upload & Analyze Chest Radiograph",
    description="Performs Image Quality Assessment (IQA), 14 NIH pathology detection, MCDO uncertainty bounds, and Grad-CAM++ anatomical localization.",
)
async def analyze_chest_xray(
    file: UploadFile = File(..., description="Chest radiograph image (PNG or JPEG)"),
    target_finding: Optional[str] = Form(None, description="Optional target finding for Grad-CAM++ saliency focus"),
    user: AuthenticatedUser = Depends(get_current_user),
    service: CentralizedInferenceService = Depends(get_inference_service),
) -> VisionAnalysisResponse:
    file_bytes = await file.read()
    validate_image_upload(file, file_bytes)

    try:
        return service.analyze_xray(file_bytes, target_finding=target_finding)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing chest radiograph: {str(e)}",
        )

