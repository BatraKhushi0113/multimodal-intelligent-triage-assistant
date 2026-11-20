"""
Voice & Speech Transcription Routes.
====================================
Module: app/api/routes/voice.py
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.auth import AuthenticatedUser, get_current_user
from app.api.schemas import VoiceTranscriptionResponse
from app.api.security import validate_audio_upload
from app.api.services import CentralizedInferenceService

router = APIRouter(prefix="/voice", tags=["Speech-to-Text Audio Intake"])


def get_inference_service() -> CentralizedInferenceService:
    from app.api.main import app_state
    return app_state["inference_service"]


@router.post(
    "/transcribe",
    response_model=VoiceTranscriptionResponse,
    summary="Transcribe Emergency Dispatch Audio",
    description="Transcribes clinical voice audio using OpenAI Whisper and runs Clinical NLP evidence extraction.",
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio recording (.wav, .mp3, .ogg)"),
    user: AuthenticatedUser = Depends(get_current_user),
    service: CentralizedInferenceService = Depends(get_inference_service),
) -> VoiceTranscriptionResponse:
    file_bytes = await file.read()
    validate_audio_upload(file, file_bytes)

    try:
        return service.transcribe_voice(file_bytes, filename=file.filename or "audio.wav")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error transcribing audio: {str(e)}",
        )

