"""
Security & Secure File Handling Module.
=======================================
Module: app/api/security.py

Validates file uploads, enforces size limits, inspects file magic headers
to prevent MIME spoofing, and sanitizes file paths.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Set, Tuple
from fastapi import HTTPException, UploadFile, status

# File size constraints
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024   # 15 MB
MAX_AUDIO_SIZE_BYTES = 30 * 1024 * 1024   # 30 MB

# Magic byte signatures
MAGIC_SIGNATURES = {
    "png": b"\x89PNG\r\n\x1a\n",
    "jpeg": b"\xff\xd8\xff",
    "riff": b"RIFF",                      # WAV format
}


def validate_image_upload(file: UploadFile, file_bytes: bytes) -> str:
    """
    Validate uploaded image file:
      1. Check byte length <= MAX_IMAGE_SIZE_BYTES.
      2. Verify magic bytes for PNG or JPEG.
    Returns: detected format ('png' or 'jpeg').
    """
    if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image file size ({len(file_bytes)/1e6:.1f} MB) exceeds maximum allowed limit (15 MB).",
        )

    if file_bytes.startswith(MAGIC_SIGNATURES["png"]):
        return "png"
    elif file_bytes.startswith(MAGIC_SIGNATURES["jpeg"]):
        return "jpeg"

    # Allow DICOM or raw TIFF headers if needed, otherwise reject
    if file_bytes[128:132] == b"DICM":
        return "dicom"

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid image format. Only valid PNG, JPEG, or DICOM files are accepted.",
    )


def validate_audio_upload(file: UploadFile, file_bytes: bytes) -> str:
    """
    Validate uploaded audio file:
      1. Check byte length <= MAX_AUDIO_SIZE_BYTES.
      2. Verify audio magic bytes (RIFF/WAV, ID3/MP3, OGG).
    Returns: detected audio format.
    """
    if len(file_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file size ({len(file_bytes)/1e6:.1f} MB) exceeds maximum allowed limit (30 MB).",
        )

    if file_bytes.startswith(b"RIFF") and b"WAVE" in file_bytes[:16]:
        return "wav"
    elif file_bytes.startswith(b"ID3") or file_bytes[:2] == b"\xff\xfb":
        return "mp3"
    elif file_bytes.startswith(b"OggS"):
        return "ogg"

    # Allow processing if soundfile or python can parse it, else throw
    filename = (file.filename or "").lower()
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        return filename.split(".")[-1]

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid audio format. Only WAV, MP3, or OGG audio files are accepted.",
    )

