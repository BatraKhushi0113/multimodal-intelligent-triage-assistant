"""
Unit Tests for Module 13: Voice Input & Speech-to-Text Module
=============================================================
Tests audio preprocessing, 16kHz resampling, stereo-to-mono conversion,
Whisper transcript sanitization, synthetic emergency audio generation,
and the end-to-end VoiceTriagePipeline integration.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.context_encoder import ContextEncoder
from src.models.fusion import AdaptiveReliabilityFusion
from src.models.vitals_encoder import VitalsEncoder
from src.voice.speech_to_text import (
    AudioPreprocessor,
    VoiceTriagePipeline,
    WhisperTranscriber,
    generate_synthetic_emergency_call,
)


def test_audio_preprocessor_resampling():
    """Verify audio preprocessor resamples non-16kHz audio to target 16,000 Hz."""
    preprocessor = AudioPreprocessor(target_sample_rate=16000)

    # 1 second of 44,100 Hz audio
    sr_orig = 44100
    t = np.linspace(0, 1.0, sr_orig, endpoint=False)
    synthetic_signal = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    data_resampled, sr_out = preprocessor.load_audio(synthetic_signal, source_sample_rate=sr_orig)

    assert sr_out == 16000
    assert len(data_resampled) == 16000
    assert data_resampled.dtype == np.float32
    assert np.max(np.abs(data_resampled)) <= 1.0


def test_audio_preprocessor_stereo_to_mono():
    """Verify multi-channel audio is converted to single-channel mono."""
    preprocessor = AudioPreprocessor(target_sample_rate=16000)

    # 16,000 samples of 2-channel stereo
    stereo_data = np.random.randn(16000, 2).astype(np.float32)
    mono_data, sr = preprocessor.load_audio(stereo_data, source_sample_rate=16000)

    assert mono_data.ndim == 1
    assert len(mono_data) == 16000
    assert sr == 16000


def test_clean_clinical_transcript():
    """Verify transcript sanitizer removes fillers and normalizes clinical text."""
    transcriber = WhisperTranscriber(model_name="tiny")

    messy_speech = "  Um, patient is uh, having acute like chest pain, ah, and shortness of breath.  "
    clean = transcriber.clean_clinical_transcript(messy_speech)

    assert "um" not in clean
    assert "uh" not in clean
    assert "chest pain" in clean
    assert "shortness of breath" in clean
    assert "  " not in clean


def test_synthetic_emergency_call_generation():
    """Verify synthetic call generation produces valid audio array with vocal harmonics."""
    duration = 2.5
    sr = 16000
    audio = generate_synthetic_emergency_call(duration_sec=duration, sample_rate=sr)

    expected_len = int(duration * sr)
    assert len(audio) == expected_len
    assert audio.dtype == np.float32
    assert -1.0 <= np.min(audio) <= np.max(audio) <= 1.0
    # Must have non-zero energy
    rms = np.sqrt(np.mean(audio**2))
    assert rms > 0.01


def test_voice_triage_pipeline_integration():
    """Verify end-to-end voice triage pipeline produces valid acuity and uncertainty output."""
    vitals_enc = VitalsEncoder(input_dim=10, hidden_dim=32, embedding_dim=64, dropout_prob=0.1)
    context_enc = ContextEncoder(hidden_dim=32, embedding_dim=32, dropout_prob=0.1)
    adaptive_model = AdaptiveReliabilityFusion(
        text_dim=384,
        vitals_dim=64,
        context_dim=32,
        shared_dim=64,
        num_classes=5,
    )
    transcriber = WhisperTranscriber(model_name="tiny")

    pipeline = VoiceTriagePipeline(
        transcriber=transcriber,
        adaptive_model=adaptive_model,
        vitals_encoder=vitals_enc,
        context_encoder=context_enc,
        device="cpu",
    )

    audio = generate_synthetic_emergency_call(duration_sec=1.5, sample_rate=16000)
    v_raw = torch.randn(10)
    c_cat = torch.tensor([0, 1, 0])
    c_num = torch.randn(2)

    out = pipeline.process_voice_triage(
        audio_source=audio,
        raw_vitals_tensor=v_raw,
        raw_context_cat=c_cat,
        raw_context_num=c_num,
        mock_fallback_text="acute chest pain radiating to left arm",
    )

    assert "audio_transcription" in out
    assert "predicted_ktas" in out
    assert 0 <= out["predicted_ktas"] < 5
    assert "KTAS" in out["predicted_ktas_name"]
    assert 0.0 <= out["confidence"] <= 1.0
    assert 0.0 <= out["predictive_entropy"] <= 1.0
    assert "modality_attention_weights" in out
    weights = out["modality_attention_weights"]
    total_w = weights["Text_Weight"] + weights["Vitals_Weight"] + weights["Context_Weight"]
    assert np.isclose(total_w, 1.0, atol=1e-4)
