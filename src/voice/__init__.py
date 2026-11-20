"""
Multimodal Intelligent Triage Assistant - Voice Module
=====================================================
Module 13: Speech-to-text processing for verbal chief complaints using OpenAI Whisper.
"""

from src.voice.speech_to_text import (
    AudioPreprocessor,
    VoiceTriagePipeline,
    WhisperTranscriber,
    generate_synthetic_emergency_call,
    plot_audio_waveform_and_spectrogram,
    run_voice_module,
)

__all__ = [
    "AudioPreprocessor",
    "WhisperTranscriber",
    "VoiceTriagePipeline",
    "generate_synthetic_emergency_call",
    "plot_audio_waveform_and_spectrogram",
    "run_voice_module",
]
