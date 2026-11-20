"""
Multimodal Intelligent Triage Assistant - System Integration Layer
===================================================================
Module 16: Multi-pipeline orchestrator uniting clinical triage intake,
speech-to-text, uncertainty estimation, explainability, and chest radiography.
"""

from src.integration.triage_orchestrator import (
    KTAS_DESCRIPTIONS,
    KTAS_LEVEL_NAMES,
    PatientIntakeRecord,
    TriageOrchestrator,
    TriageResult,
)

__all__ = [
    "PatientIntakeRecord",
    "TriageResult",
    "TriageOrchestrator",
    "KTAS_LEVEL_NAMES",
    "KTAS_DESCRIPTIONS",
]
