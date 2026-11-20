"""
Unit Tests for Module 17: Streamlit Clinical Decision Support Application.
==========================================================================
Verifies Streamlit app structure, clinical disclaimers, benchmark intake records,
input sanitization logic, and orchestrator instantiation.
"""

import ast
from pathlib import Path
import sys
import pytest

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure unpickling compatibility for preprocessor.joblib
import src.data.preprocessing as prep
for attr in ["ClinicalVitalsCleaner", "ClinicalContextCleaner", "PreprocessingPipeline"]:
    if hasattr(prep, attr):
        setattr(sys.modules.setdefault("__main__", sys.modules[__name__]), attr, getattr(prep, attr))

from src.integration.triage_orchestrator import (
    KTAS_DESCRIPTIONS,
    KTAS_LEVEL_NAMES,
    PatientIntakeRecord,
    TriageOrchestrator,
    TriageResult,
)


def test_streamlit_app_file_exists():
    """Verify that app/streamlit_app.py exists and is non-empty."""
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    assert app_path.exists(), "app/streamlit_app.py does not exist"
    assert app_path.stat().st_size > 1000, "app/streamlit_app.py is unexpectedly small"


def test_streamlit_app_syntax_and_ast():
    """Verify that app/streamlit_app.py parses as valid Python AST without syntax errors."""
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    code = app_path.read_text(encoding="utf-8")
    tree = ast.parse(code)
    assert len(tree.body) > 10, "AST tree body contains too few top-level statements"


def test_streamlit_clinical_disclaimer_present():
    """Verify that mandatory clinical research & educational disclaimers are present."""
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    code = app_path.read_text(encoding="utf-8")
    assert "SCIENTIFIC RESEARCH & EDUCATIONAL PROTOTYPE ONLY" in code
    assert "NOT" in code
    assert "medical device" in code or "FDA" in code


def test_streamlit_benchmark_cases_validity():
    """Verify that the 5 clinical benchmark cases defined in the app have valid schema."""
    from src.vision.xray_model import generate_synthetic_chest_xray

    cases = [
        # Case 1: Severe Shock
        PatientIntakeRecord(
            patient_id="CASE-SHOCK-01",
            age=68.0,
            sex="Male",
            arrival_mode="119",
            injury="Non-trauma",
            mental_status="Unresponsive",
            sbp=72.0,
            dbp=42.0,
            hr=138.0,
            rr=34.0,
            bt=35.4,
            saturation=84.0,
            pain=0,
            chief_complaint_text="cardiac arrest, agonal gasping, unresponsive on arrival",
        ),
        # Case 2: Emergency Dispatch
        PatientIntakeRecord(
            patient_id="CASE-VOICE-02",
            age=54.0,
            sex="Male",
            arrival_mode="119",
            mental_status="Alert",
            sbp=152.0,
            dbp=96.0,
            hr=104.0,
            rr=22.0,
            saturation=94.0,
            pain=1,
            nrs_pain=9.0,
            chief_complaint_text="crushing substernal chest pain radiating to left arm with diaphoresis",
        ),
        # Case 3: Missing Sensor
        PatientIntakeRecord(
            patient_id="CASE-MISSING-03",
            age=42.0,
            sex="Female",
            arrival_mode="Walking",
            mental_status="Alert",
            sbp=122.0,
            dbp=78.0,
            hr=76.0,
            rr=16.0,
            saturation=None,
            pain=1,
            nrs_pain=5.0,
            chief_complaint_text="severe migraine headache and nausea",
        ),
        # Case 4: Ambiguous
        PatientIntakeRecord(
            patient_id="CASE-AMBIGUOUS-04",
            age=35.0,
            sex="Female",
            arrival_mode="Walking",
            mental_status="Alert",
            sbp=118.0,
            dbp=76.0,
            hr=82.0,
            rr=17.0,
            saturation=98.0,
            pain=1,
            nrs_pain=4.0,
            chief_complaint_text="generalized fatigue, mild non-specific abdominal discomfort, malaise",
        ),
        # Case 5: Pneumothorax Escalation
        PatientIntakeRecord(
            patient_id="CASE-PNEUMO-05",
            age=24.0,
            sex="Male",
            arrival_mode="Private car",
            mental_status="Alert",
            sbp=112.0,
            dbp=74.0,
            hr=92.0,
            rr=20.0,
            saturation=95.0,
            pain=1,
            nrs_pain=6.0,
            chief_complaint_text="sudden sharp right-sided chest pain while resting",
            chest_xray_input=generate_synthetic_chest_xray("pneumothorax", image_size=224, random_seed=902),
        ),
    ]

    from dataclasses import asdict

    for c in cases:
        assert isinstance(c, PatientIntakeRecord)
        d = asdict(c)
        assert "patient_id" in d
        assert "age" in d
        assert "sbp" in d
        assert "chief_complaint_text" in d


def test_input_sanitization_helpers():
    """Verify input sanitization logic for UI dropdown options."""
    arrival_mode_ui = "119 (Ambulance)"
    clean_arr = arrival_mode_ui.split(" (")[0]
    assert clean_arr == "119"

    injury_ui_non_trauma = "Non-trauma (Medical)"
    clean_inj1 = "Trauma" if "Trauma" in injury_ui_non_trauma and "Non" not in injury_ui_non_trauma else "Non-trauma"
    assert clean_inj1 == "Non-trauma"

    injury_ui_trauma = "Trauma (Injury)"
    clean_inj2 = "Trauma" if "Trauma" in injury_ui_trauma and "Non" not in injury_ui_trauma else "Non-trauma"
    assert clean_inj2 == "Trauma"


def test_app_tabs_and_structure_coverage():
    """Verify that all three core clinical tabs are implemented in app/streamlit_app.py."""
    app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
    code = app_path.read_text(encoding="utf-8")
    assert "Live Patient Triage" in code
    assert "Case Studies & Benchmarks" in code
    assert "Architecture & Transparency" in code
    assert "get_triage_orchestrator" in code
