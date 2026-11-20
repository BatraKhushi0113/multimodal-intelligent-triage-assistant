"""
Multimodal Intelligent Triage Assistant (MITA) — Master Application.
=====================================================================
Pixel-perfect implementation of the MITA commercial clinical reference UI.
Screens:
  - Screen 1: Clean White Medical Intelligence Landing Page
  - Screen 2: Stage 01 — Patient Profile (with Anatomical Torso)
  - Screen 3: Stage 02 — Clinical Intake (Chief Complaint + 6-Tile Vitals Grid)
  - Screen 4: Stage 03 — Radiology Analysis (Upload, Input CXR, Grad-CAM, Finding Card)
  - Screen 5: Transition State — Analysing Multimodal Inputs (Progress 72% + Flowchart)
  - Screen 6: Stage 04 — Triage Result & AI Explanation (Hero Card, 3 Metrics, 5 Probs, Checklist)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Windows performance optimization & offline model enforcement
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / "logs" / ".mpl"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Unpickling compatibility for preprocessor.joblib
import src.data.preprocessing as prep
for attr in ["ClinicalVitalsCleaner", "ClinicalContextCleaner", "PreprocessingPipeline"]:
    if hasattr(prep, attr):
        setattr(sys.modules.setdefault("__main__", sys.modules[__name__]), attr, getattr(prep, attr))

import streamlit as st

# Design System & Sidebar
from app.components.design_system import MASTER_CSS
from app.components.sidebar import render_mita_sidebar

# Views matching MITA mockup screens
from app.views.architecture_view import render_architecture_view
from app.views.assessment_view import render_assessment_view
from app.views.benchmarks_view import render_benchmarks_view
from app.views.intake_view import render_intake_view
from app.views.landing_view import render_landing_view
from app.views.profile_view import render_profile_view
from app.views.radiology_view import render_radiology_view

# Streamlit Page Configuration
st.set_page_config(
    page_title="MITA — Multimodal Intelligent Triage Assistant",
    page_icon="∿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject MITA Master CSS
st.markdown(MASTER_CSS, unsafe_allow_html=True)


def initialize_session_state() -> None:
    """Ensures clinical form fields and navigation states persist across reruns."""
    st.session_state.setdefault("current_view", "landing")
    st.session_state.setdefault("assessment_stage", 1)

    # Defaults matching MITA reference screen 2
    st.session_state.setdefault("form_patient_id", "EM-LIVE-001")
    st.session_state.setdefault("form_age", 56)
    st.session_state.setdefault("form_sex", "Male")
    st.session_state.setdefault("form_arrival_mode", "Ambulance (119)")
    st.session_state.setdefault("form_injury", "Non-trauma")
    st.session_state.setdefault("form_mental_status", "Alert")
    st.session_state.setdefault("form_context", "Known HTN, arrived with severe chest discomfort")

    # Defaults matching MITA reference screen 3
    st.session_state.setdefault("form_chief_complaint", "crushing substernal chest pressure radiating to left jaw with diaphoresis")
    st.session_state.setdefault("form_sbp", 145.0)
    st.session_state.setdefault("form_dbp", 92.0)
    st.session_state.setdefault("form_hr", 104.0)
    st.session_state.setdefault("form_rr", 22.0)
    st.session_state.setdefault("form_bt", 36.8)
    st.session_state.setdefault("form_spo2", 94.0)
    st.session_state.setdefault("form_pain_score", 8)
    st.session_state.setdefault("form_pain_flag", True)
    st.session_state.setdefault("form_nrs", 8.0)

    st.session_state.setdefault("active_xray_image", None)
    st.session_state.setdefault("show_analyzing_transition", False)


initialize_session_state()


def main() -> None:
    current_view = st.session_state.get("current_view", "landing")

    # Screen 1: Full-width Landing Page
    if current_view == "landing":
        render_landing_view()

    # Inner Application Screens with Left Sidebar
    else:
        render_mita_sidebar(active_item=current_view)

        if current_view == "assessment":
            stage = int(st.session_state.get("assessment_stage", 1))
            if stage == 1:
                render_profile_view()
            elif stage == 2:
                render_intake_view()
            elif stage == 3:
                render_radiology_view()
            elif stage == 4:
                render_assessment_view()
            else:
                st.session_state["assessment_stage"] = 1
                render_profile_view()

        elif current_view == "benchmarks":
            render_benchmarks_view()

        elif current_view == "architecture":
            render_architecture_view()

        else:
            st.session_state["current_view"] = "assessment"
            render_profile_view()


if __name__ == "__main__":
    main()
