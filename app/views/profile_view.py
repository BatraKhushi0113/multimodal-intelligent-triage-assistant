"""
Screen 2: Stage 01 — Patient Profile.
=====================================
Matches Screen 2 of the MITA reference mockup:
  - Left: Unified anatomical glowing human torso showing lungs and respiratory tree.
  - Right: Structured clinical demographic and presentation form (no empty wrapper boxes).
  - Bottom: Royal blue "Next: Clinical Intake →" action button.
"""

from __future__ import annotations

from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from app.components.progress_stepper import render_progress_stepper

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _apply_preset(preset_type: str) -> None:
    """Helper to load standardized emergency test cohorts."""
    if preset_type == "shock":
        st.session_state["form_patient_id"] = "EM-SHOCK-001"
        st.session_state["form_age"] = 72
        st.session_state["form_sex"] = "Male"
        st.session_state["form_arrival_mode"] = "Ambulance (119)"
        st.session_state["form_injury"] = "Non-trauma"
        st.session_state["form_mental_status"] = "Unresponsive"
        st.session_state["form_context"] = "Found collapsed at home, severe circulatory shock."
        st.session_state["form_chief_complaint"] = "Severe hypotension, agonal breathing, unresponsive to verbal stimuli."
        st.session_state["form_sbp"] = 68.0
        st.session_state["form_dbp"] = 38.0
        st.session_state["form_hr"] = 142.0
        st.session_state["form_rr"] = 28.0
        st.session_state["form_temp"] = 35.8
        st.session_state["form_spo2"] = 82.0
        st.session_state["form_pain_score"] = 0

    elif preset_type == "cardiac":
        st.session_state["form_patient_id"] = "EM-LIVE-001"
        st.session_state["form_age"] = 56
        st.session_state["form_sex"] = "Male"
        st.session_state["form_arrival_mode"] = "Ambulance (119)"
        st.session_state["form_injury"] = "Non-trauma"
        st.session_state["form_mental_status"] = "Alert"
        st.session_state["form_context"] = "Known HTN, arrived with severe chest discomfort"
        st.session_state["form_chief_complaint"] = "crushing substernal chest pressure radiating to left jaw with diaphoresis"
        st.session_state["form_sbp"] = 145.0
        st.session_state["form_dbp"] = 92.0
        st.session_state["form_hr"] = 104.0
        st.session_state["form_rr"] = 22.0
        st.session_state["form_temp"] = 36.8
        st.session_state["form_spo2"] = 94.0
        st.session_state["form_pain_score"] = 8


def render_profile_view() -> None:
    """Renders Screen 2: Patient Profile."""
    # 1. Top Horizontal Stepper (Active Stage: Royal Blue Capsule Pill)
    render_progress_stepper(current_stage=1)

    # 2. Header
    st.markdown(
        '<div style="margin-bottom: 20px;">'
        '<div class="mita-page-title">Patient Profile</div>'
        '<div class="mita-page-sub">Enter demographic and presentation details</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 3. Two-Column Layout (Torso Graphic on Left + Form on Right)
    col_torso, col_form = st.columns([1.0, 2.0], gap="large")

    with col_torso:
        # Anatomical human torso graphic matching Screen 2 (SVG)
        svg_path = PROJECT_ROOT / "app" / "static" / "anatomical_torso.svg"
        if svg_path.exists():
            st.image(str(svg_path), width="stretch")
        st.markdown(
            '<div style="font-size: 0.76rem; color: #0284C7; font-weight: 700; margin-top: 4px; text-align: center; letter-spacing: 0.04em;">ANATOMICAL TARGET: THORACIC</div>',
            unsafe_allow_html=True,
        )

    with col_form:
        # Row 1: Patient ID, Age, Sex
        r1_c1, r1_c2, r1_c3 = st.columns([1.2, 0.8, 1.0])
        with r1_c1:
            pid = st.text_input(
                "Patient ID *",
                value=st.session_state.get("form_patient_id", "EM-LIVE-001"),
                key="input_pid",
            )
            st.session_state["form_patient_id"] = pid

        with r1_c2:
            age = st.number_input(
                "Age *",
                min_value=1,
                max_value=120,
                value=int(st.session_state.get("form_age", 56)),
                step=1,
                key="input_age",
            )
            st.session_state["form_age"] = age

        with r1_c3:
            sex_options = ["Male", "Female"]
            cur_sex = st.session_state.get("form_sex", "Male")
            sex_idx = sex_options.index(cur_sex) if cur_sex in sex_options else 0
            sex = st.selectbox("Sex *", options=sex_options, index=sex_idx, key="input_sex")
            st.session_state["form_sex"] = sex

        # Row 2: Arrival Mode, Mental Status
        r2_c1, r2_c2 = st.columns([1.5, 1.5])
        with r2_c1:
            arr_options = ["Ambulance (119)", "Walk-in", "Private Vehicle", "Other"]
            cur_arr = st.session_state.get("form_arrival_mode", "Ambulance (119)")
            arr_idx = arr_options.index(cur_arr) if cur_arr in arr_options else 0
            arrival_mode = st.selectbox("Arrival Mode *", options=arr_options, index=arr_idx, key="input_arr")
            st.session_state["form_arrival_mode"] = arrival_mode

        with r2_c2:
            avpu_options = ["Alert", "Verbal", "Pain", "Unresponsive"]
            cur_avpu = st.session_state.get("form_mental_status", "Alert")
            avpu_idx = avpu_options.index(cur_avpu) if cur_avpu in avpu_options else 0
            mental_status = st.selectbox("Mental Status *", options=avpu_options, index=avpu_idx, key="input_avpu")
            st.session_state["form_mental_status"] = mental_status

        # Row 3: Patient Type, Presenting Context
        r3_c1, r3_c2 = st.columns([1.2, 1.8])
        with r3_c1:
            pt_options = ["Non-trauma", "Trauma"]
            cur_pt = st.session_state.get("form_injury", "Non-trauma")
            pt_idx = pt_options.index(cur_pt) if cur_pt in pt_options else 0
            ptype = st.selectbox("Patient Type", options=pt_options, index=pt_idx, key="input_ptype")
            st.session_state["form_injury"] = ptype

        with r3_c2:
            context = st.text_input(
                "Presenting Context (Optional)",
                value=st.session_state.get("form_context", "Known HTN, arrived with severe chest discomfort"),
                key="input_context",
            )
            st.session_state["form_context"] = context

    # 4. Bottom Right Action: Next: Clinical Intake
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    c_space, c_next = st.columns([3.2, 1.2])
    with c_next:
        if st.button("Next: Clinical Intake →", key="btn_next_to_intake", type="primary", width="stretch"):
            st.session_state["assessment_stage"] = 2
            st.rerun()
