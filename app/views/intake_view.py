"""
Screen 3: Stage 02 — Clinical Intake.
=====================================
Matches Screen 3 of the MITA reference mockup:
  - Left: Chief Complaint (Text / Voice tabs, textarea, Whisper ASR waveform strip).
  - Right: Vital Signs (6-cell grid: SBP, DBP, HR, RR, Temp, SpO2) + Pain Score slider.
"""

from __future__ import annotations

import streamlit as st
from app.components.progress_stepper import render_progress_stepper


def render_intake_view() -> None:
    """Renders Screen 3: Clinical Intake."""
    # 1. Top Stepper
    render_progress_stepper(current_stage=2)

    # 2. Header
    st.markdown(
        '<div style="margin-bottom: 20px;">'
        '<div class="mita-page-title">Clinical Intake</div>'
        '<div class="mita-page-sub">Capture chief complaint, voice input and physiological measurements</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 3. Two Columns: Chief Complaint on Left, Vital Signs on Right
    col_text, col_vitals = st.columns([1.2, 1.1], gap="large")

    with col_text:
        st.markdown(
            '<div class="mita-card"><div class="mita-card-header"><span>✍️</span> Chief Complaint</div>',
            unsafe_allow_html=True,
        )

        # Ingestion Mode Tabs: Text vs Voice
        tab_t, tab_v = st.tabs(["💬  Text", "🎙️  Voice"])

        with tab_t:
            default_complaint = st.session_state.get(
                "form_chief_complaint",
                "crushing substernal chest pressure radiating to left jaw with diaphoresis",
            )
            complaint_text = st.text_area(
                "Chief Complaint Narrative",
                value=default_complaint,
                height=110,
                max_chars=500,
                label_visibility="collapsed",
                key="input_chief_complaint",
            )
            st.session_state["form_chief_complaint"] = complaint_text

            st.markdown(
                f'<div style="text-align: right; font-size: 0.74rem; color: #94A3B8; margin-top: 4px;">{len(complaint_text)} / 500</div>',
                unsafe_allow_html=True,
            )

        with tab_v:
            st.markdown(
                '<div style="font-size: 0.82rem; color: #64748B; margin-bottom: 8px;">'
                'OpenAI Whisper speech recognition for 119 emergency dispatch or audio handover:'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("Simulate Voice Dictation 🎙️", key="btn_sim_audio", type="secondary", width="stretch"):
                st.session_state["form_chief_complaint"] = "crushing substernal chest pressure radiating to left jaw with diaphoresis"
                st.rerun()

        # Voice Input (Whisper ASR) Waveform Bar (Screen 3 replica)
        waveform_svg = (
            '<svg viewBox="0 0 200 24" width="100%" height="22">'
            '<line x1="10" y1="12" x2="10" y2="4" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="18" y1="12" x2="18" y2="20" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="26" y1="12" x2="26" y2="2" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="34" y1="12" x2="34" y2="22" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="42" y1="12" x2="42" y2="6" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="50" y1="12" x2="50" y2="18" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="58" y1="12" x2="58" y2="3" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="66" y1="12" x2="66" y2="21" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="74" y1="12" x2="74" y2="7" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="82" y1="12" x2="82" y2="17" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="90" y1="12" x2="90" y2="5" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="98" y1="12" x2="98" y2="19" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="106" y1="12" x2="106" y2="2" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="114" y1="12" x2="114" y2="22" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="122" y1="12" x2="122" y2="8" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="130" y1="12" x2="130" y2="16" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="138" y1="12" x2="138" y2="4" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="146" y1="12" x2="146" y2="20" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="154" y1="12" x2="154" y2="10" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="162" y1="12" x2="162" y2="14" stroke="#2563EB" stroke-width="2.5" stroke-linecap="round"/>'
            '</svg>'
        )

        st.markdown(
            f'<div style="margin-top: 14px;">'
            f'<div style="font-size: 0.78rem; font-weight: 700; color: #142033; margin-bottom: 6px;">Voice Input (Whisper ASR)</div>'
            f'<div class="waveform-box">'
            f'<span style="color: #2563EB; font-size: 1.1rem;">🎙️</span>'
            f'{waveform_svg}'
            f'<span style="font-size: 0.76rem; font-family: monospace; color: #64748B;">00:24</span>'
            f'<span style="color: #94A3B8; font-size: 0.95rem; cursor: pointer;">🗑️</span>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_vitals:
        st.markdown(
            '<div class="mita-card"><div class="mita-card-header"><span>⚡</span> Vital Signs</div>',
            unsafe_allow_html=True,
        )

        # 6-Tile Physiological Grid
        v_r1_c1, v_r1_c2, v_r1_c3 = st.columns(3)
        with v_r1_c1:
            sbp = st.number_input("SBP (mmHg)", min_value=40.0, max_value=260.0, value=float(st.session_state.get("form_sbp", 145.0)), step=1.0, key="in_sbp")
            st.session_state["form_sbp"] = sbp

        with v_r1_c2:
            dbp = st.number_input("DBP (mmHg)", min_value=20.0, max_value=160.0, value=float(st.session_state.get("form_dbp", 92.0)), step=1.0, key="in_dbp")
            st.session_state["form_dbp"] = dbp

        with v_r1_c3:
            hr = st.number_input("HR (bpm)", min_value=20.0, max_value=220.0, value=float(st.session_state.get("form_hr", 104.0)), step=1.0, key="in_hr")
            st.session_state["form_hr"] = hr

        v_r2_c1, v_r2_c2, v_r2_c3 = st.columns(3)
        with v_r2_c1:
            rr = st.number_input("RR (/min)", min_value=6.0, max_value=60.0, value=float(st.session_state.get("form_rr", 22.0)), step=1.0, key="in_rr")
            st.session_state["form_rr"] = rr

        with v_r2_c2:
            bt = st.number_input("Temp (°C)", min_value=32.0, max_value=43.0, value=float(st.session_state.get("form_bt", 36.8)), step=0.1, key="in_bt")
            st.session_state["form_bt"] = bt

        with v_r2_c3:
            spo2 = st.number_input("SpO₂ (%)", min_value=50.0, max_value=100.0, value=float(st.session_state.get("form_spo2", 94.0)), step=1.0, key="in_spo2")
            st.session_state["form_spo2"] = spo2

        # Pain Score Slider
        st.markdown("<hr style='margin: 12px 0 8px 0; border: none; border-top: 1px solid #F1F5F9;'/>", unsafe_allow_html=True)
        pain_val = st.slider(
            "Pain Score (NRS)",
            min_value=0,
            max_value=10,
            value=int(st.session_state.get("form_pain_score", 8)),
            step=1,
            key="in_pain_score",
        )
        st.session_state["form_pain_score"] = pain_val
        st.session_state["form_nrs"] = float(pain_val)
        st.session_state["form_pain_flag"] = pain_val > 0

        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Bottom Navigation: Back on Left, Next on Right
    st.markdown("<hr style='margin: 20px 0 16px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
    c_back, c_sp, c_next = st.columns([1.2, 2.5, 1.8])

    with c_back:
        if st.button("← Back", key="btn_back_to_profile", type="secondary", width="stretch"):
            st.session_state["assessment_stage"] = 1
            st.rerun()

    with c_next:
        if st.button("Next: Radiology Analysis →", key="btn_next_to_rad", type="primary", width="stretch"):
            st.session_state["assessment_stage"] = 3
            st.rerun()
