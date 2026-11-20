"""
Clinical Case Study Benchmark Gallery & Cohort Viewer.
======================================================
Provides pre-calibrated emergency case studies spanning KTAS 1 to 5,
multimodal discordances, and decoupled radiographic escalations.
"""

from __future__ import annotations

import streamlit as st
from app.components.design_system import get_ktas_badge_html


def render_benchmarks_view() -> None:
    """
    Renders Clinical Case Study Benchmarks Gallery.
    """
    st.markdown(
        """
        <div style="margin-bottom: 20px;">
            <h2 style="font-size: 1.55rem; font-weight: 800; color: #142033; letter-spacing: -0.02em; margin-bottom: 4px;">
                📋 Clinical Case Study Benchmark Gallery
            </h2>
            <p style="font-size: 0.88rem; color: #64748B; margin: 0;">
                Standardized emergency presentation cohorts evaluated during research validation. Select any case to inspect telemetry and load directly into the guided assessment workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cases = [
        {
            "id": "CASE-01",
            "title": "Profound Cardiogenic Shock & Collapse",
            "tier": 1,
            "badge": "KTAS 1 (Resuscitation)",
            "summary": "Elderly male in circulatory collapse with agonal breathing, severe hypotension (68/38), and extreme Shock Index (2.09).",
            "vitals": "BP: 68/38 mmHg | HR: 142 bpm | SpO2: 82% | AVPU: Unresponsive",
            "nlp": "Affirmed: cardiac arrest, collapse, agonal respirations",
            "cece": "SI >= 1.3 Critical Shock, Narrow Pulse Pressure (30 mmHg)",
            "preset_key": "shock",
        },
        {
            "id": "CASE-02",
            "title": "Acute Coronary Syndrome (STEMI / NSTEMI)",
            "tier": 2,
            "badge": "KTAS 2 (Emergent)",
            "summary": "Middle-aged male with crushing substernal chest pressure, radiation to left arm, and diaphoresis for 45 minutes.",
            "vitals": "BP: 156/98 mmHg | HR: 108 bpm | SpO2: 94% | Pain: 9/10",
            "nlp": "Affirmed: chest pain, diaphoresis, left arm radiation | Acute: Yes (<1h)",
            "cece": "Concordant high-acuity cardiovascular presentation",
            "preset_key": "cardiac",
        },
        {
            "id": "CASE-03",
            "title": "Severe Bronchospasm & Tachypnea",
            "tier": 3,
            "badge": "KTAS 3 (Urgent)",
            "summary": "Young adult with acute asthma exacerbation, speaking in partial phrases, expiratory wheezing.",
            "vitals": "BP: 128/82 mmHg | HR: 102 bpm | RR: 26/min | SpO2: 93%",
            "nlp": "Affirmed: dyspnea, wheezing | Negated: chest pain, fever",
            "cece": "Mild respiratory distress corroborated by RR and SpO2",
            "preset_key": "asthma",
        },
        {
            "id": "CASE-04",
            "title": "Isolated Musculoskeletal Ankle Inversion",
            "tier": 4,
            "badge": "KTAS 4 (Less Urgent)",
            "summary": "Young female with sports-related ankle sprain, moderate localized swelling, hemodynamically completely stable.",
            "vitals": "BP: 118/76 mmHg | HR: 78 bpm | Temp: 36.6°C | Pain: 4/10",
            "nlp": "Affirmed: ankle pain, swelling | Negated: deformity, head trauma",
            "cece": "Concordant non-emergent presentation",
            "preset_key": "sprain",
        },
        {
            "id": "CASE-05",
            "title": "Tension Pneumothorax (Radiographic Escalation)",
            "tier": 1,
            "badge": "KTAS 1 (Escalated from 3)",
            "summary": "Young male with pleuritic chest pain whose intake was initially KTAS 3, but auxiliary chest X-ray revealed acute tension pneumothorax.",
            "vitals": "BP: 114/72 mmHg | HR: 92 bpm | RR: 20/min | SpO2: 96%",
            "nlp": "Affirmed: pleuritic chest pain | Acute: Yes",
            "cece": "Radiographic Acuity Modifier: Pneumothorax (>85% conf) forces KTAS 1 override",
            "preset_key": "pneumo",
        },
    ]

    for case in cases:
        st.markdown(
            f"""
            <div class="clinical-card">
                <div class="clinical-card-header">
                    <div>
                        <span class="card-header-title">{case['title']}</span>
                        <div class="card-header-desc">Case ID: {case['id']} &bull; Clinical Target: {case['badge']}</div>
                    </div>
                    {get_ktas_badge_html(case['tier'])}
                </div>
                <div style="font-size: 0.86rem; color: #334155; line-height: 1.5; margin-bottom: 10px;">
                    {case['summary']}
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px; font-size: 0.78rem; color: #475569; font-family: monospace; line-height: 1.6;">
                    <div><strong>Vitals Telemetry:</strong> {case['vitals']}</div>
                    <div><strong>NLP Entities:</strong> {case['nlp']}</div>
                    <div><strong>CECE Audit:</strong> {case['cece']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_act1, c_act2 = st.columns([1, 4])
        with c_act1:
            if st.button(f"Load Case {case['id']}", key=f"btn_load_{case['id']}", width="stretch", type="primary"):
                from app.views.profile_view import _apply_preset
                _apply_preset(case["preset_key"])
                st.session_state["current_view"] = "assessment"
                st.session_state["assessment_stage"] = 1
                st.rerun()

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

