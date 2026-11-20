"""
Screens 5 & 6: Stage 04 — Multimodal Analysis & Triage Result.
=============================================================
Matches Screens 5 & 6 of the MITA reference mockup:
  - Screen 5: Transition state with glowing brain icon, progress bar (72%), and model pipeline flowchart.
  - Screen 6:
      - Top: Acuity Hero (KTAS 2 Emergent, <= 15 min) + 3 Metric Cards (Confidence 77.4%, Entropy 0.46, Uncertainty Moderate).
      - Bottom: KTAS Posterior Probability (5 bars) + AI Explanation Checklist (Multimodal 4 green check items).
      - Bottom Actions: Back, View Detailed Report, Export Report.
"""

from __future__ import annotations

import io
from pathlib import Path
import time
import streamlit as st

from app.components.progress_stepper import render_progress_stepper

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _render_analyzing_transition() -> None:
    """Renders Screen 5: Analysing Multimodal Inputs (Processing Animation)."""
    render_progress_stepper(current_stage=4)

    st.markdown(
        '<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; padding: 48px 32px; text-align: center; margin-top: 20px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);">'
        '<div style="width: 80px; height: 80px; border-radius: 50%; background: #EFF6FF; border: 2px solid #BAE6FD; display: inline-flex; align-items: center; justify-content: center; font-size: 2.2rem; margin-bottom: 20px; box-shadow: 0 0 24px rgba(56, 189, 248, 0.35);">'
        '🧠'
        '</div>'
        '<h2 style="font-size: 1.6rem; font-weight: 800; color: #142033; letter-spacing: -0.02em; margin-bottom: 8px;">'
        'Analysing Multimodal Inputs'
        '</h2>'
        '<p style="font-size: 0.90rem; color: #64748B; margin-bottom: 28px;">'
        'Running NLP, CNN and Vital Signs models with multimodal fusion...'
        '</p>'
        '<div style="max-width: 480px; margin: 0 auto 32px auto; display: flex; align-items: center; gap: 14px;">'
        '<div style="flex: 1; height: 8px; background: #E2E8F0; border-radius: 9999px; overflow: hidden;">'
        '<div style="width: 72%; height: 100%; background: #174EA6; border-radius: 9999px;"></div>'
        '</div>'
        '<span style="font-size: 0.95rem; font-weight: 800; color: #142033; font-family: monospace;">72%</span>'
        '</div>'
        '<div style="display: inline-flex; align-items: center; gap: 16px; padding: 14px 24px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px;">'
        '<div style="text-align: center;"><div style="font-size: 1.4rem;">📄</div><div style="font-size: 0.72rem; font-weight: 700; color: #142033;">NLP</div><div style="font-size: 0.65rem; color: #64748B;">(MiniLM)</div></div>'
        '<div style="color: #94A3B8; font-size: 0.9rem;">→</div>'
        '<div style="text-align: center;"><div style="font-size: 1.4rem;">🖼️</div><div style="font-size: 0.72rem; font-weight: 700; color: #142033;">Vision</div><div style="font-size: 0.65rem; color: #64748B;">(DenseNet-121)</div></div>'
        '<div style="color: #94A3B8; font-size: 0.9rem;">→</div>'
        '<div style="text-align: center;"><div style="font-size: 1.4rem;">📈</div><div style="font-size: 0.72rem; font-weight: 700; color: #142033;">Vitals</div><div style="font-size: 0.65rem; color: #64748B;">(MLP)</div></div>'
        '<div style="color: #94A3B8; font-size: 0.9rem;">→</div>'
        '<div style="text-align: center;"><div style="font-size: 1.4rem;">🧠</div><div style="font-size: 0.72rem; font-weight: 700; color: #174EA6;">Fusion</div><div style="font-size: 0.65rem; color: #174EA6; font-weight: 600;">(RCCF-Net v2)</div></div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_assessment_view() -> None:
    """Renders Screen 6: Triage Result & AI Explanation (with Screen 5 transition support)."""
    # Check for temporary transition state
    if st.session_state.get("show_analyzing_transition"):
        _render_analyzing_transition()
        time.sleep(0.8)
        st.session_state["show_analyzing_transition"] = False
        st.rerun()

    # 1. Top Stepper
    render_progress_stepper(current_stage=4)

    # 2. Header
    st.markdown(
        '<div style="margin-bottom: 20px;"><div class="mita-page-title">Triage Result & AI Explanation</div></div>',
        unsafe_allow_html=True,
    )

    # Gather State
    pid = st.session_state.get("form_patient_id", "EM-LIVE-001")
    age = float(st.session_state.get("form_age", 56))
    sex = st.session_state.get("form_sex", "Male")
    sbp = float(st.session_state.get("form_sbp", 145.0))
    dbp = float(st.session_state.get("form_dbp", 92.0))
    pain_score = int(st.session_state.get("form_pain_score", 8))

    # 3. Top Row: Left Hero Card + Right 3 Metric Cards
    col_hero, col_metrics = st.columns([1.5, 1.2], gap="large")

    with col_hero:
        # Acuity Hero Card (Screen 6 replica)
        st.markdown(
            '<div style="background: #FFFFFF; border: 1.5px solid #FECACA; border-radius: 12px; padding: 24px; display: flex; align-items: center; gap: 20px; height: 100%;">'
            '<div style="color: #DC2626; font-size: 2.8rem; font-weight: 300;">⚠️</div>'
            '<div>'
            '<div style="font-size: 0.78rem; font-weight: 700; color: #DC2626; text-transform: uppercase; letter-spacing: 0.05em;">Recommended KTAS</div>'
            '<div style="font-size: 1.85rem; font-weight: 800; color: #DC2626; margin: 2px 0 4px 0;">KTAS 2 (Emergent)</div>'
            '<div style="font-size: 0.82rem; color: #64748B;">Requires prompt medical evaluation (≤ 15 minutes)</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_metrics:
        m1, m2, m3 = st.columns(3)

        # Metric 1: Model Confidence Ring
        with m1:
            ring_svg = (
                '<svg viewBox="0 0 36 36" width="62" height="62">'
                '<path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#E2E8F0" stroke-width="3"/>'
                '<path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#0EA5E9" stroke-width="3.5" stroke-dasharray="77.4, 100"/>'
                '</svg>'
            )
            st.markdown(
                f'<div class="mita-card" style="text-align: center; height: 100%; padding: 14px 10px;">'
                f'<div style="font-size: 0.70rem; font-weight: 700; color: #64748B; margin-bottom: 8px;">Model Confidence</div>'
                f'<div style="position: relative; width: 62px; height: 62px; margin: 0 auto 6px auto;">'
                f'{ring_svg}'
                f'<div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-size: 0.76rem; font-weight: 800; color: #142033; font-family: monospace;">77.4%</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Metric 2: Predictive Entropy
        with m2:
            st.markdown(
                '<div class="mita-card" style="text-align: center; height: 100%; padding: 14px 10px; display: flex; flex-direction: column; justify-content: center;">'
                '<div style="font-size: 0.70rem; font-weight: 700; color: #64748B; margin-bottom: 12px;">Predictive Entropy</div>'
                '<div style="font-size: 1.55rem; font-weight: 800; color: #142033; font-family: monospace;">0.46</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Metric 3: Uncertainty Pill
        with m3:
            st.markdown(
                '<div class="mita-card" style="text-align: center; height: 100%; padding: 14px 10px; display: flex; flex-direction: column; justify-content: center;">'
                '<div style="font-size: 0.70rem; font-weight: 700; color: #64748B; margin-bottom: 12px;">Uncertainty</div>'
                '<div>'
                '<span style="background: #EFF6FF; border: 1px solid #BFDBFE; color: #1D4ED8; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700;">Moderate</span>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # 4. Bottom Row: KTAS Posterior Probability (Left) + AI Explanation Checklist (Right)
    col_probs, col_expl = st.columns([1.2, 1.3], gap="large")

    with col_probs:
        st.markdown(
            '<div class="mita-card">'
            '<div class="mita-card-header" style="margin-bottom: 16px;">KTAS Posterior Probability</div>'
            '<div style="display: flex; flex-direction: column; gap: 10px; font-size: 0.82rem;">'
            '<div style="display: flex; align-items: center; justify-content: space-between;">'
            '<span style="width: 110px; color: #475569; font-weight: 600;">KTAS 1 (Resus)</span>'
            '<div style="flex: 1; height: 12px; background: #F1F5F9; border-radius: 4px; margin: 0 12px; overflow: hidden;"><div style="width: 2.2%; height: 100%; background: #EF4444; border-radius: 4px;"></div></div>'
            '<span style="width: 45px; text-align: right; font-family: monospace; font-weight: 700; color: #142033;">2.2%</span>'
            '</div>'
            '<div style="display: flex; align-items: center; justify-content: space-between;">'
            '<span style="width: 110px; color: #DC2626; font-weight: 700;">KTAS 2 (Emerg)</span>'
            '<div style="flex: 1; height: 12px; background: #F1F5F9; border-radius: 4px; margin: 0 12px; overflow: hidden;"><div style="width: 77.4%; height: 100%; background: #EF4444; border-radius: 4px;"></div></div>'
            '<span style="width: 45px; text-align: right; font-family: monospace; font-weight: 700; color: #DC2626;">77.4%</span>'
            '</div>'
            '<div style="display: flex; align-items: center; justify-content: space-between;">'
            '<span style="width: 110px; color: #475569; font-weight: 600;">KTAS 3 (Urgent)</span>'
            '<div style="flex: 1; height: 12px; background: #F1F5F9; border-radius: 4px; margin: 0 12px; overflow: hidden;"><div style="width: 15.5%; height: 100%; background: #F59E0B; border-radius: 4px;"></div></div>'
            '<span style="width: 45px; text-align: right; font-family: monospace; font-weight: 700; color: #142033;">15.5%</span>'
            '</div>'
            '<div style="display: flex; align-items: center; justify-content: space-between;">'
            '<span style="width: 110px; color: #475569; font-weight: 600;">KTAS 4 (Less)</span>'
            '<div style="flex: 1; height: 12px; background: #F1F5F9; border-radius: 4px; margin: 0 12px; overflow: hidden;"><div style="width: 3.7%; height: 100%; background: #3B82F6; border-radius: 4px;"></div></div>'
            '<span style="width: 45px; text-align: right; font-family: monospace; font-weight: 700; color: #142033;">3.7%</span>'
            '</div>'
            '<div style="display: flex; align-items: center; justify-content: space-between;">'
            '<span style="width: 110px; color: #475569; font-weight: 600;">KTAS 5 (Non)</span>'
            '<div style="flex: 1; height: 12px; background: #F1F5F9; border-radius: 4px; margin: 0 12px; overflow: hidden;"><div style="width: 1.3%; height: 100%; background: #10B981; border-radius: 4px;"></div></div>'
            '<span style="width: 45px; text-align: right; font-family: monospace; font-weight: 700; color: #142033;">1.3%</span>'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_expl:
        st.markdown(
            '<div class="mita-card">'
            '<div class="mita-card-header" style="margin-bottom: 16px;">AI Explanation (Multimodal)</div>'
            '<div style="display: flex; flex-direction: column; gap: 12px;">'
            '<div style="display: flex; align-items: center; gap: 10px; font-size: 0.86rem; color: #1E293B;">'
            '<span style="background: #ECFDF5; border: 1.5px solid #10B981; color: #059669; width: 20px; height: 20px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 800;">✓</span>'
            '<span>Severe chest pain reported (NRS 8/10)</span>'
            '</div>'
            '<div style="display: flex; align-items: center; gap: 10px; font-size: 0.86rem; color: #1E293B;">'
            '<span style="background: #ECFDF5; border: 1.5px solid #10B981; color: #059669; width: 20px; height: 20px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 800;">✓</span>'
            '<span>Pulmonary edema on chest X-ray (99.9%)</span>'
            '</div>'
            '<div style="display: flex; align-items: center; gap: 10px; font-size: 0.86rem; color: #1E293B;">'
            '<span style="background: #ECFDF5; border: 1.5px solid #10B981; color: #059669; width: 20px; height: 20px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 800;">✓</span>'
            '<span>Elevated blood pressure (145/92 mmHg)</span>'
            '</div>'
            '<div style="display: flex; align-items: center; gap: 10px; font-size: 0.86rem; color: #1E293B;">'
            '<span style="background: #ECFDF5; border: 1.5px solid #10B981; color: #059669; width: 20px; height: 20px; border-radius: 4px; display: inline-flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 800;">✓</span>'
            '<span>Consistent multimodal evidence for emergent triage</span>'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # 5. Bottom Navigation: Back on Left, Action Buttons on Right
    st.markdown("<hr style='margin: 24px 0 16px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
    c_b, c_s, c_rep_btn, c_exp_btn = st.columns([1.2, 1.8, 1.4, 1.4])

    with c_b:
        if st.button("← Back", key="btn_back_to_rad_res", type="secondary", width="stretch"):
            st.session_state["assessment_stage"] = 3
            st.rerun()

    with c_rep_btn:
        if st.button("👁️ View Detailed Report", key="btn_view_det_rep", type="secondary", width="stretch"):
            st.session_state["current_view"] = "architecture"
            st.rerun()

    with c_exp_btn:
        report_md = f"""# MITA Emergency Triage Decision Support Report
**Patient ID:** {pid} | **Age/Sex:** {age:.0f} / {sex}
**Recommended KTAS:** KTAS 2 (Emergent)
**Confidence:** 77.4% | **Entropy:** 0.46
**Findings:** Severe chest pain (NRS {pain_score}/10), Pulmonary edema on CXR (99.9%), BP {sbp:.0f}/{dbp:.0f} mmHg.
"""
        st.download_button(
            "📥 Export Report",
            data=report_md,
            file_name=f"MITA_Triage_Report_{pid}.md",
            mime="text/markdown",
            type="primary",
            width="stretch",
        )
