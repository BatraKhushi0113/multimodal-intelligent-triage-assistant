"""
Sidebar Navigation Component for MITA Platform.
==============================================
Persistent clinical navigation matching the left sidebar in the reference mockup.
"""

from __future__ import annotations

import streamlit as st


def render_mita_sidebar(active_item: str = "new_assessment") -> None:
    """Renders the left navigation sidebar."""
    with st.sidebar:
        # MITA Logo
        st.markdown(
            '<div class="mita-brand"><span class="mita-pulse-icon">∿</span> MITA</div>',
            unsafe_allow_html=True,
        )

        # Primary Action: New Assessment
        if st.button(
            "✚  New Assessment",
            key="sb_btn_new",
            type="primary",
            width="stretch",
            help="Start a new multimodal patient triage assessment",
        ):
            st.session_state["current_view"] = "assessment"
            st.session_state["assessment_stage"] = 1
            st.session_state.pop("last_result", None)
            st.session_state.pop("show_analyzing_transition", None)
            st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Secondary Navigation Items
        if st.button(
            "📋  Assessments",
            key="sb_btn_assessments",
            type="secondary",
            width="stretch",
            help="Review past emergency triage encounters",
        ):
            st.session_state["current_view"] = "benchmarks"
            st.rerun()

        if st.button(
            "🧠  Model Insights",
            key="sb_btn_insights",
            type="secondary",
            width="stretch",
            help="Deep learning architecture, E0-E3 benchmarks and CECE rules",
        ):
            st.session_state["current_view"] = "architecture"
            st.rerun()

        if st.button(
            "📄  Reports",
            key="sb_btn_reports",
            type="secondary",
            width="stretch",
            help="Clinical audit documentation and safety verdicts",
        ):
            st.session_state["current_view"] = "benchmarks"
            st.rerun()

        if st.button(
            "⚙️  Settings",
            key="sb_btn_settings",
            type="secondary",
            width="stretch",
            help="API connection settings and model configuration",
        ):
            st.session_state["current_view"] = "architecture"
            st.rerun()

        st.markdown("<hr style='margin: 28px 0 16px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

        # System Status Pill
        st.markdown(
            '<div style="font-size: 0.74rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;">Platform Engine:</div>'
            '<div style="display: flex; flex-direction: column; gap: 6px;">'
            '<span style="background: #EFF6FF; border: 1px solid #BFDBFE; color: #1D4ED8; padding: 4px 8px; border-radius: 6px; font-size: 0.74rem; font-weight: 600; font-family: monospace;">● FASTAPI :8000</span>'
            '<span style="background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; padding: 4px 8px; border-radius: 6px; font-size: 0.74rem; font-weight: 600;">● MITA ONLINE :8501</span>'
            '</div>',
            unsafe_allow_html=True,
        )

