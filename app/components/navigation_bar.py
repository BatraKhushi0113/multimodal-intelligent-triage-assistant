"""
Global Clinical Navigation Bar & System Health Header.
======================================================
Provides persistent navigation across:
  - Clinical Assessment Workspace
  - Assessment History
  - Model Insights & Case Studies
  - Deep Learning Architecture & Transparency
"""

from __future__ import annotations

import streamlit as st
import torch


def render_navbar(active_view: str = "assessment") -> None:
    """
    Renders top clinical command bar with brand, live system indicators, and route switcher.
    """
    device_label = "CUDA GPU" if torch.cuda.is_available() else "CPU ENGINE"

    c_logo, c_nav, c_status = st.columns([1.8, 2.5, 1.8])

    with c_logo:
        st.markdown(
            '<div style="display: flex; align-items: center; gap: 10px;">'
            '<div style="background: #174EA6; color: white; width: 34px; height: 34px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.1rem;">✚</div>'
            '<div>'
            '<div style="font-size: 1.05rem; font-weight: 800; color: #142033; letter-spacing: -0.02em;">Multimodal Triage Assistant</div>'
            '<div style="font-size: 0.72rem; color: #64748B; font-weight: 500;">Clinical Decision Support Platform &bull; v2.0</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with c_nav:
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("Assessment", key="nav_btn_assess", width="stretch", type="primary" if active_view == "assessment" else "secondary"):
                st.session_state["current_view"] = "assessment"
                st.rerun()
        with b2:
            if st.button("Benchmarks", key="nav_btn_bench", width="stretch", type="primary" if active_view == "benchmarks" else "secondary"):
                st.session_state["current_view"] = "benchmarks"
                st.rerun()
        with b3:
            if st.button("Architecture", key="nav_btn_arch", width="stretch", type="primary" if active_view == "architecture" else "secondary"):
                st.session_state["current_view"] = "architecture"
                st.rerun()
        with b4:
            if st.button("Landing", key="nav_btn_land", width="stretch", type="secondary"):
                st.session_state["current_view"] = "landing"
                st.rerun()

    with c_status:
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end; align-items: center; gap: 8px; height: 100%;">'
            f'<a href="http://localhost:8000/docs" target="_blank" style="text-decoration: none;">'
            f'<span style="background: #EFF6FF; border: 1px solid #BFDBFE; color: #1D4ED8; padding: 4px 10px; border-radius: 6px; font-size: 0.74rem; font-weight: 600; font-family: monospace; display: inline-flex; align-items: center; gap: 4px;">● API :8000</span>'
            f'</a>'
            f'<span style="background: #F1F5F9; border: 1px solid #E2E8F0; color: #334155; padding: 4px 10px; border-radius: 6px; font-size: 0.74rem; font-weight: 600; font-family: monospace;">⚡ {device_label}</span>'
            f'<span style="background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; padding: 4px 10px; border-radius: 6px; font-size: 0.74rem; font-weight: 600;">● UI :8501</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin: 8px 0 16px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)

