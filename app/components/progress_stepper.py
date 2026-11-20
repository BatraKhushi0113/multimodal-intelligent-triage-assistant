"""
Horizontal Progress Stepper for MITA Clinical Workflow.
======================================================
Exact horizontal layout matching the reference mockup:
  (1) Patient Profile  ---  (2) Clinical Intake  ---  (3) Radiology Analysis  ---  (4) AI Result
"""

from __future__ import annotations

import streamlit as st

STAGES = [
    {"num": 1, "name": "Patient Profile"},
    {"num": 2, "name": "Clinical Intake"},
    {"num": 3, "name": "Radiology Analysis"},
    {"num": 4, "name": "AI Result"},
]


def render_progress_stepper(current_stage: int) -> None:
    """
    Renders horizontal progress stepper matching the MITA UI reference.
    Active stage rendered as a solid royal blue capsule (#174EA6).
    """
    elements = []
    for i, stage in enumerate(STAGES, 1):
        num = stage["num"]
        name = stage["name"]

        if i < current_stage:
            # Completed Stage (Green outline + checkmark)
            circle = '<span style="background: #ECFDF5; border: 1.5px solid #10B981; color: #059669; width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 800;">✓</span>'
            label = f'<span style="font-size: 0.82rem; font-weight: 600; color: #059669;">{name}</span>'
            item = f'<div style="display: flex; align-items: center; gap: 8px;">{circle}{label}</div>'
        elif i == current_stage:
            # Active Stage (Royal Blue Pill Capsule)
            circle = f'<span style="background: #FFFFFF; color: #174EA6; width: 20px; height: 20px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 800;">{num}</span>'
            label = f'<span style="font-size: 0.82rem; font-weight: 700; color: #FFFFFF;">{name}</span>'
            item = f'<div style="background: #174EA6; padding: 5px 14px 5px 8px; border-radius: 9999px; display: inline-flex; align-items: center; gap: 8px; box-shadow: 0 2px 6px rgba(23, 78, 166, 0.25);">{circle}{label}</div>'
        else:
            # Upcoming Stage
            circle = f'<span style="background: #FFFFFF; border: 1.5px solid #CBD5E1; color: #94A3B8; width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 700;">{num}</span>'
            label = f'<span style="font-size: 0.82rem; font-weight: 500; color: #64748B;">{name}</span>'
            item = f'<div style="display: flex; align-items: center; gap: 8px;">{circle}{label}</div>'

        elements.append(item)

    connector = '<div style="width: 48px; height: 1.5px; background: #E2E8F0; margin: 0 6px;"></div>'
    track = connector.join(elements)

    html = f'<div style="display: flex; align-items: center; justify-content: center; margin-bottom: 24px; padding: 6px 0;">{track}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_stage_navigation(
    current_stage: int,
    can_continue: bool = True,
    continue_label: str = "Next Stage",
    back_label: str = "Back",
    on_continue_callback=None,
) -> None:
    """
    Renders standard Back and Next actions matching the MITA mockup.
    """
    st.markdown("<hr style='margin: 24px 0 16px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 2.5, 1.8])

    with c1:
        if current_stage > 1:
            if st.button(f"← {back_label}", key=f"btn_nav_back_{current_stage}", type="secondary", width="stretch"):
                st.session_state["assessment_stage"] = current_stage - 1
                st.rerun()

    with c3:
        if current_stage < 4:
            if st.button(f"{continue_label} →", key=f"btn_nav_next_{current_stage}", type="primary", width="stretch", disabled=not can_continue):
                if on_continue_callback:
                    on_continue_callback()
                st.session_state["assessment_stage"] = current_stage + 1
                st.rerun()
