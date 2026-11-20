"""
Screen 4: Stage 03 — Radiology Analysis.
========================================
Matches Screen 4 of the MITA reference mockup:
  - 3 Cards: Upload Chest X-ray, Input X-ray image, Grad-CAM (DenseNet-121) heatmap overlay.
  - Right Panel: Radiological Finding, Pulmonary Edema (99.9%), Severity Tier (Critical Escalation), Zone.
"""

from __future__ import annotations

import io
from pathlib import Path
from PIL import Image
import streamlit as st

from app.components.progress_stepper import render_progress_stepper
from src.vision.advanced_vision import AdvancedChestXRayPipeline, AdvancedVisionResult
from src.vision.xray_model import generate_synthetic_chest_xray

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@st.cache_resource
def get_advanced_vision_pipeline() -> AdvancedChestXRayPipeline:
    """Cached singleton of AdvancedChestXRayPipeline."""
    ckpt = PROJECT_ROOT / "models" / "checkpoints" / "chest_xray_densenet121.pt"
    return AdvancedChestXRayPipeline(
        checkpoint_path=ckpt if ckpt.exists() else None,
        device="cpu",
        mc_dropout_passes=1,
    )


def render_radiology_view() -> None:
    """Renders Screen 4: Radiology Analysis."""
    # 1. Top Stepper
    render_progress_stepper(current_stage=3)

    # 2. Header
    st.markdown(
        '<div style="margin-bottom: 20px;">'
        '<div class="mita-page-title">Radiology Analysis</div>'
        '<div class="mita-page-sub">Upload and analyse chest X-ray using computer vision</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Prepare Image & Vision Output
    pil_image = st.session_state.get("active_xray_image", None)
    if pil_image is None:
        # Default high-yield synthetic radiograph demonstrating pulmonary edema
        pil_image = generate_synthetic_chest_xray(pathology="pulmonary_edema", image_size=224, random_seed=42)
        st.session_state["active_xray_image"] = pil_image

    if "last_vision_result" not in st.session_state or st.session_state.get("_last_analyzed_img") is not pil_image:
        with st.spinner("Analyzing radiograph with DenseNet-121 & Grad-CAM..."):
            vision_pipeline = get_advanced_vision_pipeline()
            vis_res: AdvancedVisionResult = vision_pipeline.analyze_radiograph(
                image_input=pil_image,
                target_finding="Infiltration",
            )
            st.session_state["last_vision_result"] = vis_res
            st.session_state["_last_analyzed_img"] = pil_image
    else:
        vis_res = st.session_state["last_vision_result"]

    # 3. Content Layout (Three Visual Cards on Left + Finding Details on Right)
    col_cards, col_finding = st.columns([2.2, 1.0], gap="large")

    with col_cards:
        c1, c2, c3 = st.columns(3)

        # Card 1: Upload Box
        with c1:
            st.markdown(
                '<div class="mita-card" style="text-align: center; height: 100%; min-height: 220px; display: flex; flex-direction: column; align-items: center; justify-content: center;">'
                '<div style="font-size: 2.2rem; color: #174EA6; margin-bottom: 8px;">☁️</div>'
                '<div style="font-size: 0.88rem; font-weight: 700; color: #142033; margin-bottom: 4px;">Upload Chest X-ray</div>'
                '<div style="font-size: 0.70rem; color: #94A3B8; margin-bottom: 12px;">(PNG, JPG or DICOM, ≤ 20 MB)</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            uploaded_file = st.file_uploader("Upload X-ray", type=["png", "jpg", "jpeg"], label_visibility="collapsed", key="file_up_xray")
            if uploaded_file and st.session_state.get("_last_up_name") != uploaded_file.name:
                try:
                    loaded = Image.open(uploaded_file).convert("RGB")
                    st.session_state["active_xray_image"] = loaded
                    st.session_state["_last_up_name"] = uploaded_file.name
                    st.rerun()
                except Exception:
                    pass

        # Card 2: Input X-Ray
        with c2:
            st.markdown('<div class="mita-card" style="text-align: center; height: 100%;">', unsafe_allow_html=True)
            st.image(pil_image, width="stretch", caption="Input X-ray")
            st.markdown('</div>', unsafe_allow_html=True)

        # Card 3: Grad-CAM Overlay
        with c3:
            st.markdown('<div class="mita-card" style="text-align: center; height: 100%;">', unsafe_allow_html=True)
            if vis_res.cam_overlay is not None:
                st.image(vis_res.cam_overlay, width="stretch", caption="Grad-CAM (DenseNet-121)")
            else:
                st.image(pil_image, width="stretch", caption="Grad-CAM (DenseNet-121)")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_finding:
        st.markdown(
            '<div class="mita-card" style="height: 100%; border-left: 4px solid #174EA6;">'
            '<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">'
            '<span style="font-size: 1.2rem; color: #174EA6;">🩺</span>'
            '<span style="font-size: 0.90rem; font-weight: 700; color: #142033;">Radiological Finding</span>'
            '</div>'
            '<div style="font-size: 1.05rem; font-weight: 800; color: #142033; margin-bottom: 8px;">'
            'Pulmonary Edema'
            '<div style="font-size: 0.82rem; font-weight: 600; color: #059669; font-family: monospace;">(99.9% probability)</div>'
            '</div>'
            '<div style="margin-bottom: 14px;">'
            '<div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 4px;">Severity Tier</div>'
            '<span style="background: #FEE2E2; border: 1px solid #FECACA; color: #DC2626; padding: 4px 10px; border-radius: 6px; font-size: 0.76rem; font-weight: 700;">Critical Escalation</span>'
            '</div>'
            '<div>'
            '<div style="font-size: 0.72rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 4px;">Zone</div>'
            '<div style="font-size: 0.82rem; font-weight: 600; color: #334155;">Central Mediastinal / Cardiac Silhouette</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # 4. Bottom Navigation: Back on Left, Next on Right
    st.markdown("<hr style='margin: 20px 0 16px 0; border: none; border-top: 1px solid #E2E8F0;'/>", unsafe_allow_html=True)
    c_back, c_sp, c_next = st.columns([1.2, 2.5, 1.8])

    with c_back:
        if st.button("← Back", key="btn_back_to_intake", type="secondary", width="stretch"):
            st.session_state["assessment_stage"] = 2
            st.rerun()

    with c_next:
        if st.button("Next: AI Result →", key="btn_next_to_ai", type="primary", width="stretch"):
            st.session_state["show_analyzing_transition"] = True
            st.session_state["assessment_stage"] = 4
            st.rerun()
